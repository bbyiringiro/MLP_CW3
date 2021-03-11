import os
import numpy as np 
import pandas as pd 
from datetime import datetime
import time
import random
from tqdm.auto import tqdm
import argparse


#Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler

#sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold

#CV
import cv2

################# DETR FUCNTIONS FOR LOSS######################## 
import sys
sys.path.extend(['/tmp/packages/detr/'])

from models.matcher import HungarianMatcher
from models.detr import SetCriterion
#################################################################

#Albumenatations
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2

#Glob
from glob import glob

from tqdm.notebook import tqdm


# print(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


parser = argparse.ArgumentParser()
parser.add_argument('--n_folds', metavar='n_folds', type=int)
# parser.add_argument('--model_name',metavar='model_name', type=str)

args = parser.parse_args()


dim = 256
train_df = pd.read_csv(f'data/meta_data/train_combined_meta.csv')
# print(train_df)
train_df['image_path'] = f'data/vin_vig_256x256/train/'+train_df.image_id+('.png' if dim!='original' else '.jpg')

#remove normal class
train_df = train_df[train_df.class_id!=14].reset_index(drop = True)

#256 by 256
train_meta = pd.read_csv("data/resized_train_meta.csv")

train_df=train_df.merge(train_meta, left_on='image_id', right_on='image_id') 
orginal_df = train_df.copy()
# print(orginal_df)

train_df['x_min'] = train_df.apply(lambda row: (row.x_min)*(dim/row.dim1), axis =1)
train_df['y_min'] = train_df.apply(lambda row: (row.y_min)*(dim/row.dim0), axis =1)

train_df['x_max'] = train_df.apply(lambda row: (row.x_max)*(dim/row.dim1), axis =1)
train_df['y_max'] = train_df.apply(lambda row: (row.y_max)*(dim/row.dim0), axis =1)

train_df['x_mid'] = train_df.apply(lambda row: (row.x_max+row.x_min)/2, axis =1)
train_df['y_mid'] = train_df.apply(lambda row: (row.y_max+row.y_min)/2, axis =1)

train_df['w'] = train_df.apply(lambda row: row.x_max-row.x_min, axis =1)
train_df['h'] = train_df.apply(lambda row: row.y_max-row.y_min, axis =1)

train_df['area'] = train_df['w']*train_df['h']

features = ['image_id','rad_id','class_id','image_path','x_min', 'y_min', 'w', 'h','area']
needed_df = train_df[features]
needed_df=needed_df.rename(columns={"x_min": "x", "y_min": "y"})

marking = needed_df

marking['image_id']=marking.apply(lambda x:'%s_%s' % (x['image_id'],x['rad_id']),axis=1)

image_data = marking.groupby('image_id')

def get_data(img_id):
    if img_id not in image_data.groups:
        return dict(image_id=img_id, source='', boxes=list())
    
    data  = image_data.get_group(img_id)
    source = np.unique(data.image_path.values)
    assert len(source)==1, 'corrupted data: %s image_id has many sources: %s' %(img_id,source)
    source=source[0]
    boxes = data[['x','y','w','h']].values
    classes_ids=data['class_id'].values+1 ## remeber to classes ids are +1
    return dict(image_id = img_id, source=source, boxes = boxes, classes_ids=classes_ids)

image_list = [get_data(img_id) for img_id in marking['image_id'].unique()]

def add_fold_index(lst,n_folds):
    lens = [len(x['boxes']) for x in lst]
    lens_unique = np.unique(lens)
    i = np.random.randint(n_folds)
    fold_indexes = [[] for _ in range(n_folds)]
    idx = []
    
    for _l in lens_unique:
        idx.extend(np.nonzero(lens==_l)[0].tolist())
        if len(idx)<n_folds: continue
        random.shuffle(idx)
        while len(idx)>= n_folds:
            fold_indexes[i].append(lst[idx.pop()]['image_id'])
            i = (i+1) % n_folds
    while len(idx):
        fold_indexes[i].append(lst[idx.pop()]['image_id'])
        i = (i+1) % n_folds
    
    return fold_indexes
    
sources = np.unique(marking['image_path'])

splitted_image_list = {s:sorted([x for x in image_list if x['source']==s],key=lambda x: len(x['boxes'])) 
                       for s in sources}
splitted_image_list = {k: add_fold_index(v,n_folds=args.n_folds) for k,v in splitted_image_list.items()}

fold_indexes = [[] for _ in range(args.n_folds)]
for k,v in splitted_image_list.items():
    for i in range(args.n_folds):
        fold_indexes[i].extend(v[i])  
    
# print([len(v) for v in fold_indexes])

def get_train_transforms():
    return A.Compose(
        [
            A.OneOf(
            [
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, val_shift_limit=0.2, p=0.9),      
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9)
            ],
            p=0.9),         
            A.ToGray(p=0.01),         
            A.HorizontalFlip(p=0.5),         
#             A.VerticalFlip(p=0.5),         
            A.Resize(height=256, width=256, p=1), #256     
            A.Normalize(max_pixel_value=1),
            A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, fill_value=0, p=0.01),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.1),
            ToTensorV2(p=1.0)
        ], 
        p=1.0,         
        bbox_params=A.BboxParams(format='coco',min_area=0, min_visibility=0,label_fields=['labels'])
        )

def get_valid_transforms():
    return A.Compose([A.Resize(height=256, width=256, p=1.0), # 512
                      A.Normalize(max_pixel_value=1),
                      ToTensorV2(p=1.0),
                      ], 
                      p=1.0, 
                      bbox_params=A.BboxParams(format='coco',min_area=0, min_visibility=0,label_fields=['labels'])
                      )

def get_test_transforms():
    return A.Compose([A.Resize(height=256, width=256, p=1.0), # 512
                      A.Normalize(max_pixel_value=1),
                      ToTensorV2(p=1.0),
                      ]
                      )


class VinDataset(Dataset):
    def __init__(self,image_list,src_path,transforms=None):
        self.images = image_list
        self.transforms = transforms
        self.img_ids = {x['image_id']:i for i,x in enumerate(image_list)}
        self.img_src = src_path
        
    def get_indices(self,img_ids):
        return [self.img_ids[x] for x in img_ids]
        
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self,index):
        record = self.images[index]
        image_id = record['image_id']
        image_path_id = image_id.split("_")[0]
        
        # print(self.img_src,image_path_id)
        
        image = cv2.imread(f'{self.img_src}/{image_path_id}.png', cv2.IMREAD_COLOR)
#         print(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        
        # DETR takes in data in coco format 
        boxes = record['boxes'] 
        
        
        labels =  record['classes_ids']

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': boxes,
                'labels': labels
            }
            sample = self.transforms(**sample)
            image  = sample['image']
            boxes  = sample['bboxes']
            labels = sample['labels']

        _,h,w = image.shape
        boxes = A.augmentations.bbox_utils.normalize_bboxes(sample['bboxes'],rows=h,cols=w)
        
        ## detr uses center_x,center_y,width,height !!
        if len(boxes)>0:
            boxes = np.array(boxes)
#             print(boxes)
#             boxes[:,2:] /= 2
#             print(boxes)
            boxes[:,:2] += boxes[:,2:]/2
#             print(boxes)
        else:
            boxes = np.zeros((0,4))
    
        target = {}
        target['boxes'] = torch.as_tensor(boxes,dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels,dtype=torch.long)
        target['image_id'] = torch.tensor([index])
#         print(target['labels'])
        
        return image, target, image_id
    
DIR_TRAIN = 'data/vin_vig_256x256/train'
train_ds = VinDataset(image_list,DIR_TRAIN, get_train_transforms())
valid_ds = VinDataset(image_list,DIR_TRAIN, get_valid_transforms())



np.save('train_ds',train_ds,allow_pickle =False)
np.save('data_ds',train_ds,allow_pickle =False)


import os
import numpy as np 
import pandas as pd 
from datetime import datetime
import time
import random
from tqdm.auto import tqdm
import argparse
import gc


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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--n_folds', metavar='n_folds', type=int)
parser.add_argument('--model_name',metavar='model_name', type=str)
parser.add_argument('--BATCH_SIZE',metavar='BATCH_SIZE', type=int)
parser.add_argument('--num_classes', metavar='num_classes', type=int)
parser.add_argument('--EPOCHS', metavar='EPOCHS', type=int)

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

# def DETRModel(_num_classes,model_name_path=args.model_name, num_queries=50):
#     model = torch.hub.load('facebookresearch/detr', model_name_path, pretrained=False, num_classes=_num_classes, num_queries=num_queries)
#     def parameter_groups(self):
#         return { 'backbone': [p for n,p in self.named_parameters()
#                               if ('backbone' in n) and p.requires_grad],
#                  'transformer': [p for n,p in self.named_parameters() 
#                                  if (('transformer' in n) or ('input_proj' in n)) and p.requires_grad],
#                  'embed': [p for n,p in self.named_parameters()
#                                  if (('class_embed' in n) or ('bbox_embed' in n) or ('query_embed' in n)) 
#                            and p.requires_grad]}
#     setattr(type(model),'parameter_groups',parameter_groups)
#     return model

class DETRModel(nn.Module):
    def __init__(self,num_classes=14, num_queries=50):
        super(DETRModel,self).__init__()
        self.num_classes = num_classes
        #TASK need to incorporate this in our model
#         self.num_queries = num_queries # 
        
        self.model = torch.hub.load('facebookresearch/detr', args.model_name, pretrained=True)
        
        
        self.out = nn.Linear(in_features=self.model.class_embed.out_features,out_features=num_classes+1)
        
    def forward(self,images):
        d = self.model(images)
        d['pred_logits'] = self.out(d['pred_logits'])
        return d
    
    def parameter_groups(self):
        return { 
            'backbone': [p for n,p in self.model.named_parameters()
                              if ('backbone' in n) and p.requires_grad],
            'transformer': [p for n,p in self.model.named_parameters() 
                                 if (('transformer' in n) or ('input_proj' in n)) and p.requires_grad],
            'embed': [p for n,p in self.model.named_parameters()
                                 if (('class_embed' in n) or ('bbox_embed' in n) or ('query_embed' in n)) 
                           and p.requires_grad],
            'final': self.out.parameters()
            }
    
model = DETRModel()
# model.parameter_groups().keys()

matcher = HungarianMatcher(cost_giou=2,cost_class=1,cost_bbox=5)

weight_dict = {'loss_ce': 1, 'loss_bbox': 5 , 'loss_giou': 2}

losses = ['labels', 'boxes', 'cardinality']


def collate_fn(batch):
    return tuple(zip(*batch))


def get_fold(fold):
    
    #chnage num_workers to 0 for Windows acc to https://stackoverflow.com/questions/50480689/pytorch-torchvision-brokenpipeerror-errno-32-broken-pipe
    
    train_indexes = train_ds.get_indices([x for i,f in enumerate(fold_indexes) if i!=fold for x in f])
    valid_indexes = valid_ds.get_indices(fold_indexes[fold])
    
    train_data_loader = DataLoader(
        torch.utils.data.Subset(train_ds,train_indexes),
        batch_size=args.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    valid_data_loader = DataLoader(
        torch.utils.data.Subset(valid_ds,valid_indexes),
        batch_size=args.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    return train_data_loader,valid_data_loader

train_loader,valid_loader = get_fold(0)

valid_iter = iter(valid_loader)
batch  = next(valid_iter)

images,targets,image_id = batch
torch.cat([v['boxes'] for v in targets])


import util.box_ops  as box_ops

def challenge_metric(outputs,targets):
    logits = outputs['pred_logits']
    boxes  = outputs['pred_boxes']
    return sum(avg_precision(logit[:,0]-logit[:,1],box,target['boxes'])
            for logit,box,target in zip(logits,boxes,targets))/len(logits)

    return {target['image_id']:avg_precision(logit[:,0]-logit[:,1],box,target['boxes'])
            for logit,box,target in zip(logits,boxes,targets)}


@torch.no_grad()
def avg_precision(logit,pboxes,tboxes,reduce=True):
    idx = logit.gt(0)
    if sum(idx)==0 and len(tboxes)==0: 
        return 1 if reduce else [1]*6
    if sum(idx)>0 and len(tboxes)==0: 
        return 0 if reduce else [0]*6
    
    pboxes = pboxes[idx]
    logit = logit[idx]
    
    idx = logit.argsort(descending=True)
    pboxes=box_ops.box_cxcywh_to_xyxy(pboxes.detach()[idx])
    tboxes=box_ops.box_cxcywh_to_xyxy(tboxes)
    
    iou = box_ops.box_iou(pboxes,tboxes)[0].cpu().numpy()
    prec = [precision(iou,th) for th in [0.5,0.55,0.6,0.65,0.7,0.75]]
    if reduce:
        return sum(prec)/6
    return prec
    

def precision(iou,th):
    #if iou.shape==(0,0): return 1

    #if min(*iou.shape)==0: return 0
    tp = 0
    iou = iou.copy()
    num_pred,num_gt = iou.shape
    for i in range(num_pred):
        _iou = iou[i]
        n_hits = (_iou>th).sum()
        if n_hits>0:
            tp += 1
            j = np.argmax(_iou)
            iou[:,j] = 0
    return tp/(num_pred+num_gt-tp)

def gen_box(n,scale=1):
    par = torch.randn((n,4)).mul(scale).sigmoid() 
    max_hw = 2*torch.min(par[:,:2],1-par[:,:2])
    par[:,2:] = par[:,2:].min(max_hw)
    return par

pboxes = gen_box(50)
logit = torch.randn(50)
tboxes = gen_box(3) 
#iou = 
print(avg_precision(logit,pboxes,tboxes))
#iou.gt(0.5),iou,pboxes,tboxes




def train_fn(data_loader,model,criterion,optimizer,device,scheduler,epoch):
    model.train()
    criterion.train()
    
    tk0 = tqdm(data_loader, total=len(data_loader),leave=False)
    log = None
    
    for step, (images, targets, image_ids) in enumerate(tk0):
        
        batch_size = len(images)
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        

        output = model(images)
        
        loss_dict = criterion(output, targets)
        
        if log is None:
            log = {k:AverageMeter() for k in loss_dict}
            log['total_loss'] = AverageMeter()
            log['avg_prec'] = AverageMeter()
            
        weight_dict = criterion.weight_dict
        
        total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        optimizer.zero_grad()

        total_loss.backward()
        
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        log['total_loss'].update(total_loss.item(),batch_size)
        
        for k,v in loss_dict.items():
            log[k].update(v.item(),batch_size)
            
        log['avg_prec'].update(challenge_metric(output,targets),batch_size)
            
        tk0.set_postfix({k:v.avg for k,v in log.items()}) 
        
    return log



def eval_fn(data_loader, model,criterion, device):
    model.eval()
    criterion.eval()
    log = None
    
    with torch.no_grad():
        
        tk0 = tqdm(data_loader, total=len(data_loader),leave=False)
        for step, (images, targets, image_ids) in enumerate(tk0):
            
            batch_size = len(images)
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            output = model(images)
        
            loss_dict = criterion(output, targets)
            weight_dict = criterion.weight_dict
        
            if log is None:
                log = {k:AverageMeter() for k in loss_dict}
                log['total_loss'] = AverageMeter()
                log['avg_prec'] = AverageMeter()
            
            for k,v in loss_dict.items():
                log[k].update(v.item(),batch_size)
        
            total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            log['total_loss'].update(total_loss.item(),batch_size)
            log['avg_prec'].update(challenge_metric(output,targets),batch_size)
            
            tk0.set_postfix({k:v.avg for k,v in log.items()}) 
    
    return log #['total_loss']



class Logger:
    def __init__(self,filename,format='csv'):
        self.filename = filename + '.' + format
        self._log = []
        self.format = format
    def save(self,log,epoch=None):
        log['epoch'] = epoch+1
        self._log.append(log)
        if self.format == 'json':
            with open(self.filename,'w') as f:
                json.dump(self._log,f)
        else:
            pd.DataFrame(self._log).to_csv(self.filename,index=False)
            
            
def run(fold,epochs=args.EPOCHS, load_model=False):
    
    train_data_loader,valid_data_loader = get_fold(fold)
    
    logger = Logger(f'log_{fold}')
    device = torch.device('cuda')
    if load_model:
        model = DETRModel(num_classes=args.num_classes)
        
        #loading previously trained model
        model.load_state_dict(torch.load("../input/model2/detr_best_-1.pth"))
        model.to(device)
    else:
        model = DETRModel(num_classes=args.num_classes)
    model = model.to(device)
    criterion = SetCriterion(args.num_classes, 
                             matcher, weight_dict, 
                             eos_coef = null_class_coef, 
                             losses=losses)
    
    criterion = criterion.to(device)
    

    optimizer = torch.optim.AdamW([{
        'params': v,
        'lr': lr_dict.get(k,1)*LR
    } for k,v in model.parameter_groups().items()], weight_decay=1e-4)
    
    best_precision = 0
    header_printed = False
    for epoch in range(epochs):
        train_log = train_fn(train_data_loader, model,criterion, optimizer,device,scheduler=None,epoch=epoch)
        valid_log = eval_fn(valid_data_loader, model,criterion, device)
    
        log = {k:v.avg for k,v in train_log.items()}
        log.update({'V/'+k:v.avg for k,v in valid_log.items()})
        logger.save(log,epoch)
        keys = sorted(log.keys())
        
        if not header_printed:
            print(' '.join(map(lambda k: f'{k[:8]:8}',keys)))
            header_printed = True
        print(' '.join(map(lambda k: f'{log[k]:8.3f}'[:8],keys)))
        
#         if log['V/avg_prec'] > best_precision:
        if epoch%5==0:
            best_precision = log['V/avg_prec']
            print('Best model found at epoch {}'.format(epoch+1))
            torch.save(model.state_dict(), f'detr_best_{fold}.pth')
        
        

gc.collect()



# # model = DETRModel(num_classes=num_classes,num_queries=num_queries)
# model = DETRModel(num_classes=args.num_classes)
# model.load_state_dict(torch.load("./detr_best_0.pth"))
# # model = DETRModel1(num_classes,model_name="detr_best_0.pth")

model.to(device)
# model.eval();

run(fold=0,epochs=20, load_model=False)

import sys
import os
import time
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
import torchvision.datasets as datasets
from torchvision import models
from PIL import Image
import csv
import cv2
import imageio
from resnet import resnet18,resnet50
from sklearn.metrics import accuracy_score,jaccard_similarity_score
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import skimage.transform
import matplotlib.image as mpimg
from PIL import Image
from torch import topk
from HeatMap import HeatMap
from scipy.ndimage import gaussian_filter
import matplotlib.image as mpimg
import utils as ut

feature_hooks=[]

class ChestXRayDataSet(Dataset):
    def __init__(self,data_path,label_path,transform=None):
        self.X=np.uint8(np.load(data_path)*224*224)
        with open(label_path,"rb") as f:
            self.y=pickle.load(f)
        sub_bool=(self.y.sum(axis=1)!=0)
        #print('sub_bool',sub_bool)
        self.y=self.y[sub_bool,:]
        self.X=self.X[sub_bool,:]
        self.transform=transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self,index):
        item=self.X[index]
        label=self.y[index]

        if self.transform:
            item=self.transform(item)
        return item,torch.from_numpy(label).type(torch.FloatTensor)


def feature_hook(module,input,output):
    feature_hooks.append(output)

class ResNet(nn.Module):
    def __init__(self,num_classes):
        super(ResNet,self).__init__()
        '''
        self.resnet18=resnet18(pretrained=False)
        num_ftrs=self.resnet18.fc.in_features
        self.resnet18.fc=nn.Linear(num_ftrs,num_classes)
        '''
        self.resnet50=resnet50(pretrained=False)
        num_ftrs=self.resnet50.fc.in_features
        self.resnet50.fc=nn.Linear(num_ftrs,num_classes)

    
    def forward(self,x):
        x=self.resnet50(x)
        return x


def main():
    start_time=time.time()
    #cwd=sys.argv[1]#'C:\\Users\\mushu\\Desktop\\PythonTest\\'
    cwd=sys.argv[1] #'test_5'
    nodule_file_name=sys.argv[2]#'nodules.txt'
    output_directory=sys.argv[3] #'/output_masks'

    if not os.path.exists(os.path.join(os.getcwd(),output_directory)):
        os.makedirs(output_directory)

    #cwd = os.path.join(cwd,data_fol_name)
    #print('\nCurrent working directory:',cwd)
    N_CLASSES=15

    model=ResNet(N_CLASSES)
    model_file_name='/home/killua/ml/beproject/showntest/ResNet50_Epoch_6_04032019.pkl'
    checkpoint=torch.load(model_file_name,map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    #model.cuda()
    model.eval()
    conv_layer= model._modules.get('resnet50').layer4
    model._modules.get('resnet50').layer4.register_forward_hook(feature_hook)

    display_transform=transforms.Compose([transforms.Resize((512,512))])

    data_files=sorted([x for x in os.listdir(cwd) if x.endswith('.png')])
    nodule_files_path=os.path.join(os.getcwd(),nodule_file_name)
    fp=open(nodule_files_path,"r")
    #fpn=open(nodule_files_name,"w+")
    file_names= sorted([x.strip() for x in fp.readlines() if x!=""])
    #print('\n\nNo of files to be read:',len(data_files))
    #print('\n')
    CLASS_NAMES = sorted(['Atelectasis', 'Cardiomegaly','Effusion', 'Infiltration','Mass','Nodule', 'Pneumonia', 'Pneumothorax','Consolidation','Edema','Emphysema','Fibrosis','Pleural_Thickening','Hernia','No Finding'])
    class_map={}
    i=0
    for s in CLASS_NAMES:
        class_map[s]=i
        i=i+1

    #thresholds = np.load('thresholds.npy')
    #print(thresholds)
    
    for i in range(len(file_names)):
        for j in range(len(data_files)):
            if(file_names[i]==data_files[j]):
                impath=os.path.join(cwd,file_names[i])
                img=cv2.imread(impath)
                img=cv2.resize(img,(224,224))
                img_tensor=ut.image_loader(impath)
                img_tensor=Variable(img_tensor.view(-1,3,224,224),0)
                if torch.no_grad():
                    
                    output=model(img_tensor)
                    sigm=torch.nn.Sigmoid()
                    pred_probabilities=sigm(output)[0]

                    #indices=np.argwhere(pred_probabilities>=0.066)[0]

                    #<================== Object Localization =========================>

                    #preds=topk(pred_probabilities,5)[1].data
                    preds=np.argwhere(pred_probabilities>=0.066)[0]

                    weight_softmax_params=list(model._modules.get('resnet50').fc.parameters())
                    weight_softmax=np.squeeze(weight_softmax_params[0].cpu().data.numpy())
                    overlay=ut.getCAM(feature_hooks[-1],weight_softmax,11)

                    imgoverlay=Image.fromarray(overlay[0],'RGB')
                    imgoverlay.save('overlay.png')
                    imgoverlay=cv2.imread('overlay.png')
                    #imshow(imgoverlay)
                    #print(imgoverlay)
                    
                    #red_boundary=[np.array([28,28,128]),np.array([97,105,255])] 
                    red_boundary=[np.array([28,28,128]),np.array([63,63,255])]
                    mask=cv2.inRange(imgoverlay,red_boundary[0],red_boundary[1])
                    cropped=cv2.bitwise_and(imgoverlay,imgoverlay,mask=mask)
                    contour_out=cropped
                    cropped_gray=cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)

                    _,thresh=cv2.threshold(cropped_gray,0,1,0)
                    contours,hierarchy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

                    if(len(contours)!=0):
                        cv2.drawContours(contour_out,contours,-1,(0,255,0),3)
                        c=max(contours,key=cv2.contourArea)
                        #x,y,w,h=cv2.boundingRect(c)
                        #cv2.rectangle(contour_out,(x,y),(x+w,y+h),(255,0,0),2)

                    cv2.imshow("result",np.hstack([cropped,contour_out]))
                    cv2.waitKey(0)
                    #imshow(cropped_gray,cmap='jet')


                    heatmap_image=Image.fromarray(cropped_gray)
                    heatmap_image=heatmap_image.resize((224,224))
                    heatmap_image=gaussian_filter(heatmap_image,sigma=(10,10),order=0)
                    heatmap_image=np.asarray(heatmap_image)

                    heatmap_resized=Image.fromarray(heatmap_image)
                    heatmap_resized.save('heatmap.png')

                    heatmap_resized=cv2.imread('heatmap.png')
                    box_mask=cv2.inRange(heatmap_resized,np.array([1,1,1]),np.array([108,108,108]))
                    masked_overlay=np.zeros((224,224,3),dtype=np.uint8)
                    #imshow(box_mask,cmap='jet')

                    for k in range(box_mask.shape[0]):
                        for l in range(box_mask.shape[1]):
                            if(box_mask[k,l]==0):
                                masked_overlay[k,l]=[225,0,0]
                            else:
                                masked_overlay[k,l]=img[k,l]


                    masked_out=Image.fromarray(masked_overlay,'RGB')
                    fname='output_mask_of_'+file_names[i]
                    masked_out.save(os.path.join(output_directory,fname))



        done=int(((i+1)/len(file_names))*100)
        sys.stdout.write("\r% of files done processing: "+str(done))

    print("\n\nResults saved in",output_directory)
    total_time_exec=int(time.time()-start_time)//60
    print("\nTotal Time taken: "+str(total_time_exec)+" minutes")
    

if __name__ == "__main__":
    sys.exit(main())

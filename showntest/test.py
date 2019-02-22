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
from resnet import resnet18
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

feature_hooks=[]
fc_hooks=[]

class AlexNet(nn.Module):

    def __init__(self,num_classes=15):
        super(AlexNet,self).__init__()
        self.features = nn.Sequential(
                nn.Conv2d(3,64,kernel_size=11,stride=4,padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3,stride=2),
                nn.Conv2d(64,192,kernel_size=5,padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3,stride=2),
                nn.Conv2d(192,384,kernel_size=3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384,256,kernel_size=3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256,256,kernel_size=3,padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2,stride=2),
            )
        self.classifier=nn.Sequential(
                nn.Dropout(),
                nn.Linear(256*6*6,4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096,4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096,num_classes),
            )

    def forward(self,x):
        x=self.features(x)
        x=x.view(x.size(0),256*6*6)
        x=self.classifier(x)
        return x

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

def image_loader(image_name):
    imsize=224
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
    transform = transforms.Compose([transforms.ToPILImage(),
    	transforms.Resize(imsize),
        transforms.RandomCrop(imsize),
    transforms.ToTensor(),
    normalize])
    #current_X=Image.open(image_name)
    current_X=imageio.imread(image_name)
    #print(len(current_X))
    if current_X.shape!=(1024,1024):
        current_X=current_X[:,:,0]

    x_data=[]
    x_data.append(np.array(current_X).reshape(1024,1024,1))
    image=np.tile(x_data[0],3)
    #print(type(image))
    image=transform(image)
    image=Variable(image,requires_grad=True)
    return image

def feature_hook(module,input,output):
    feature_hooks.append(output)

def fc_hook(module,input,output):
    fc_hooks.append(output)

def getCAM(feature_conv,weight_fc,class_idx):
    _,nc,h,w=feature_conv.shape
    cam=weight_fc[class_idx].dot(feature_conv.data.numpy().reshape((nc,h*w)))
    cam=cam.reshape(h,w)
    cam=cam-np.min(cam)
    cam_img=cam/np.max(cam)
    return [cam_img]

class ResNet(nn.Module):
    def __init__(self,num_classes):
        super(ResNet,self).__init__()
        self.resnet18=resnet18(pretrained=True)
        num_ftrs=self.resnet18.fc.in_features
        self.resnet18.fc=nn.Linear(num_ftrs,num_classes)
    
    def forward(self,x):
        x=self.resnet18(x)
        return x


def main():
    start_time=time.time()
    #cwd=sys.argv[1]#'C:\\Users\\mushu\\Desktop\\PythonTest\\'
    cwd=sys.argv[1]#'test_5'
    output_file_name=sys.argv[2]#'out.csv'
    #cwd = os.path.join(cwd,data_fol_name)
    #print('\nCurrent working directory:',cwd)
    N_CLASSES=15

    model=ResNet(N_CLASSES)
    model_file_name='/home/killua/ml/beproject/showntest/ResNet_Epoch_1_14022019.pkl'
    checkpoint=torch.load(model_file_name,map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    #model.cuda()
    model.eval()
    conv_layer= model._modules.get('resnet18').layer4
    #model._modules.get('resnet18').layer4[1].conv2.register_forward_hook(feature_hook)
    model._modules.get('resnet18').layer4.register_forward_hook(feature_hook)

    display_transform=transforms.Compose([transforms.Resize((512,512))])

    data_files=sorted([x for x in os.listdir(cwd) if x.endswith('.png')])
    label_file=[x for x in  os.listdir(cwd) if x.endswith('.txt')][0]
    label_file=os.path.join(cwd,label_file)
    fp=open(label_file,"r")
    labels = sorted([x for x in fp.readlines() if x!=""])
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
    fp=open(output_file_name,'w')
    csv_writer=csv.writer(fp)
    file_header=['File Name','Finding Labels']
    csv_writer.writerow(file_header)
    avg_acc=0.0
    all_targets=[]
    all_bin_preds=[]
    
    for i in range(len(data_files)):
        fname=labels[i].split(":")[0]
        target=labels[i].split(":")[1].rstrip()

        bin_target=[0]*N_CLASSES
        bin_pred_list=[0]*N_CLASSES


        gtlabels=[]
        if("|" in target):
            gtlabels=target.split("|")
        else:
            gtlabels.append(target)
        for s in gtlabels:
            bin_target[class_map[s]]=1

        #all_targets.append(bin_target)

        if(fname==data_files[i]):
            impath=os.path.join(cwd,fname)
            img=cv2.imread(impath)
            img=cv2.resize(img,(224,224))
            img_tensor=image_loader(impath)
            img_tensor=Variable(img_tensor.view(-1,3,224,224),0)
            if torch.no_grad():
                
                output=model(img_tensor)
                sigm=torch.nn.Sigmoid()
                pred_probabilities=sigm(output)[0]

                
                indices=np.argwhere(pred_probabilities>=0.066)[0]
                pred_label_list=[]
                for j in indices.data:
                    pred_label_list.append(CLASS_NAMES[j])
                    bin_pred_list[class_map[CLASS_NAMES[j]]]=1

                #all_bin_preds.append(bin_pred_list)
                avg_acc=avg_acc+accuracy_score(np.array(bin_target),np.array(bin_pred_list),normalize=True)
                pred_label="|".join(pred_label_list)
                #print(pred_label)
                row=[fname,pred_label]
                csv_writer.writerow(row)
                preds=topk(pred_probabilities,10)[1].data


                #<================== Object Localization =========================>

                weight_softmax_params=list(model._modules.get('resnet18').fc.parameters())
                weight_softmax=np.squeeze(weight_softmax_params[0].cpu().data.numpy())
                overlay=getCAM(feature_hooks[-1],weight_softmax,11)

                imgoverlay=Image.fromarray(overlay[0],'RGB')
                imgoverlay.save('overlay.png')
                imgoverlay=cv2.imread('overlay.png')
                #imshow(imgoverlay)
                #print(imgoverlay)
                
                red_boundary=[np.array([28,28,128]),np.array([97,105,255])]
                mask=cv2.inRange(imgoverlay,red_boundary[0],red_boundary[1])
                cropped=cv2.bitwise_and(imgoverlay,imgoverlay,mask=mask)
                cropped_gray=cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
                #imshow(cropped_gray,cmap='jet')


                #heatmap_image=Image.fromarray(overlay[0]*255)
                heatmap_image=Image.fromarray(cropped_gray)
                #print(list(heatmap_image.getdata()))
                heatmap_image=heatmap_image.resize((224,224))
                heatmap_image=gaussian_filter(heatmap_image,sigma=(10,10),order=0)
                heatmap_image=np.asarray(heatmap_image)

                heatmap_resized=Image.fromarray(heatmap_image)
                heatmap_resized.save('heatmap.png')

                heatmap_resized=cv2.imread('heatmap.png')
                #print(heatmap_resized.shape)
                unique_vals=[]
                #print(np.amax(heatmap_resized))
                #imshow(heatmap_resized)
                box_mask=cv2.inRange(heatmap_resized,np.array([1,1,1]),np.array([108,108,108]))
                #imshow(box_mask,cmap='jet')


                fig=plt.figure()
                ax=fig.add_subplot(111)
                img=np.asarray(img)
                #'''
                ax.imshow(img)
                ax.imshow(box_mask,alpha=0.5,cmap='jet')
                plt.show()
                #'''



        done=int(((i+1)/len(labels))*100)
        #sys.stdout.write("\r% of files done processing: "+str(done))

    plt.show()
    avg_acc=avg_acc/len(data_files)
    #avg_acc=jaccard_similarity_score(np.array(all_targets),np.array(all_bin_preds),normalize=True)
    #print("\n\nResults saved in",output_file_name)
    print("Average accuracy is: %.2f" %(avg_acc*100))
    total_time_exec=int(time.time()-start_time)//60
    #return avg_acc*100
    #print("\nTotal Time taken: "+str(total_time_exec)+" minutes")
    

if __name__ == "__main__":
    sys.exit(main())

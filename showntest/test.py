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
from HeatMap import HeatMap
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

def old_method():
    dataset=ChestXRayDataSet(data_files,label_file,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.224])
            ]))
    data_loader=DataLoader(dataset=dataset,batch_size=1,shuffle=True)

    gt=torch.FloatTensor()
    pred=torch.FloatTensor()
    i=1
    print(len(data_loader))

    if torch.no_grad():
        for (image,target) in data_loader:
            image=Variable(image.view(-1,1,224,224),0)
            output=model(image)
            sigm=torch.nn.Sigmoid()
            probs=sigm(output)
            print(probs)
            indices=np.argwhere(probs>=0.067)

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

def compute_AUCs(gt, pred):

    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


def feature_hook(module,input,output):
    feature_hooks.append(output)

def fc_hook(module,input,output):
    fc_hooks.append(output)

def getCAM(feature_conv,weight_fc,class_idx):
    _,nc,h,w=feature_conv.shape
    cam=weight_fc[class_idx].dot(feature_conv.data.numpy().reshape((nc,h*w)))
    print('shape',cam.shape)
    cam=cam.reshape(h,w)
    '''cam=cam-np.min(cam)
    cam_img=cam/np.max(cam)
    '''
    return [cam]

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
    model_file_name='/home/mikasa/ml/beproject/showntest/ResNet_Epoch_6_09022019.pkl'
    checkpoint=torch.load(model_file_name,map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    #model.cuda()
    model.eval()
    conv_layer= model._modules.get('resnet18').layer4
    model._modules.get('resnet18').layer4[1].conv2.register_forward_hook(feature_hook)
    model._modules.get('resnet18').fc.register_forward_hook(fc_hook)

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
    prev_overlay=list()
    overlays=[]

    fig=plt.figure(figsize=(8,8))
    col=2
    rows=5

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
            img_tensor=image_loader(impath)
            img_tensor=Variable(img_tensor.view(-1,3,224,224),0)
            if torch.no_grad():
                
                output=model(img_tensor)
                pred_probabilities=F.softmax(output).data.squeeze()

                
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

                if 5 in indices:
                    weight_softmax_params=list(model._modules.get('resnet18').fc.parameters())
                    weight_softmax=np.squeeze(weight_softmax_params[0].cpu().data.numpy())
                    overlay=getCAM(feature_hooks[-1],weight_softmax,5)
                    overlays.append(overlay[0])

                    fig.add_subplot(rows,col,i+1)
                    plt.imshow(overlay[0])


        done=int(((i+1)/len(labels))*100)
        #sys.stdout.write("\r% of files done processing: "+str(done))

    plt.show()
    print(overlays)
    avg_acc=avg_acc/len(data_files)
    #avg_acc=jaccard_similarity_score(np.array(all_targets),np.array(all_bin_preds),normalize=True)
    #print("\n\nResults saved in",output_file_name)
    print("Average accuracy is: %.2f" %(avg_acc*100))
    total_time_exec=int(time.time()-start_time)//60
    return avg_acc*100
    #print("\nTotal Time taken: "+str(total_time_exec)+" minutes")
    

if __name__ == "__main__":
    sys.exit(main())

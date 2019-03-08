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
        '''self.resnet18=resnet18(pretrained=False)
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
    cwd=sys.argv[1]#'test_5'
    output_file_name=sys.argv[2]#'out.csv'
    nodule_files_name=sys.argv[3] #'nodules.txt'
    #cwd = os.path.join(cwd,data_fol_name)
    #print('\nCurrent working directory:',cwd)
    N_CLASSES=15

    model=ResNet(N_CLASSES)
    #model_file_name='/home/killua/ml/beproject/showntest/ResNet_Epoch_1_14022019.pkl'
    model_file_name='/home/killua/ml/beproject/showntest/ResNet50_Epoch_6_04032019.pkl'
    checkpoint=torch.load(model_file_name,map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    #model.cuda()
    model.eval()
    conv_layer= model._modules.get('resnet50').layer4
    model._modules.get('resnet50').layer4.register_forward_hook(feature_hook)

    display_transform=transforms.Compose([transforms.Resize((512,512))])

    data_files=sorted([x for x in os.listdir(cwd) if x.endswith('.png')])
    label_file=[x for x in  os.listdir(cwd) if x.endswith('.txt')][0]
    label_file=os.path.join(cwd,label_file)
    fp=open(label_file,"r")
    fpn=open(nodule_files_name,"w+")
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


        if(fname==data_files[i]):
            impath=os.path.join(cwd,fname)
            img=cv2.imread(impath)
            img=cv2.resize(img,(224,224))
            img_tensor=ut.image_loader(impath)
            img_tensor=Variable(img_tensor.view(-1,3,224,224),0)
            if torch.no_grad():
                
                output=model(img_tensor)
                sigm=torch.nn.Sigmoid()
                pred_probabilities=sigm(output)[0]

                #indices=np.argwhere(pred_probabilities>=0.066)[0]
                indices=topk(pred_probabilities,5)[1].data
                pred_label_list=[]
                for j in indices.data:
                    if(j==11):
                        fpn.write(fname+'\n')
                    pred_label_list.append(CLASS_NAMES[j])
                    bin_pred_list[class_map[CLASS_NAMES[j]]]=1

                avg_acc=avg_acc+accuracy_score(np.array(bin_target),np.array(bin_pred_list),normalize=True)
                pred_label="|".join(pred_label_list)
                row=[fname,pred_label]
                csv_writer.writerow(row)


        done=int(((i+1)/len(labels))*100)
        sys.stdout.write("\r% of files done processing: "+str(done))

    fpn.close()

    plt.show()
    avg_acc=avg_acc/len(data_files)
    #avg_acc=jaccard_similarity_score(np.array(all_targets),np.array(all_bin_preds),normalize=True)
    print("\n\nResults saved in",output_file_name)
    print("Average accuracy is: %.2f" %(avg_acc*100))
    total_time_exec=int(time.time()-start_time)//60
    print("\nTotal Time taken: "+str(total_time_exec)+" minutes")
    

if __name__ == "__main__":
    sys.exit(main())

import os,sys
import time
import argparse
import statistics
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
from skimage import transform
import torchvision.transforms as transforms
from torch.autograd import Variable
import pandas as pd;
import numpy as np;
from torch.utils.data import Dataset, DataLoader
#from vis_utils import *
import random;
import math;

#plt.ion()   
num_epochs=3
batch_size=4
learning_rate=0.001
os.environ['CUDA_VISIBLE_DEVICES']="0"

class ChestXRayDataSet(Dataset):
    def __init__(self,data_path,label_path,transform=None):
        self.X=np.uint8(np.load(data_path)*255*255)
        with open(label_path,"rb") as f:
            self.y=pickle.load(f)
        sub_bool=(self.y.sum(axis=1)!=0)
        print('sub_bool',sub_bool)
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
        return (item,label)


if __name__=='__main__':

    cwd=os.getcwd()
    print('Using device:',torch.cuda.get_device_name(0))
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",action="store",required=True,help='get path to dataset',dest='dataset')
    parser.add_argument("--output",action="store",required=True,help='get path to output',dest='output')
    args=parser.parse_args()
    dataset=os.path.join(cwd,args.__dict__['dataset'])
    output=os.path.join(cwd,args.__dict__['output'])
    epoch_losses=[]

    
    for epoch in range(num_epochs):
        epoch_start_time=time.time()
        print("Epoch",epoch)

        data_files=sorted([x for x in os.listdir(dataset) if x.startswith('train') and x.endswith('.npy')])
        label_files=sorted([x for x in os.listdir(dataset) if x.startswith('train') and x.endswith('.pkl')])
        print(data_files)
        print(label_files)
        batch_losses=[]
        epoch_loss=0.0

        #Augmentation of training data
        for i in range(len(data_files)):
            print("Training",data_files[i])
            dataset=ChestXRayDataSet(data_files[i],label_files[i],
                    transform=transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.224])
                        ]))
            augment_img=[]
            augment_label=[]
            augment_weight=[]
            for i in range(2):
                for j in range(len(dataset)):
                    single_img,single_label,single_weight=dataset[j]
                    augment_img.append(single_img)
                    augment_label.append(single_label)
                    augment_weight.append(single_weight)
                    if j%1000==0:
                        print(j)
            print("Data Augmentation done OK")
            print("Length of augment data:",len(augment_label))

            #shuffle data
            permuted_order=torch.randperm(len(augment_label))
            augment_img=torch.stack(augment_img)[permuted_order]
            augment_label=torch.stack(augment_label)[permuted_order]
            augment_weight=torch.stack(augment_weight)[permuted_order]

            #======= start training =======#

            cuddn.benchmark=True
            N_CLASSES=15
            BATCH_SIZE=4
            intermediate_model_filepath=os.path.join(cwd,'intermediate_model.pkl')

            #Check for intermediate model state
            if os.path.exists(intermediate_model_filepath):
                checkpoint=torch.load('intermediate_model.pkl')
                model.load_state_dict(checkpoing['state_dict'])
            else:
                model=AlexNet(N_CLASSES).cuda()
            #set model to train mode
            model.train()
            criterion=nn.CrossEntropyLoss()
            optimizer=optim.Adam(model.parameters,lr=0.0002,betas=(0.9,0.999))
            total_length=len(augment_img)
            running_loss=0.0

            for i in range(0,total_length,BATCH_SIZE):
                if i+BATCH_SIZE>total_length:
                    break

                sub_input=augment_img[i:i+BATCH_SIZE]
                sub_label=augment_label[i:i+BATCH_SIZE]
                sub_weights=augment_weight[i:i+BATCH_SIZE]

                sub_input,sub_label,sub_weights=Variable(sub_input.cuda()),Variable(sub_label.cuda()),Variable(sub_weights.cuda())

                #forward + backward + optimizer update

                output=model(sub_input)
                loss=criterion(output,sub_label)
                loss.backward()
                optimizer.step()
                running_loss+=loss.data[0]

            batch_loss=running_lost/total_length
            batch_losses.append(batch_loss)

            #Save model for use in next batch
            state = {'state_dict':model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'loss':batch_loss,}
            torch.save(state,'intermediate_model.pkl')

        
            #========= validation ==========#

            data_files=sorted([x for x in os.listdir(dataset) if x.startswith('train') and x.endswith('.npy')])
            label_files=sorted([x for x in os.listdir(dataset) if x.startswith('train') and x.endswith('.pkl')])
            print(data_files)
            print(label_files)

            for i in range(len(data_files)):

                #Initialize validation set
                dataset=ChestXRayDataSet(data_files[i],label_files[i],
                        transform=transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.RandomCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.224])
                                ]))
                data_loader=DataLoader(dataset=dataset,batch_size=4,shuffle=True)


                #switch to eval mode 
                model.eval()
                correct=0
                total=0

                for (image,label,weight) in data_loader:
                    labels=label.cuda()
                    image=Variable(image.view(-1,3,224,224).cuda())
                    output=model(image)
                    _,predicted=torch.max(output.data,1)
                    total+=label.size(0)
                    correct+=(predicted==labels).sum()

            if os.path.exists(intermediate_model_filepath):
                os.remove(intermediate_model_filepath)

            #Print statistics
            print('Validation data test accuracy of the model on in epoch',str(epoch+1),'is',len(data_files),' test images: %.3f %%'
                    %(100*correct/total))
            print('Time taken for epoch #'+str(epoch+1)+' is: '+str((time.time()-start_time())/3600)+' hours')
            epoch_loss=statistics.mean(batch_losses)

            #Visualize losses
            plt.xkcd()
            plt.xlabel('Batch #')
            plt.ylabel('Loss')
            plt.plot(batch_losses)
            #plt.show()
            plt.savefig('Epoch_#'+str(epoch+1))

            #Save model for use in next batch
            state = {'state_dict':model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'loss':epoch_loss,}

            torch.save(state,'AlexNet_Epoch_'+str(epoch+1)+'.pkl')
            


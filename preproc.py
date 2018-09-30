import numpy as np
import matplotlib.pyplot as pit
import pandas as pd
import imageio
import skimage.transform
import pickle
import sys,os
import argparse
from os.path import exists 
from os import listdir
from sklearn.preprocessing import MultiLabelBinarizer


cwd=os.getcwd()
csv_path=os.path.join(cwd,'Data_Entry_2017.csv')
datacsv=pd.read_csv(csv_path)
parser=argparse.ArgumentParser()
parser.add_argument("--dataset",action="store",required=True,help='get path of dataset',dest='dataset')
parser.add_argument("--output",action="store",required=True,help='get path to output',dest='output')
args=parser.parse_args()
dataset=os.path.join(cwd,args.__dict__['dataset'])
output=os.path.join(cwd,args.__dict__['output'])
dirs=[x for x in os.listdir(os.path.join(cwd,dataset))]
#fnames=list(datacsv['Image Index'])
#flabel=list(datacsv['Finding Labels'])
outpath=os.path.join(cwd,output)
lbs=datacsv["Finding Labels"]
CLASSES=[]

for lb in lbs:
    if "|" in lb:
        CLASSES.append(lb.split("|"))
    else:
        CLASSES.append([lb])


encoder = MultiLabelBinarizer()
encoder.fit(CLASSES)
print(encoder.classes_)

for s in dirs:
    tp=os.path.join(dataset,s)
    dirs1=[x for x in os.listdir(tp)]

    for s1 in dirs1:
        txt_file=s1+".txt"
        tp1=os.path.join(tp,s1)
        labels=[]
        data=[]
        #print(tp1)
        if exists(os.path.join(tp1,txt_file)): 
            #print(txt_file)
            train_x=[]
            train_y=[]
            npname=s+"_"+s1+"_x.npy"
            lbname=s+"_"+s1+"_y.pkl"
            #files=[x for x in os.listdir(tp1) if x.endswith('png')]
            line=""
            with open(os.path.join(tp1,txt_file),"r") as fp:
                line=fp.read()
            lines=line.split('\n')
            for ls in lines:
                if ls:
                    fname=ls.split(':')[0]
                    labels=ls.split(':')[1]
                    #print('hi',fname,labels)
                    img_path=os.path.join(tp1,fname)
                    if(exists(img_path)):
                        img=imageio.imread(img_path)
                        if(img.shape!=(1024,1024)):
                            img=img[:,:,0]
                        img_resized=skimage.transform.resize(img,(224,224))
                        train_x.append(np.array(img_resized/255).reshape(224,224,1))
                        if "|" in labels:
                            train_y.append(labels.split('|'))
                        else:
                            train_y.append([labels])
            print('hi',train_y)
            train_x=np.array(train_x)
            np.save(os.path.join(outpath,npname),train_x)
            print("Saved",s1,"of",s)
            #encoder=MultiLabelBinarizer()
            #encoder.fit(CLASSES)
            onehot=encoder.transform(train_y)
            with open(os.path.join(outpath,lbname),"wb") as fp:
                pickle.dump(onehot,fp)
            with open(os.path.join(outpath,s+"_"+s1+"_label_encoder"),"wb") as fp:
                pickle.dump(encoder,fp)

        


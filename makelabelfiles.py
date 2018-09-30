import numpy as np
import matplotlib.pyplot as pit
import pandas as pd
#import imageio
#import skimage.transform
import pickle
import sys,os
import argparse
from os.path import exists 
from os import listdir
#from sklearn.preprocessing import MultiLabelBinarizer

cwd=os.getcwd()
csv_path=os.path.join(cwd,'Data_Entry_2017.csv')
datacsv=pd.read_csv(csv_path)
parser=argparse.ArgumentParser()
parser.add_argument("--dataset",action="store",required=True,help='get path of dataset',dest='dataset')
args=parser.parse_args()
dataset=os.path.join(cwd,args.__dict__['dataset'])
dirs=[x for x in os.listdir(os.path.join(cwd,dataset))]
fnames=list(datacsv['Image Index'])
flabel=list(datacsv['Finding Labels'])


for s in dirs:
    tp=os.path.join(dataset,s)
    dirs1=[x for x in os.listdir(tp)]

    for s1 in dirs1:
        txt_file=s1+".txt"
        tp1=os.path.join(tp,s1)
        labels=[]
        data=[]
        files=[x for x in os.listdir(tp1) if x.endswith('.png')]

        txt_path=os.path.join(tp1,txt_file)
        fp=open(txt_path,'w')
        for file in files:
            for i in range(len(fnames)):
                if(fnames[i]==file):
                    fp.write(fnames[i]+":"+flabel[i]+"\n")
        fp.close()
                    

        


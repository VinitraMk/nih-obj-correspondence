import sys,os
import pandas as pd
import numpy as np
import argparse
from shutil import copy,move
#path=sys.argv[1]

cwd=os.getcwd()
parser=argparse.ArgumentParser()
parser.add_argument("--dataset",action="store",required=True,help="add dataset path",dest="dataset")
args=parser.parse_args()
dataset=os.path.join(cwd,args.__dict__["dataset"])
print(dataset)

dirs=[x for x in os.listdir(dataset) if x.endswith('.png')==False]
print(len(dirs))

for s in dirs:
    mp=os.path.join(dataset,s)
    files=[x for x in os.listdir(mp) if x.endswith('.png')]
    n=len(files)
    rn=int(0.3*n)
    os.chdir(mp)
    j=1
    for i in range(0,len(files),rn):
        folder="batch_"+str(j)
        path=os.path.join(mp,folder)
        try:
            os.makedirs(path)
        except: 
            print('exception occured')

        if(i+rn<n):
            batch=files[i:i+rn]
            for s1 in batch:
                move(s1,path)
            print(j,len(os.listdir(path)))
        else:
            batch=files[i:]
            for s1 in batch:
                move(s1,path)
            print(j,len(os.listdir(path)))
        j=j+1
            

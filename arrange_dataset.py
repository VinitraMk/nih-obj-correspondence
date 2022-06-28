import os,sys
import pandas as pd
from math import ceil
from random import randint
import numpy as np
from shutil import copy

path_to_dataset=sys.argv[1]
print(path_to_dataset)
filelist=[file for file in os.listdir(path_to_dataset) if file.endswith('.png')]
ln=len(filelist)
print(ln)
vis=np.zeros(ln,dtype=int)
os.chdir(path_to_dataset)
test_length=int(ceil(ln*0.3))
print(test_length)
rn=ln-test_length
train_length=int(ceil(rn*0.7))
valid_length=rn-train_length

for i in range(train_length):
    rn=randint(0,ln-1)
    while vis[rn]==1:
        rn=randint(0,ln-1)
    filename=filelist[rn]
    dst=path_to_dataset+'/train/'
    copy(filename,dst)
    vis[rn]=1

for i in range(valid_length):
    rn=randint(0,ln-1)
    while vis[rn]==1:
        rn=randint(0,ln-1)
    filename=filelist[rn]
    dst=path_to_dataset+'/valid/'
    copy(filename,dst)
    vis[rn]=1


for i in range(len(vis)):
    if vis[i]==0:
        filename=filelist[i]
        vis[i]=1
        dst=path_to_dataset+'/test/'
        copy(filename,dst)


print('Enjoy your BE-Project! :-D')












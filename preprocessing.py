import numpy as np
import matplotlib.pyplot as pit
import pandas as pd
import imageio
import skimage.transform
import pickle
import sys,os
from os.path import exists 
from os import listdir
from sklearn.preprocessing import MultiLabelBinarizer


def get_training_list():
    print("Getting training list data...")
    sz=1
    i=0
    while sz<=12:
        train_x=[]
        file_found=True
        prefix="images_0"
        if(sz<10):
            dirname=prefix+"0"+str(sz)
        else:
            dirname=prefix+str(sz)
        print('Exploring',dirname)
        tl=len(train_list)
        while(file_found and i<tl):
            #print(i,end=" ")
            image_path=os.path.join(input_path+dirname,train_list[i])
            #print(image_path)
            if(exists(image_path)==False):
                print(i,dirname,train_list[i])
                file_found=False
                sz=sz+1
                break
            #print(image_path)
            img=imageio.imread(image_path)
            if img.shape!=(1024,1024):
                img=img[:,:,0]
            if img.shape==(1024,1024,4):
                print(image_path)
            #print(img.shape)
            img_resized=skimage.transform.resize(img,(256,256))
            train_x.append(np.array(img_resized/255).reshape(256,256,1))
            i=i+1
        if(i>=tl or sz==12):
            sz=sz+1
        if(len(train_x)>0):
            print()
            npname="train_reshaped_"+str(sz-1)+".npy"
            print("Writing into",npname,"...")
            train_x=np.array(train_x)
            np.save(os.path.join(output_path,npname),train_x)
            print('Saved',npname,'\n')




def get_valid_list():
    print("Getting valid list data...")
    #global valid_x
    sz=1
    i=0
    while sz<=12:
        valid_x=[]
        file_found=True
        prefix="images_0"
        if(sz<10):
            dirname=prefix+"0"+str(sz)
        else:
            dirname=prefix+str(sz)
        print('Exploring',dirname)
        tl=len(valid_list)
        while(file_found and i<tl):
            #print(valid_list[i])
            image_path=os.path.join(input_path+dirname,valid_list[i])
            if(exists(image_path)==False):
                print(i,dirname,valid_list[i])
                file_found=False
                sz=sz+1
                break
            img=imageio.imread(image_path)
            if img.shape!=(1024,1024):
                img=img[:,:,0]
            img_resized=skimage.transform.resize(img,(256,256))
            valid_x.append(np.array(img_resized/255).reshape(256,256,1))
            i=i+1
        #print(i)
        if(i>=tl or sz==12):
            sz=sz+1
        if(len(valid_x)>0):
            print()
            npname="valid_reshaped_"+str(sz-1)+".npy"
            print("Writing into",npname,"...")
            valid_x=np.array(valid_x)
            np.save(os.path.join(output_path,npname),valid_x)
            print("Saved",npname,"\n")


def get_labels(data_element):
    labels=meta_data.loc[meta_data["Image Index"]==data_element,"Finding Labels"]
    li=labels.tolist()
    if(li[0].find("|")==-1):
        return [li[0]]
    return li[0].split("|")



def label_process():
    fl=0
    for train_element in train_list:
        train_y.append(get_labels(train_element))

    for valid_element in valid_list:
        valid_y.append(get_labels(valid_element))


input_path=sys.argv[1]
output_path=sys.argv[2]
data_entry_path=input_path+"/Data_Entry_2017.csv"
bbox_list_path=input_path+"/BBox_List_2017.csv"
train_list_path=input_path+"/train_list.txt"
valid_list_path=input_path+"/valid_list.txt"
print(input_path,output_path)

meta_data=pd.read_csv(data_entry_path)
print(len(meta_data))
bbox_list=pd.read_csv(bbox_list_path)
print(len(bbox_list))

with open(train_list_path,"r") as f:
    train_list=sorted([s.strip() for s in f.readlines()])
    print('train len',len(train_list))
with open(valid_list_path,"r") as f:
    valid_list=sorted([s.strip() for s in f.readlines()])
    print('valid len',len(valid_list))
labels=list(np.unique(bbox_list["Finding Label"]))+["No Findings"]
print(labels)

train_x=[]
train_y=[]
valid_x=[]
valid_y=[]

get_training_list()
get_valid_list()
label_process()
encoder = MultiLabelBinarizer()
#print(encoder.classes_)
encoder.fit(train_y)
#encoder.fit(train_y+valid_y)
encoder.fit(valid_y)
train_y_onehot=encoder.transform(train_y)
valid_y_onehot=encoder.transform(valid_y)
train_y_onehot=np.delete(train_y_onehot,[2,3,5,6,7,10,12],1)
valid_y_onehot=np.delete(valid_y_onehot,[2,3,5,6,7,10,12],1)

with open(output_path+"/train_y_onehot.pkl","wb") as f:
    pickle.dump(train_y_onehot,f)
with open(output_path+"/valid_y_onehot.pkl","wb") as f:
    pickle.dump(valid_y_onehot,f)
with open(output_path+"/label_encoder.pkl","wb") as f:
    pickle.dump(encoder,f)

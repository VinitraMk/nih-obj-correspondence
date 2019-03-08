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


def getCAM(feature_conv,weight_fc,class_idx):
    _,nc,h,w=feature_conv.shape
    cam=weight_fc[class_idx].dot(feature_conv.data.numpy().reshape((nc,h*w)))
    cam=cam.reshape(h,w)
    cam=cam-np.min(cam)
    cam_img=cam/np.max(cam)
    return [cam_img]


import torch
import torchvision
import torch.nn.functional as F
import torch.nn as n

class twoconv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(twoconv,self).__init__()
        self.conv=nn.Sequential(
                nn.Conv2d(in_ch,out_ch,3,padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch,out_ch,3,padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
                )

    def forward(self,x):
        x=self.conv(x)
        return x

class downsamp(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(downsamp,self).__init__()
        self.convpool=nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_ch,out_ch)
                )

    def forward(self,x):
        x=self.convpool(x)
        return x

class upsamp(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(upsamp,self).__init__()
        self.upsamp=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.conv=double_conv(in_ch,out_ch)

    def forward(self,x1,x2):
        x1=self.upsamp(x1)
        diffy=x2.size()[2]-x1.size()[2]
        diffx=x2.size()[3]-x1.size()[3]

        x1=F.pad(x1,(diffx//2,diffx-diffx//2,diffy//2,diffy-diffy//2))
        x=torch.cat([x2,x1],dim=1)
        x=self.conv(x)
        return x


class outputconv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(outputconv,self).__init__()
        self.conv=nn.Conv2d(in_ch,out_ch,1)

    def forward(self,x):
        x=self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self,n_channels,n_classes):
        super(UNet,self).__init__()
        self.down_layer1=downsamp(n_channels,64)
        self.down_layer2=downsamp(64,128)
        self.down_layer3=downsamp(128,256)
        self.down_layer4=downsamp(256,512)
        self.down_layer5=downsamp(512,1024)
        self.up_layer1=upsamp(1024,512)
        self.up_layer2=upsamp(512,256)
        self.up_layer3=upsamp(256,128)
        self.up_layer4=upsamp(128,64)
        self.up_layer5=outputconv(64,n_classes)

    def forward(self,x):
        x1=self.down_layer1(x)
        x2=self.down_layer2(x1)
        x3=self.down_layer3(x2)
        x4=self.down_layer4(x3)
        x5=self.down_layer5(x4)
        x=self.up_layer1(x5,x4)
        x=self.up_layer2(x,x3)
        x=self.up_layer3(x,x2)
        x=self.up_layer4(x,x1)
        x=self.up_layer5(x)
        return F.sigmoid(x)




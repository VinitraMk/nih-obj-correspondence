import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
import os,sys
from unet_model import *
import matplotlib.pyplot as plt

num_epochs=10
learning_rate=0.01
momentum=0.99
N_CLASSES=1

def main():

    #Setting relevant paths in vars
    root_path=sys.argv[1]
    cwd=os.getcwd()
    data_path=os.path.join(root_path,sys.argv[2])
    mask_path=os.path.join(root_path,sys.argv[3])
    output_path=os.path.join(root_path,sys.argv[4])

    epoch_losses=[]

    #Starting epochs
    for epoch in range(num_epochs):
        data_files=sorted([x for x in os.listdir(data_path) if x.endswith('.png')])
        mask_files=sorted([x for x in os.listdir(mask_path) if x.endswith('.png')])

        model=UNet(3,N_CLASSES)
        model=torch.nn.DataParallel(model).cuda()
        criterion=nn.BCEWithLogitsLoss()
        optimizer=optim.Adam(model.parameters(),lr=learning_rate,momentum=momentum)
        total_data_length=len(data_files)

        #Starting training

        for i in range(len(data_files)):
            print(data_files[i],mask_files[i])
            img=np.array(data_files[i])
            true_mask=np.array(mask_files[i])
            pred_mask=model(img)
            loss=criterion(pred_mask,true_mask)
            loss.backward()
            optimizer.step()
            running_loss+=loss.data[0]
            compcount=int((i+1)/(total_data_length)*100)
            sys.stdout.write('\r'+"% of training completed: "+str(compcount))

        temp_avg_loss=running_loss/total_length
        epoch_losses.append(temp_avg_loss)

        state={'state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'loss':temp_avg_loss,}
        torch.save(state,'Unet_Epoch#'+str(epoch)+'.pkl')
        torch.cuda.empty_cache()

    print('\n\nVisualizing losses')
    plt.xkcd()
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.plot(epoch_losses,marker='o',color='b')
    plt.savefig('Epoch_AUROC_and_Losses')


if __name__=='__main__':
    if torch.cuda.is_available()==False:
        sys.exit(0)
    else:
        device=torch.device("cuda:0")

    main()

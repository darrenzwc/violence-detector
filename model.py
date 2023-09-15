import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, tqdm
from video_dataset import  VideoFrameDataset, ImglistToTensor
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import torchvision.transforms as transforms
import torch
import torchaudio

class model(nn.Module):
    def __init__(self, lr=0.0001, lrDecay=0.95, **kwargs):
        super(model, self).__init__()

        log_dir = 'logs'
        self.writer = SummaryWriter(log_dir=log_dir)
        
        self.visualModel = None
        self.createVisualModel()
        
        self.visualModel = self.visualModel.cuda()
        
        self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)
        
        self.loss_fn = nn.CrossEntropyLoss()
        
    def createVisualModel(self):
        #self.vgg16_input_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        #vgg16 = models.vgg16(weights='DEFAULT')
        #self.visualModel = nn.Sequential(*(list(vgg16.children())[:-1]))
        #self.flatten = nn.Flatten()
        self.visualModel = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.Flatten()
        )
        

    def train_network(self, loader, epoch, **kwargs):
        
        self.train()
        self.scheduler.step(epoch-1)
        lr = self.optim.param_groups[0]['lr']
        index, top1, loss = 0, 0, 0
        for num, (visualFeatures, labels) in enumerate(loader, start=1):
                self.zero_grad()                
                visualFeatures = visualFeatures.cuda()#
                #visualFeatures = torch.transpose(visualFeatures, 4, 1) #adjusted dimension 4 -> 3
                #visualFeatures = torch.squeeze(visualFeatures)#
                #visualFeatures = self.vgg16_input_transform(visualFeatures)#
                #visualFeatures = visualFeatures.cuda()#
                labels = labels.squeeze().cuda()
                visualFeatures = visualFeatures.view(-1, 3, 299, 299)                
                visualEmbed = self.visualModel(visualFeatures)

                visualEmbed = visualEmbed[:labels.size(0)]

                nloss = self.loss_fn(visualEmbed, labels)
                
                self.optim.zero_grad()
                nloss.backward()
                self.optim.step()
                
                loss += nloss.detach().cpu().numpy()
                
                top1 += (visualEmbed.argmax(1) == labels).type(torch.float).sum().item()
                index += len(labels)
                sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
                " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), 100 * (top1/index)))
                sys.stderr.flush()  
                sys.stdout.write("\n")
                self.writer.add_scalar("Loss/train", loss/(num), epoch)
                self.writer.add_scalar("Accuracy/train", 100 * (top1/index), epoch)
                self.writer.add_scalar("Learning_rate", lr, epoch)
                return loss/num, lr

    def evaluate_network(self, loader, **kwargs):
        self.eval()
        predScores = []
        
        loss, top1, index, numBatches = 0, 0, 0, 0
        top10, top11, index1, index0 = 0, 0, 0, 0
        for visualFeatures, labels in tqdm.tqdm(loader):
            #visualFeatures = torch.transpose(visualFeatures, 4, 1)#
            #visualFeatures = torch.squeeze(visualFeatures)#
            #visualFeatures = visualFeatures.cuda()
            #visualFeatures = self.vgg16_input_transform(visualFeatures)#
            visualFeatures = visualFeatures.cuda()#

            labels = labels.squeeze().cuda()
            
            with torch.no_grad():
                visualFeatures = visualFeatures.view(-1, 3, 299, 299)  
                visualEmbed = self.visualModel(visualFeatures)
                
                visualEmbed = visualEmbed[:labels.size(0)]
                nloss = self.loss_fn(visualEmbed, labels)
                
                loss += nloss.detach().cpu().numpy()
                top1 += (visualEmbed.argmax(1) == labels).type(torch.float).sum().item()
                index += len(labels)
                numBatches += 1
                
                label1bool = labels ==1
                alllabel1 = labels[label1bool]
                label0bool = labels == 0
                alllabel0 = labels[label0bool]
                if len(alllabel1) > 0:
                    predlabel1 = visualEmbed.argmax(1)[label1bool]
                    top11 += (predlabel1 == alllabel1).type(torch.float).sum().item()
                    index1 += len(alllabel1)
                if len(alllabel0) > 0:
                    predlabel0 = visualEmbed.argmax(1)[label0bool]
                    top10 += (predlabel0 == alllabel0).type(torch.float).sum().item()
                    index0 += len(alllabel0)
        p1 = top11/index1
        p0 = top10/index0
        mAP = (p0+p1)/2
        print('eval mAP', mAP)
        print('eval loss ', loss/numBatches)
        print('eval accuracy ', top1/index)
        
        return top1/index
            
            
    def saveParameters(self, path):
            torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)

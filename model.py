import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, tqdm

import torchvision.transforms as transforms

import torch
import torchaudio
class model(nn.Module):
    def __init__(self, lr=0.0001, lrDecay=0.95, **kwargs):
        super(model, self).__init__()
    def createVisualModel(self):
        
        self.visualModel = nn.Sequential(
            # Matrices resizes to 128x128
            transforms.Resize((128, 128)),
            # Runs Matrix through Convolutional layer and applys ReLu to introduce nonlinearlization
            nn.ReLU(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)),
            #Normalizes layers and feature maps before putting it through more hidden layers
            nn.BatchNorm2d(64),
            # Emphasizes features and maximizes the layer in certain parts adding more weight
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            #Flattens matrice into 1D
            nn.Flatten()
        )
    def train_network(self, loader, epoch, **kwargs):
        # Sets network on training mode
        self.fcModel.train()
        # Iterates over the dataloader
        for i, (visualFeatures, labels) in enumerate(loader):
            # Iterates through entire dataset in batches
            visualFeatures = visualFeatures.cuda()
            # GPU acceleration
            labels = labels.squeeze().cuda()

            visualEmbed = self.visualModel(visualFeatures)
            # Calculates loss between prediction and actual ground truth (the labels)
            nloss = self.loss_fn(fcOutput, labels)

            self.optim.zero_grad()
            nloss.backward()
            self.optim.step()
            # Backpropagation and optimization
            loss += nloss.detach().cpu().numpy()

            top1 += (fcOutput.argmax(1) == labels).type(torch.float).sum().item()
            index += len(labels)
            # Prints training progress information to the console using standard error output
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
            " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), 100 * (top1/index)))
            sys.stderr.flush()  
        sys.stdout.write("\n")
        # Returns the average loss over the entire training datase and the learning rate
        return loss/num, lr

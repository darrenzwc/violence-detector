import os, glob, time
import argparse
from model import *
from dataLoader_Image_audio import train_loader, val_loader
from torch.utils.tensorboard import SummaryWriter

def parser(): #provide configurations to command line for model

    args = argparse.ArgumentParser(description="Violence Detector")

    args.add_argument("--lr", type=float, default=0.0001, help="Learning Rate")
    args.add_argument('--maxEpoch', type=int, default=20, help='# of epochs')
    args.add_argument('--testInterval', type=int, default=1, help='Test and save every [testInterval] epochs')
    args.add_argument('--batchSize', type=int, default=128, help='Dynamic batch size, default is 500 frames.')
    args.add_argument('--nDataLoaderThread', type=int, default=4, help='Number of loader threads')
    args.add_argument('--datasetPath', type=str, default="/notebooks/violence_ds", help='Path to the Violence Dataset')
    args.add_argument('--loadAudioSeconds', type=float, default=3, help='Number of seconds of audio to load for each training sample')
    args.add_argument('--loadNumImages', type=int, default=1, help='Number of images to load for each training sample')
    args.add_argument('--savePath', type=str, default="exps/exp1")
    args.add_argument('--evalDataType', type=str, default="val", help='The dataset for evaluation, val or test')
    args.add_argument('--evaluation', dest='evaluation', action='store_true', help='Only do evaluation')
    args.add_argument('--eval_model_path', type=str, default="path not specified", help="model path for evaluation")

    args = args.parse_args()

    return args
    
def main(args):

if __name__=="__main__":

    args = parser()

    main(args)
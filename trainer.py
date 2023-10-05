import os, glob, time
import argparse
from model import *
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
#
def parser(): #provide configurations to command line for model

    args = argparse.ArgumentParser(description="Violence Detector")

    args.add_argument("--lr", type=float, default=0.0001, help="Learning Rate")
    args.add_argument('--maxEpoch', type=int, default=75, help='# of epochs')
    args.add_argument('--testInterval', type=int, default=1, help='Test and save every [testInterval] epochs')
    args.add_argument('--batchSize', type=int, default=20, help='Dynamic batch size, default is 500 frames.')
    args.add_argument('--nDataLoaderThread', type=int, default=4, help='Number of loader threads')
    args.add_argument('--datasetPath', type=str, default="/notebooks/VD-DS/", help='Path to the Violence Dataset')
    args.add_argument('--savePath', type=str, default="exps/exp1")
    args.add_argument('--evalDataType', type=str, default="val", help='The dataset for evaluation, val or test')
    args.add_argument('--evaluation', dest='evaluation', action='store_true', help='Only do evaluation')
    args.add_argument('--eval_model_path', type=str, default="path not specified", help="model path for evaluation")

    args = args.parse_args()
    return args

def plot_video(rows, cols, frame_list, plot_width, plot_height, title: str):
    fig = plt.figure(figsize=(plot_width, plot_height))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(rows, cols),  # creates 2x2 grid of axes
                     axes_pad=0.3,  # pad between axes in inch.
                     )

    for index, (ax, im) in enumerate(zip(grid, frame_list)):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_title(index)
    plt.suptitle(title)
    plt.show()

def main(args):
    preprocess = transforms.Compose([
        ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        transforms.Resize(299),  # image batch, resize smaller edge to 299
        transforms.CenterCrop(299),  # image batch, center crop to square 299x299
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = VideoFrameDataset(
        root_path = args.datasetPath,
        annotationfile_path=os.path.join(args.datasetPath,"annotations.txt"),
        num_segments=5,
        frames_per_segment=1,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess,
        test_mode=False
    )

    sample = dataset[2]
    frame_tensor = sample[0]  # tensor of shape (NUM_SEGMENTS*FRAMES_PER_SEGMENT) x CHANNELS x HEIGHT x WIDTH
    label = sample[1]  # integer label
    
    print('Video Tensor Size:', frame_tensor.size())

    def denormalize(video_tensor):
        """
        Undoes mean/standard deviation normalization, zero to one scaling,
        and channel rearrangement for a batch of images.
        args:
            video_tensor: a (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        """
        inverse_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
        return (inverse_normalize(video_tensor) * 255.).type(torch.uint8).permute(0, 2, 3, 1).numpy()
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batchSize,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    valLoader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batchSize,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    args.modelSavePath = os.path.join(args.savePath, 'model')
    os.makedirs(args.modelSavePath, exist_ok=True)
    args.scoreSavePath = os.path.join(args.savePath, 'score.txt')
    modelfiles = glob.glob('%s/model_0*.model'%args.modelSavePath)
    modelfiles.sort()  
    if len(modelfiles) >= 1:
        print("Model %s loaded from previous state!"%modelfiles[-1])
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        s = model(epoch = epoch, **vars(args))
        s.loadParameters(modelfiles[-1])
    else:
        epoch = 1
        s = model(epoch = epoch, **vars(args))

    mAPs = []
    scoreFile = open(args.scoreSavePath, "a+")
    bestmAP = 0

    while True:        
        loss, lr = s.train_network(epoch = epoch, loader = dataloader, **vars(args))
        
        if epoch % args.testInterval == 0:        
            #mAPs, acc = s.evaluate_network(epoch = epoch, loader = valLoader, **vars(args))
            mAPs.append(s.evaluate_network(epoch = epoch, loader = valLoader, **vars(args)))
            if mAPs[-1] > bestmAP:
                bestmAP = mAPs[-1]
                s.saveParameters(args.modelSavePath + "/best.model")
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, mAP %2.2f%%, bestmAP %2.2f%%"%(epoch, mAPs[-1], max(mAPs)))
            scoreFile.write("%d epoch, LR %f, LOSS %f, mAP %2.2f%%, bestmAP %2.2f%%\n"%(epoch, lr, loss, mAPs[-1], max(mAPs)))
            scoreFile.flush()

        if epoch >= args.maxEpoch:
            quit()

        epoch += 1
"""
    for epoch in range(args.maxEpoch):
        for video_batch, labels in dataloader:
            
            
            print(labels)
            print("\nVideo Batch Tensor Size:", video_batch.size())
            print("Batch Labels Size:", labels.size())
            
"""
if __name__=="__main__":
    args = parser()
    main(args)
    

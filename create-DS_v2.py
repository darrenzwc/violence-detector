import cv2
import os
import sys
import glob
from tqdm import tqdm
from alive_progress import alive_bar 
def delete_files(dPath):
    try:
        with alive_bar(title = 'Deleting Files and Directories',spinner='twirls') as bar: 
            # Iterates through all directories and files within
            for dirpath, dirnames, filenames in os.walk(dPath, topdown=False):
            # Delete all files in the directory
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    os.remove(file_path)
            # Delete the directory itself
                for dirname in dirnames:
                    dir_path = os.path.join(dirpath, dirname)
                    os.rmdir(dir_path)
            # Progress bar
            bar()
    except OSError:
        print("unable to remove")
    print("All files and directories from " + dPath + " removed!")
    
PATH_nonviolence = '/notebooks/violence_ds/NonViolence'

os.makedirs('./data/NonViolence',exist_ok=True)
delete_files("data/NonViolence")
delete_files("data/Violence")
for path in tqdm(glob.glob(PATH_nonviolence+'/*')[0:350]):
    fname = os.path.basename(path).split('.')[0]
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    count = 0
    while success:
        if count % 30 == 0:
            cv2.imwrite("./data/NonViolence/{}-{}.jpg".format(fname,str(count).zfill(4)),image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1
        
PATH_violence = '/notebooks/violence_ds/Violence'
        
os.makedirs('./data/Violence',exist_ok=True)
for path in tqdm(glob.glob(PATH_violence+'/*')[0:350]):
    fname = os.path.basename(path).split('.')[0]
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    count = 0
    while success:
        if count % 30 == 0:
            cv2.imwrite("./data/Violence/{}-{}.jpg".format(fname,str(count).zfill(4)),image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1

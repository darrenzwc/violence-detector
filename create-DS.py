from sklearn.model_selection import train_test_split
from alive_progress import alive_bar
import os, cv2, time, numpy as np

# Annotations Path
file_name = "/notebooks/VD-DS/annotations.txt"
# text file will follow the format for every video
# /path/to/video start_frame end_frame classification(0 or 1)
# Parent Directory path to transformed DS
parent_dir = "/notebooks/VD-DS/NonViolent/"
# Path to the directory holding the videos
RawPath = "/notebooks/violence_ds/NonViolence/"
# File label count
fCount = 0

#Delete previous annotations
open('/notebooks/VD-DS/annotations.txt', 'w').close()
#annotations for DS
labels = open("/notebooks/VD-DS/annotations.txt","a")

#Method that will transform the videos into separate
def FrameCapture(path,directory):
    # Input video to capture each frame
    vidObj = cv2.VideoCapture(path)
    # Get width, height, fps, and frames of the video
    w_frame, h_frame = int(vidObj.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, frames = vidObj.get(cv2.CAP_PROP_FPS), vidObj.get(cv2.CAP_PROP_FRAME_COUNT)
    cSize = (240,240)
    # Frame starting num
    frame = 1
    success = 1
    # Binary class 0 = nonviolent or 1 = violent
    bTag = 0
    # Writes down labeling in the annotations.txt file
    if "NonViolent" in directory:
        labels.write("NonViolent/" + dFormat + " " + str(frame) + " ")
    else:
        bTag = 1
        labels.write("Violent/" + dFormat + " " + str(frame) + " ")
    while True:
        # Reads and iterates through each frame in the image
        success, image = vidObj.read()
        # If there are frames availible and video can be opened
        if success and vidObj.isOpened():
            # Creates a photo of each frame @ the directory under format of img_00001.jpg
            image = resizeAndPad(image, cSize)
            cv2.imwrite(os.path.join(directory,"img_%s.jpg"% str(frame).zfill(5)), image)
            frame += 1
        else:
            break
    # Releases access to frames of the video
    vidObj.release()
    # Close any window popups
    cv2.destroyAllWindows()
    # Continues rest of the label with ending frame, classification, and new line
    labels.write(str(frame-1) + " " +str(bTag) + "\n")

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
    
# Iterates through all directories in path_abs
def remove_empty_folders(path_abs):
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.rmdir(path)
            
# Resizes the input img and sizes it to "size" which consists of a tuple
def resizeAndPad(img, size):
    padColor = 0
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:   # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0
            # set pad color
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img
# Deletes all current processed videos inside the data set to avoid errors
delete_files(parent_dir)
# pause 3 seconds
time.sleep(3)
#Processes all the videos that are Non-violent
with alive_bar(1000, title = 'Processing Non-Violent Videos',spinner='twirls') as bar: 
    for file in os.listdir(RawPath):  
        fCount +=1
        #Formats the directory name
        #dFormat produces: 0043_fileName
        dFormat = str(fCount).zfill(4) + "_" + file[:len(file)-4] # file directory name
        nPath = os.path.join(parent_dir, dFormat)
        os.mkdir(nPath, 0o666)
        #capturing
        FrameCapture(RawPath+file,nPath)
        #Progress bar increment
        bar()
remove_empty_folders(parent_dir)
#Processes all the videos that are violent

#Update new parameters for violent folder videos
fCount = 0
parent_dir = "/notebooks/VD-DS/Violent/"
RawPath = "/notebooks/violence_ds/Violence/"
delete_files(parent_dir)
with alive_bar(1000, title = 'Processing Violent Videos',spinner='twirls') as bar: 
    for file in os.listdir(RawPath):  
        fCount +=1
        dFormat = str(fCount).zfill(4) + "_" + file[:len(file)-4] # file directory name
        nPath = os.path.join(parent_dir, dFormat)
        os.mkdir(nPath, 0o666)
        #capturing
        FrameCapture(RawPath+file,nPath)
        # Progress bar increment
        bar()
remove_empty_folders(parent_dir)
print("Processed videos have now fully populated the VD-DS Directory.")
#Closes annotations.txt file and allows changes to be saved
labels.close()


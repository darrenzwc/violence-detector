import matplotlib
matplotlib.use("Agg")

# import the necessary packages 
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import AveragePooling2D
from keras.applications import ResNet50,MobileNetV2,MobileNet,InceptionV3
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.utils import to_categorical 
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import warnings
import argparse
import pickle
import cv2
import os

def parser(): #provide configurations to command line for model

    args = argparse.ArgumentParser(description="Violence Detector")

    args.add_argument("--lr", type=float, default=0.0001, help="Learning Rate")
    args.add_argument('--maxEpoch', type=int, default=10, help='# of epochs')
    args.add_argument('--nDataLoaderThread', type=int, default=4, help='Number of loader threads')
    args.add_argument('--datasetPath', type=str, default="data", help='Path to the Violence Dataset')
    args.add_argument('--savePath', type=str, default="model")

    args = args.parse_args()
    return args

def main(args):
    warnings.filterwarnings('ignore',category=FutureWarning)
    warnings.filterwarnings('ignore',category=DeprecationWarning)
    
    args = {
        "dataset": args.datasetPath,
        "model": "model/violence_model.h5",
        "label-bin": "model/lb.pickle",
        "epochs": args.maxEpoch,
        "plot": "plot.png"

    }

    LABELS = set(["Violence", "NonViolence"])
    # grab the list of images in our dataset directory, then initialize
    # the list of data (i.e., images) and class images
    
    print('-'*100)
    print("[INFO] loading images...")
    print('-'*100)
    imagePaths = list(paths.list_images(args["dataset"]))
    data = []
    labels = []
    
    # loop over the image paths
    for imagePath in tqdm(imagePaths[::]):
        # imagePath : file name ex) V_123
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2] # Violence / NonViolence

        # if the label of the current image is not part of of the labels
        # are interested in, then ignore the image
        if label not in LABELS:
            continue

        # load the image, convert it to RGB channel ordering, and resize
        # it to be a fixed 224x224 pixels, ignoring aspect ratio
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        # update the data and labels lists, respectively
        data.append(image)
        labels.append(label)
        # convert the data and labels to NumPy arrays
    data = np.array(data)
    labels = np.array(labels)
    # perform one-hot encoding on the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)
    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing

    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, train_size=0.75, stratify=labels, random_state=42)
    # initialize the training data augmentation object
    trainAug = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
    # initialize the validation/testing data augmentation object (which
    # we'll be adding mean subtraction to)
    valAug = ImageDataGenerator()
    # define the ImageNet mean subtraction (in RGB order) and set the
    # the mean subtraction value for each of the data augmentation
    # objects
    mean = np.array([123.68, 116.779, 103.939], dtype="float32")
    trainAug.mean = mean
    valAug.mean = mean
    # define the ImageNet mean subtraction (in RGB order) and set the
    # the mean subtraction value for each of the data augmentation
    # objects
    mean = np.array([123.68, 116.779, 103.939], dtype="float32")
    trainAug.mean = mean
    valAug.mean = mean
    # load the InceptionV3 network, ensuring the head FC layer sets are left
    # off
    baseModel = InceptionV3(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))

    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(5, 5))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(512, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(len(lb.classes_), activation="softmax")(headModel)
    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)
    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the training process
    model.trainable = True
    # train the head of the network for a few epochs (all other layers
    # are frozen) -- this will allow the new FC layers to start to become
    # initialized with actual "learned" values versus pure random
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.FalseNegatives(),
                       tf.keras.metrics.Accuracy()])
    print('-'*100)
    print("[INFO] training head...")
    print('-'*100)
    H = model.fit(
        trainAug.flow(trainX, trainY, batch_size=8),
        steps_per_epoch=len(trainX) // 16,
        validation_data=valAug.flow(testX, testY),
        validation_steps=len(testX) // 16,
        epochs=args["epochs"],
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False)
    # evaluate the network
    print('-'*100)
    print("[INFO] evaluating network...")
    print('-'*100)
    predictions = model.predict(testX, batch_size=16)
    print(classification_report(testY.argmax(axis=1),
        predictions.argmax(axis=1), target_names=lb.classes_))
    # serialize the model to disk
    print('-'*100)
    print("[INFO] serializing network...")
    print('-'*100)
    model.save(args["model"])
    
if __name__=="__main__":
    print(tf.config.list_physical_devices('GPU'))
    gpu_device = tf.config.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu_device, True)
    args = parser()
    main(args)
    

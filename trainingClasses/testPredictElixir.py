MAIN_DIR = "C:/Users/kylel/Documents/Development/Clash-Royale-AI-Card-Tracker"

import numpy as np

# Setting up data
import cv2
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.utils import to_categorical
from imutils import paths

from random import randint

# Used for live predictions
import time
from PIL import ImageGrab

# Used for GUI
import tkinter
from PIL import ImageTk
from PIL import Image

# Used for Generating/Labeling Data
from shutil import copyfile
import os
from random import randint

def loadTrainingImagesPredictElixir():

    imagePaths = sorted(list(paths.list_images("trainData2/")))
    x_train = np.zeros((len(imagePaths)*2, 28, 28, 3))

    j = 0

    for i in range(len(imagePaths)):

        # Positive Label

        img = cv2.imread(imagePaths[i])
        arr = img_to_array(img)
        #cv2.imwrite(MAIN_DIR + "/outputImages/deck.png", arr[58:88, 702:1215])

        arr = arr[51:173, 680:1208]

        card = int(imagePaths[i][imagePaths[i].find('/')+1])

        if (card == 0):
            arr = arr[30:50, 50:104]

        elif (card == 1):
            arr = arr[30:50, 109:163]

        elif (card == 2):
            arr = arr[30:50, 170:224]

        elif (card == 3):
            arr = arr[30:50, 228:282]

        elif (card == 4):
            arr = arr[30:50, 287:341]

        elif (card == 5):
            arr = arr[30:50, 346:400]

        elif (card == 6):
            arr = arr[30:50, 405:459]

        elif (card == 7):
            arr = arr[30:50, 464:518]


        img = arr
        img = cv2.resize(img, (28, 28))
        img = img_to_array(img)
        x_train[j] = img

        # Negative Label

        img = cv2.imread(imagePaths[i])
        arr = img_to_array(img)
        #cv2.imwrite(MAIN_DIR + "/outputImages/deck.png", arr[58:88, 702:1215])

        arr = arr[51:173, 680:1208]

        card = int(imagePaths[i][imagePaths[i].find('/')+1])
        nonPlayedCards = np.arange(8)
        nonPlayedCards = nonPlayedCards.tolist()
        nonPlayedCards.remove(card)

        cardNotPlayed = randint(0, 6)

        if (cardNotPlayed == 0):
            arr = arr[30:50, 50:104]

        elif (cardNotPlayed == 1):
            arr = arr[30:50, 109:163]

        elif (cardNotPlayed == 2):
            arr = arr[30:50, 170:224]

        elif (cardNotPlayed == 3):
            arr = arr[30:50, 228:282]

        elif (cardNotPlayed == 4):
            arr = arr[30:50, 287:341]

        elif (cardNotPlayed == 5):
            arr = arr[30:50, 346:400]

        elif (cardNotPlayed == 6):
            arr = arr[30:50, 405:459]

        elif (cardNotPlayed == 7):
            arr = arr[30:50, 464:518]

        img = arr
        img = cv2.resize(img, (28, 28))
        img = img_to_array(img)
        x_train[j+1] = img

        j += 2

    y_train = np.zeros(len(x_train))

    for i in range(len(y_train)):
        y_train[i] = (i+1)%2

    return x_train, y_train

def loadTestingImagesPredictElixir():

    img = cv2.imread(MAIN_DIR + "/outputImages/screen.png")
    arr = img_to_array(img)
    # cv2.imwrite(MAIN_DIR + "/outputImages/deck.png", arr[88:118, 702:1215])

    arr = arr[51:173, 680:1208]

    cv2.imwrite(MAIN_DIR + "/predictElixirOutputImages/output1.png", arr[30:50, 50:104])

    cv2.imwrite(MAIN_DIR + "/predictElixirOutputImages/output2.png", arr[30:50, 109:163])

    cv2.imwrite(MAIN_DIR + "/predictElixirOutputImages/output3.png", arr[30:50, 170:224])

    cv2.imwrite(MAIN_DIR + "/predictElixirOutputImages/output4.png", arr[30:50, 228:282])

    cv2.imwrite(MAIN_DIR + "/predictElixirOutputImages/output5.png", arr[30:50, 287:341])

    cv2.imwrite(MAIN_DIR + "/predictElixirOutputImages/output6.png", arr[30:50, 345:400])

    cv2.imwrite(MAIN_DIR + "/predictElixirOutputImages/output7.png", arr[30:50, 405:459])

    cv2.imwrite(MAIN_DIR + "/predictElixirOutputImages/output8.png", arr[30:50, 464:518])

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

def generateImagesPredictElixir():
    currentNumOfData = len(sorted(list(paths.list_images(MAIN_DIR + "/generatedData/"))))

    print("[INFO] Type anything and press enter to begin...")
    input()

    startTime = time.time()

    i = 0

    while (True):

        if (time.time() - startTime > 1):
            print("--------Captured Data--------")

            im = ImageGrab.grab()
            im.save(MAIN_DIR + "/generatedData/input" + str(i + 1 + currentNumOfData) + ".png")
            i += 1

            startTime = time.time()

def labelTrainingDataPredictElixir():
    imagePaths = sorted(list(paths.list_images(MAIN_DIR + "/generatedData/")))
    currentNumOfLabeledData = len(sorted(list(paths.list_images(MAIN_DIR + "/elixirImages/"))))

    root = tkinter.Tk()
    myFrame = tkinter.LabelFrame(root, text="Unlabeled Data", labelanchor="n")
    myFrame.pack()

    labeledCount = 0

    for i in range(len(imagePaths)):
        img = Image.open(imagePaths[i])
        img.thumbnail((1500, 1500), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        panel = tkinter.Label(myFrame, image=img)
        panel.image = img
        panel.grid(row=0, column=0)
        root.update()

        label = input()

        if (label != 'e'):
            copyfile(imagePaths[i],
                     MAIN_DIR + "/elixirImages/" + label + "input" + str(labeledCount + currentNumOfLabeledData) + ".png")
            labeledCount += 1

        os.remove(imagePaths[i])

def loadTrainingImagesPredictElixir():

    imagePaths = sorted(list(paths.list_images(MAIN_DIR + "/elixirImages/")))
    x_train = np.zeros((len(imagePaths)*2, 28, 28, 3))

    j = 0

    for i in range(len(imagePaths)):

        # Positive Label

        img = cv2.imread(imagePaths[i])
        arr = img_to_array(img)
        cv2.imwrite(MAIN_DIR + "/outputImages/deck.png", arr[31:139, 586:1029])

        arr = arr[31:139, 586:1029]

        card = int(imagePaths[i].split('/')[-1].split('input')[0])

        if (card == 0):
            arr = arr[30:55, 44:88]

        elif (card == 1):
            arr = arr[30:55, 93:138]

        elif (card == 2):
            arr = arr[30:55, 142:188]

        elif (card == 3):
            arr = arr[30:55, 191:236]

        elif (card == 4):
            arr = arr[30:55, 240:286]

        elif (card == 5):
            arr = arr[30:55, 290:335]

        elif (card == 6):
            arr = arr[30:55, 339:384]

        elif (card == 7):
            arr = arr[30:55, 388:435]


        img = arr
        img = cv2.resize(img, (28, 28))
        img = img_to_array(img)
        x_train[j] = img

        # Negative Label

        img = cv2.imread(imagePaths[i])
        arr = img_to_array(img)
        cv2.imwrite(MAIN_DIR + "/outputImages/deck.png", arr[31:139, 586:1029])

        arr = arr[31:139, 586:1029]

        card = int(imagePaths[i].split('/')[-1].split('input')[0])
        nonPlayedCards = np.arange(8)
        nonPlayedCards = nonPlayedCards.tolist()
        nonPlayedCards.remove(card)

        cardNotPlayed = randint(0, 6)

        if (cardNotPlayed == 0):
            arr = arr[30:55, 44:88]

        elif (cardNotPlayed == 1):
            arr = arr[30:55, 93:138]

        elif (cardNotPlayed == 2):
            arr = arr[30:55, 142:188]

        elif (cardNotPlayed == 3):
            arr = arr[30:55, 191:236]

        elif (cardNotPlayed == 4):
            arr = arr[30:55, 240:286]

        elif (cardNotPlayed == 5):
            arr = arr[30:55, 290:335]

        elif (cardNotPlayed == 6):
            arr = arr[30:55, 339:384]

        elif (cardNotPlayed == 7):
            arr = arr[30:55, 388:435]

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
    cv2.imwrite(MAIN_DIR + "/outputImages/deck.png", arr[31:139, 586:1029])

    arr = arr[31:139, 586:1029]

    cv2.imwrite(MAIN_DIR + "/predictElixirOutputImages/output1.png", arr[30:55, 44:88])
    cv2.imwrite(MAIN_DIR + "/predictElixirOutputImages/output2.png", arr[30:55, 93:138])
    cv2.imwrite(MAIN_DIR + "/predictElixirOutputImages/output3.png", arr[30:55, 142:188])
    cv2.imwrite(MAIN_DIR + "/predictElixirOutputImages/output4.png", arr[30:55, 191:236])
    cv2.imwrite(MAIN_DIR + "/predictElixirOutputImages/output5.png", arr[30:55, 240:286])
    cv2.imwrite(MAIN_DIR + "/predictElixirOutputImages/output6.png", arr[30:55, 290:335])
    cv2.imwrite(MAIN_DIR + "/predictElixirOutputImages/output7.png", arr[30:55, 339:384])
    cv2.imwrite(MAIN_DIR + "/predictElixirOutputImages/output8.png", arr[30:55, 388:435])

import os.path

MAIN_DIR = "C:/Users/kylel/Documents/Development/Clash-Royale-AI-Card-Tracker"

import numpy as np

# Used for predictions
from keras.models import load_model

# Used for live predictions
import time
from PIL import ImageGrab

# Used for GUI
import tkinter
from tkinter import ttk
from PIL import ImageTk
from PIL import Image

# Setting up data
import cv2
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.utils import to_categorical
from imutils import paths

from trainingClasses.testPredictCards import loadTestingImagesPredictCards
from trainingClasses.testPredictElixir import loadTestingImagesPredictElixir

def liveBothModelPredicts():

    imagePaths = sorted(list(paths.list_images(MAIN_DIR + "/cardImages/")))
    imageNames = sorted(list(paths.list_images(MAIN_DIR + "/cardImages/")))

    for i in range(len(imageNames)):
        imageNames[i] = os.path.splitext(os.path.basename(imageNames[i]))[0]

    cardCollection = loadCardCollection()

    print("[INFO] loading both networks...")
    model1 = load_model(MAIN_DIR + "/models/predictCards.h5")
    model2 = load_model(MAIN_DIR + "/models/predictElixir.h5")

    opponentCards = ['MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard',
                     'MysteryCard', 'MysteryCard']
    tempOpponentCards = ['MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard',
                         'MysteryCard', 'MysteryCard']

    continuousClassificationCount = [0, 0, 0, 0, 0, 0, 0, 0]
    requiredContinuousClassificationCount = 3

    opponentHand = ['MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard',
                    'MysteryCard', 'MysteryCard']

    # Cards that are placed before getting classified
    pending = []
    tempPending = []

    pendingElixir = 0

    root = tkinter.Tk()
    elixerFrame = tkinter.LabelFrame(root, text="Opponent's Elixer", labelanchor="n")
    elixerFrame.pack()

    myFrame = tkinter.LabelFrame(root, text="Opponent's Cards in Hand", labelanchor="n")
    myFrame.pack()

    myFrame2 = tkinter.LabelFrame(root, text="Opponent's Upcoming Cards", labelanchor="n")
    myFrame2.pack()

    # Temp Uncommented
    myFrame3 = tkinter.LabelFrame(root, text="Opponent's Deck", labelanchor="n")
    myFrame3.pack()

    panel = tkinter.Label(elixerFrame, text='L')
    panel.grid(row=0, column=0)
    root.update()

    for i in range(4):
        img = Image.open(MAIN_DIR + "/cardImages/MysteryCard.png")
        img.thumbnail((128, 128), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        panel = tkinter.Label(myFrame, image=img, borderwidth=10, bg='green')
        panel.image = img
        panel.grid(row=0, column=i)
        root.update()

    for i in range(4):
        img = Image.open(MAIN_DIR + "/cardImages/MysteryCard.png")
        img.thumbnail((128, 128), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        panel = tkinter.Label(myFrame2, image=img, borderwidth=10, bg='orange')
        panel.image = img
        panel.grid(row=0, column=i)
        root.update()

    print("[INFO] Enter the starting elixir to begin..")
    elixir = int(input())

    elixirRatio = 1 / 2.8

    startTime = time.time()
    trueStartTime = time.time()
    snapshotTime = 0.4

    while (True):

        elapsedTime = time.time() - startTime

        if (elapsedTime > 120):
            elixirRatio = 1 / 1.4

        if (elapsedTime > snapshotTime):

            startTime = time.time()

            elixir += elixirRatio * elapsedTime
            if (elixir > 10):
                elixir = 10

            panel = tkinter.Label(elixerFrame, text=format(elixir, '.1f'))
            panel.grid(row=0, column=0)
            root.update()

            im = ImageGrab.grab()
            im.save(MAIN_DIR + "/outputImages/screen.png")
            loadTestingImagesPredictCards()
            loadTestingImagesPredictElixir()

            for i in range(8):

                if (opponentCards[i] != "MysteryCard"):
                    continue

                img = cv2.imread(MAIN_DIR + "/predictCardsOutputImages/output" + str(i + 1) + ".png")
                img = cv2.resize(img, (32, 32))
                img = img.astype("float") / 255.0
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)

                output = model1.predict(img)[0]
                label = output.argmax()

                if (imageNames[label] == "MysteryCard"):
                    continue

                elif (tempOpponentCards[i] == imageNames[label]):
                    if (continuousClassificationCount[i] == requiredContinuousClassificationCount):
                        opponentCards[i] = imageNames[label]
                    else:
                        continuousClassificationCount[i] += 1

                    # Temp Uncommented
                    img = Image.open(imagePaths[label])
                    img.thumbnail((128, 128), Image.LANCZOS)
                    img = ImageTk.PhotoImage(img)
                    panel = tkinter.Label(myFrame3, image=img, borderwidth=10)
                    panel.image = img
                    panel.grid(row=0, column=i)
                    root.update()

                else:
                    tempOpponentCards[i] = imageNames[label]
                    continuousClassificationCount[i] = 0

                labelString = "{}: {:.2f}%".format(imageNames[label], output[label] * 100)

                print(labelString)

            # Move all pending cards to the back

            for i in range(len(pending)):

                if (opponentCards[pending[i]] == "MysteryCard"):
                    tempPending.append(pending[i])
                    continue

                else:
                    opponentHand.pop(0)
                    opponentHand.append(opponentCards[pending[i]])

                    elixir -= cardCollection[opponentCards[pending[i]]]

                if (i == len(pending) - 1 and len(tempPending) == 0):
                    elixir += elixirRatio * (time.time() - pendingElixir)
                    if (elixir > 10):
                        elixir = 10
                    pendingElixir = 0

            pending = tempPending
            tempPending = []

            for i in range(8):
                img = cv2.imread(MAIN_DIR + "/predictElixirOutputImages/output" + str(i + 1) + ".png")
                img = cv2.resize(img, (28, 28))
                img = img.astype("float") / 255.0
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)

                output = model2.predict(img)[0]
                label = output.argmax()
                msg = "Not Placed"

                if (label == 1 or (label == 0 and output[label] < .80)):
                    msg = "Placed"

                    if (opponentCards[i] == "MysteryCard"):
                        if (i not in pending):
                            pending.append(i)
                            if (pendingElixir == 0):
                                pendingElixir = time.time()

                    else:
                        if opponentCards[i] in opponentHand:
                            index = opponentHand.index(opponentCards[i])
                            if index < 4:
                                opponentHand.remove(opponentCards[i])
                                opponentHand.append(opponentCards[i])
                                elixir -= cardCollection[opponentCards[i]]

                labelString = "Card " + str(i + 1) + " - {}: {:.2f}%".format(msg, output[label] * 100)

                print(labelString)

            for i in range(4):
                img = Image.open(MAIN_DIR + "/cardImages/" + opponentHand[i] + ".png")
                img.thumbnail((128, 128), Image.LANCZOS)
                img = ImageTk.PhotoImage(img)
                panel = tkinter.Label(myFrame, image=img, borderwidth=10, bg='green')
                panel.image = img
                panel.grid(row=0, column=i)
                root.update()

            for i in range(4):
                img = Image.open(MAIN_DIR + "/cardImages/" + opponentHand[i + 4] + ".png")
                img.thumbnail((128, 128), Image.LANCZOS)
                img = ImageTk.PhotoImage(img)
                panel = tkinter.Label(myFrame2, image=img, borderwidth=10, bg='orange')
                panel.image = img
                panel.grid(row=0, column=i)
                root.update()

            print("--------Opponent's Deck--------")
            print(opponentCards)
            print("--------Opponent's Hand--------")
            print(opponentHand)
            print("--------Pending--------")
            print(pending)
            print()
            print()

def loadCardCollection():
    data = dict()
    with open('elixirCosts.txt') as raw_data:
        for item in raw_data:
            key, value = item.split(':', 1)
            data[key] = int(value[0:value.find('/')])

    return data


liveBothModelPredicts()
# createCardCollection()
# print(loadCardCollection())

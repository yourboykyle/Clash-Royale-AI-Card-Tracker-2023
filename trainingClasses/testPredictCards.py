MAIN_DIR = "C:/Users/kylel/Documents/Development/Clash-Royale-AI-Card-Tracker"

import numpy as np

# Setting up data
import cv2
from keras.utils import img_to_array
from keras.utils import array_to_img
from keras.utils import to_categorical
from imutils import paths

def loadTrainingImagesPredictCards():
    x_train = np.zeros((118, 32, 32, 3))

    imagePaths = sorted(list(paths.list_images(MAIN_DIR + "/cardImages/")))

    print(len(imagePaths))
    for i in range(len(imagePaths)):
        img = cv2.imread(imagePaths[i])
        img = cv2.resize(img, (32, 32))
        img = img_to_array(img)
        x_train[i] = img

    y_train = np.zeros(len(x_train))

    for i in range(len(y_train)):
        y_train[i] = i

    return x_train, y_train

def loadTestingImagesPredictCards():
    img = cv2.imread(MAIN_DIR + "/outputImages/screen.png")
    arr = img_to_array(img)

    cv2.imwrite(MAIN_DIR + "/outputImages/deck.png", arr[31:139, 586:1029])

    arr = arr[30:140, 585:1030]

    # top Y to bottom Y, left X to right X
    # arr[y:y, x:x]
    cv2.imwrite(MAIN_DIR + "/predictCardsOutputImages/output1.png", arr[53:108, 44:88])
    cv2.imwrite(MAIN_DIR + "/predictCardsOutputImages/output2.png", arr[53:108, 93:138])
    cv2.imwrite(MAIN_DIR + "/predictCardsOutputImages/output3.png", arr[53:108, 142:187])
    cv2.imwrite(MAIN_DIR + "/predictCardsOutputImages/output4.png", arr[53:108, 191:236])
    cv2.imwrite(MAIN_DIR + "/predictCardsOutputImages/output5.png", arr[53:108, 240:286])
    cv2.imwrite(MAIN_DIR + "/predictCardsOutputImages/output6.png", arr[53:108, 290:335])
    cv2.imwrite(MAIN_DIR + "/predictCardsOutputImages/output7.png", arr[53:108, 339:384])
    cv2.imwrite(MAIN_DIR + "/predictCardsOutputImages/output8.png", arr[53:108, 388:435])

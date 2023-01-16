import numpy as np

# Setting up data
import cv2
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from tensorflow.keras.utils import to_categorical
from imutils import paths

def loadTrainingImages1():
    x_train = np.zeros((110, 32, 32, 3))

    imagePaths = sorted(list(paths.list_images("trainData/")))

    for i in range(len(imagePaths)):

        img = cv2.imread(imagePaths[i])
        img = cv2.resize(img, (32, 32))
        img = img_to_array(img)
        x_train[i] = img

    y_train = np.zeros(len(x_train))

    for i in range(len(y_train)):
        y_train[i] = i

    return x_train, y_train

def loadTestingImages1():

    img = cv2.imread("testCNN.png")
    arr = img_to_array(img)
    cv2.imwrite("croppped.png", arr[51:173, 680:1208])

    arr = arr[51:173, 680:1208]

    cv2.imwrite("testData/output1.png", arr[50:138, 50:104])

    cv2.imwrite("testData/output2.png", arr[50:138, 109:163])

    cv2.imwrite("testData/output3.png", arr[50:138, 170:224])

    cv2.imwrite("testData/output4.png", arr[50:138, 228:282])

    cv2.imwrite("testData/output5.png", arr[50:138, 287:341])

    cv2.imwrite("testData/output6.png", arr[50:138, 346:400])

    cv2.imwrite("testData/output7.png", arr[50:138, 405:459])

    cv2.imwrite("testData/output8.png", arr[50:138, 464:518])


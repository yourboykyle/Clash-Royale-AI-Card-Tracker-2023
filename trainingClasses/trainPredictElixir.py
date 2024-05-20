MAIN_DIR = "C:/Users/kylel/Documents/Development/Clash-Royale-AI-Card-Tracker"

import numpy as np

# Training the data
from keras.utils import to_categorical
from AIClass import AI

# Used for aug data gen
from keras.preprocessing.image import ImageDataGenerator

# Used for training
from keras.optimizers.legacy import Adam

# Setting up data
import cv2
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.utils import to_categorical
from imutils import paths

# Used for predictions
from keras.models import load_model

# Used for live predictions
import time
from PIL import ImageGrab

# Used for GUI
import tkinter
from PIL import ImageTk
from PIL import Image

# Use other files
from testPredictElixir import loadTrainingImagesPredictElixir, loadTestingImagesPredictElixir, generateImagesPredictElixir, labelTrainingDataPredictElixir

def trainModelPredictElixir():
    EPOCHS = 150
    INIT_LR = 1e-3
    BS = 8

    print("[INFO] Loading Images")
    x_train, y_train = loadTrainingImagesPredictElixir()
    print(x_train.shape)
    print(y_train.shape)
    print("[INFO] Images have been loaded.")

    x_train /= 255

    y_train = to_categorical(y_train, num_classes=2)

    aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2)

    print("[INFO] compiling model...")
    model = AI.build(width=28, height=28, depth=3, classes=2)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


    print("[INFO] training network...")
    H = model.fit_generator(aug.flow(x_train, y_train, batch_size=BS),
                            validation_data=(x_train, y_train), steps_per_epoch=len(x_train) // BS,
                            epochs=EPOCHS, verbose=1)

    print("[INFO] serializing network...")
    model.save(MAIN_DIR + "/models/predictElixir.h5", overwrite=True, save_format="h5")

def predictElixir():

    loadTestingImagesPredictElixir()

    print("[INFO] loading network...")
    model = load_model(MAIN_DIR + "/models/predictElixir.h5")

    for i in range(8):
        img = cv2.imread(MAIN_DIR + "/predictElixirOutputImages/output" + str(i+1) + ".png")
        orig = img.copy()

        img = cv2.resize(img, (28, 28))
        img = img.astype("float")/255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)


        output = model.predict(img)[0]
        label = output.argmax()
        msg = "Not Placed"

        if (label == 1):
            msg = "Placed"

        print(output)
        print(label)

        label = "Card " + str(i) + " - {}: {:.2f}%".format(msg, output[label] * 100)

        print(label)

        orig = cv2.resize(orig, (400, 400))
        cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Output", orig)
        cv2.waitKey(0)

def predictElixirLive():

    print("[INFO] loading network...")
    model = load_model(MAIN_DIR + "/models/predictElixir.h5")

    opponentHand = ['Card 1', 'Card 2', 'Card 3', 'Card 4', 'Card 5', 'Card 6', 'Card 7', 'Card 8']

    print("[INFO] Type anything and press enter to begin...")
    input()

    startTime = time.time()

    while (True):

        # if (time.time()-startTime > 1):
        if (time.time()-startTime > 0.5): # Try to double capture speed to reduce number of missed placements

            im = ImageGrab.grab()
            im.save(MAIN_DIR + "/outputImages/screen.png")
            loadTestingImagesPredictElixir()

            for i in range(8):
                img = cv2.imread(MAIN_DIR + "/predictElixirOutputImages/output" + str(i+1) + ".png")
                img = cv2.resize(img, (28, 28))
                img = img.astype("float")/255.0
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)

                output = model.predict(img)[0]
                label = output.argmax()
                msg = "Not Placed"

                if (label == 1):
                    msg = "Placed"
                    opponentHand.remove("Card " + str(i+1))
                    opponentHand.append("Card " + str(i+1))

                labelString = "Card " + str(i+1) + " - {}: {:.2f}%".format(msg, output[label] * 100)

                print(labelString)

            print("--------Opponent's Hand--------")
            print(opponentHand)
            print()
            print()

            startTime = time.time()

# Train Predict Elixir Models
# generateImagesPredictElixir()
# labelTrainingDataPredictElixir()
# trainModelPredictElixir()
# predictElixir()
predictElixirLive()

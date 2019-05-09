import os
import numpy as np
import cv2
import cvlib as cv
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.models import Model, Sequential

from .collect_dataset import get_dataset
from .utils import rotateImage

# randomized colors of bounding boxes
COLORS = np.random.uniform(0, 255, size=(2, 1))

def cnn_net():
    ''' Convolutional neural network model to classify specified person. '''

    model = Sequential()
    model.add(Conv2D(filters=32, 
                kernel_size=(3, 3), 
                input_shape=(None, None, 3), 
                padding="same",
                activation='relu',
                data_format='channels_last'))
    model.add(GlobalMaxPooling2D())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    return model


def cnn_net2():
    ''' Convolutional neural network model to classify specified person. '''

    model = Sequential()
    model.add(Conv2D(filters=32, 
                kernel_size=(3, 3), 
                input_shape=(None, None, 3), 
                padding="same",
                activation='relu',
                data_format='channels_last'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(filters=32, 
                kernel_size=(3, 3), 
                padding="same",
                activation='relu',
                data_format='channels_last'))
    model.add(GlobalMaxPooling2D())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    return model


def train_model(source, dataset, weights_filename, epochs, restore=False):
    ''' Train model to detect only specified person. '''

    # load neural network model
    model = cnn_net2()

    # if retrain model weights
    if restore == True:
        model.load_weights(weights_filename)

    # create generator of dataset
    generator = get_dataset(source, dataset)

    # train on each sample from dataset
    for sample in generator:
        image = sample.image
        id = sample.id
        model.fit(image, [id], epochs=epochs, batch_size=1)

    model.save_weights(weights_filename)


def check_model(source, weights_filename, threshold, confidence=0.25, scale=0.3):
    ''' Display video and draw bounding boxes of detected people with
    predicted classes. '''

    # check if video file exists
    if not os.path.isfile(source):
        print("Video file does not exist, exiting")
        exit()

    # read video
    video = cv2.VideoCapture(source)

    # exit if video not opened
    if not video.isOpened():
        print("Could not open video")
        exit()

    # read frame from video
    status, frame = video.read()

    if not status:
        exit()

    # load neural network model
    model = cnn_net2()

    # check if file with weights of model exists
    if not os.path.isfile(weights_filename):
        print("File with weights does not exist, exiting")
        exit()

    model.load_weights(weights_filename)

    # iterate over video frame and predict class of detected people
    while True:
        status, frame = video.read()

        if not status:
            exit()

        # rotate frame to vertical position
        frame = rotateImage(frame, 270)

        # apply object detection
        bbox, label, _ = cv.detect_common_objects(frame, confidence=confidence, model='yolov3-tiny')

        # filter people class
        for i in range(len(label)):
            if label[i] == 'person':

                # extract detected person from video frame
                image = frame[bbox[i][0]:bbox[i][2], bbox[i][1]:bbox[i][3]]

                # reshape to fit neural network input
                image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
                
                # predict class of detected person
                id = model.predict(image)

                # get probability
                id = id.item(0)

                # if probability is greater than threshold draw with different color
                if id > threshold:
                    cv2.rectangle(frame, 
                        (bbox[i][0], bbox[i][1]), 
                        (bbox[i][2], bbox[i][3]), 
                        COLORS[0], 
                        10)
                    cv2.putText(frame, 
                        "class: 1, prob: " + str(id), 
                        (bbox[i][0],bbox[i][1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        COLORS[0], 
                        5)
                else:
                    cv2.rectangle(frame, 
                        (bbox[i][0], bbox[i][1]), 
                        (bbox[i][2], bbox[i][3]), 
                        COLORS[1], 
                        10)
                    cv2.putText(frame, 
                        "class: 0, prob: " + str(1 - id), 
                        (bbox[i][0],bbox[i][1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        COLORS[1], 
                        5)

        frame = cv2.resize(frame, dsize=None, fx=scale, fy=scale)

        # display output
        cv2.imshow("Real-time object detection", frame)

        # press "ESC" to stop
        if cv2.waitKey(0) == 27:
            break

    # release resources
    cv2.destroyAllWindows()

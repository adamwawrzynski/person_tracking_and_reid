import os
import numpy as np
import cv2
import cvlib as cv
import math
from collections import deque

from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.optimizers import Adam

from .collect_dataset import get_dataset
from .collect_dataset import split_dataset
from .collect_dataset import normalize_dataset
from .utils import rotateImage

# randomized colors of bounding boxes
COLORS = ((0,0,255), (255,0,0))


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
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    return model


def cnn_net3():
    ''' Convolutional neural network model to classify specified person. '''

    model = Sequential()
    model.add(Conv2D(filters=64, 
                kernel_size=(5, 5), 
                input_shape=(None, None, 3), 
                padding="same",
                activation='relu',
                data_format='channels_last'))
    model.add(MaxPooling2D((3,3)))
    model.add(Conv2D(filters=64, 
                kernel_size=(5, 5), 
                padding="same",
                activation='relu',
                data_format='channels_last'))
    model.add(GlobalMaxPooling2D())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    return model


def cnn_net4():
    ''' Convolutional neural network model to classify specified person. '''

    model = Sequential()

    model.add(Conv2D(filters=128, 
                kernel_size=(5, 5), 
                input_shape=(None, None, 3), 
                padding="same",
                activation='relu',
                data_format='channels_last'))
    model.add(MaxPooling2D(pool_size=(3,3),
                strides=(2,2)))
    model.add(Conv2D(filters=128, 
                kernel_size=(5, 5), 
                padding="same",
                activation='relu',
                data_format='channels_last'))


    model.add(GlobalMaxPooling2D())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(2, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    return model


def cifar_10_cnn(input_shape, num_classes=1):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('sigmoid'))

    model.summary()

    optimizer = Adam(lr=1e-5)

    model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

    return model


def train_model(dataset,
    model,
    weights_filename,
    epochs,
    restore=False):
    ''' Train model to detect only specified person. '''

    # if retrain model weights
    if restore == True:
        model.load_weights(weights_filename)

    # load images from disk
    class_0, class_1 = get_dataset(dataset)

    # split data to train and validation dataset
    class_0_train, class_1_train, class_0_test, class_1_test = split_dataset(class_0, class_1)

    X_test = np.concatenate((class_0_test, class_1_test))

    y_class_0_test = np.zeros((class_0_test.shape[0], 1))
    y_class_1_test = np.ones((class_1_test.shape[0], 1))

    y_test = np.concatenate((y_class_0_test, y_class_1_test))

    for i in range(0, epochs):
        # balance occurance of classes in training dataset
        X, y = normalize_dataset(class_0_train, class_1_train)

        # run training
        model.fit(X,
            y,
            epochs=1,
            batch_size=25,
            validation_data=(X_test, y_test))

    model.save_weights(weights_filename)


class Detection():
    ''' Class representing detected bounding box. '''
    def __init__(self):
        self.bbox = None
        self.id = None



def iou(bbox1, bbox2):
    ''' Calculate intersect over union for 2 bounding boxes. '''
    bbox1 = [int(x) for x in bbox1]
    bbox2 = [int(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection
    return size_intersection / size_union


def draw_bbox(image, bbox, class_no, prob, color):
    ''' Draw rectangle representing detected object. '''
    cv2.rectangle(image, 
        (bbox[0], bbox[1]), 
        (bbox[2], bbox[3]), 
        color, 
        10)
    cv2.putText(image, 
        "class: "+str(class_no)+", prob: " + "{:.2f}".format(prob),
        (bbox[0],bbox[1]-10),
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        color, 
        5)


def check_model(source,
    model,
    weights_filename,
    image_size,
    start=0,
    threshold=0.5,
    confidence=0.1,
    scale=0.3):
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

    # check if file with weights of model exists
    if not os.path.isfile(weights_filename):
        print("File with weights does not exist, exiting")
        exit()

    model.load_weights(weights_filename)

    counter = 0

    old_detections_list = []

    # iterate over video frame and predict class of detected people
    while True:
        counter += 1
        status, frame = video.read()

        if not status:
            exit()

        # skip number of frames
        if counter < start:
            continue

        # rotate frame to vertical position
        frame = rotateImage(frame, 270)

        # apply object detection
        bbox, label, _ = cv.detect_common_objects(frame, confidence=confidence, model='yolov3-tiny')

        # initialize FIFO buffer
        detections_list = []

        # filter people class
        for i in range(len(label)):
            if label[i] == 'person':

                # extract detected person from video frame
                image = frame[bbox[i][1]:bbox[i][3],bbox[i][0]:bbox[i][2]]
                if image.shape[0] == 0 or image.shape[1] == 0:
                    continue

                if image_size != None:
                    image = cv2.resize(image, dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC)

                # reshape to fit neural network input
                image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
                
                # predict class of detected person
                id = model.predict(image)
                id = id.item(0)

                detection = Detection()
                detection.bbox = bbox[i]

                # if first frame set default weight
                previous_id_weight = 0.5

                # if old detections are present
                if len(old_detections_list) > 0:
                    # number of bounding boxes wich have maximum IoU of each iterations
                    iou_counter = 1
                    max_iou = 0
                    max_iou_bbox = None

                    # for each past frame wich is stored
                    for j in range(0, len(old_detections_list)):

                        # for each detection of past frames starting with the oldest
                        for k in range(len(old_detections_list[j])): 
                            val = iou(bbox[i], old_detections_list[j][k].bbox)
                            if val > max_iou:
                                max_iou = val
                                max_iou_bbox = old_detections_list[j][k]

                        # for the best result in frame update weight
                        if max_iou_bbox is not None and max_iou > 0:
                            iou_counter += 1
                            tmp_bbox = max_iou_bbox.bbox
                            previous_id_weight += math.pow(max_iou, 2) * max_iou_bbox.id

                    # divide weight by number of accumulations
                    previous_id_weight /= iou_counter

                # calulate new probability of classes
                if np.mean((id ,previous_id_weight)) > threshold:
                    detection.id = 1
                else:
                    detection.id = 0

                # add current detection to list of current detections
                detections_list.append(detection)

                # if probability is greater than threshold draw with different color
                if detection.id > threshold:
                    draw_bbox(frame, bbox[i], 1, detection.id, COLORS[0])
                else:
                    draw_bbox(frame, bbox[i], 0, detection.id, COLORS[1])

        frame = cv2.resize(frame, dsize=None, fx=scale, fy=scale)

        # display output
        cv2.imshow("Person ReID", frame)

        # if FIFO is full remove first element
        if len(old_detections_list) > 10:
            old_detections_list.remove(old_detections_list[0])

        # append current detections list to FIFO buffer
        old_detections_list.append(detections_list)

        # press "ESC" to stop
        if cv2.waitKey(0) == 27:
            break

    # release resources
    cv2.destroyAllWindows()

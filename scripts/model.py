import os
import numpy as np
import cv2
import cvlib as cv
import math
from collections import deque
from itertools import combinations
from itertools import product

from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.layers import subtract
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.applications import MobileNetV2

from .collect_dataset import get_dataset
from .collect_dataset import split_dataset
from .collect_dataset import normalize_dataset
from .utils import rotateImage

# randomized colors of bounding boxes
COLORS = ((0,0,255), (255,0,0))


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def siamese_network(input_shape):
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    model = MobileNetV2(include_top=False, pooling="max", weights="imagenet", input_shape=input_shape)

    model.trainable = False

    x = Dense(1280)(model.output)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1280)(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    model = Model(model.input, x)

    encoded_l = model(left_input)
    encoded_r = model(right_input)

    distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([encoded_l, encoded_r])

    siamese_net = Model(inputs=[left_input, right_input],outputs=distance)

    siamese_net.compile(loss=contrastive_loss, optimizer='rmsprop', metrics=[accuracy])

    siamese_net.summary()

    return siamese_net


def siamese_encoder(input_shape):
    model = MobileNetV2(include_top=False, pooling="max", weights="imagenet", input_shape=input_shape)
    return model


def siamese_network_core(input_shape):
    encoded_l = Input(input_shape)
    encoded_r = Input(input_shape)

    distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([encoded_l, encoded_r])

    siamese_net = Model(inputs=[encoded_l, encoded_r],outputs=distance)

    siamese_net.compile(loss=contrastive_loss, optimizer='rmsprop', metrics=[accuracy])

    return siamese_net


def train_siamese_net(dataset,
    model,
    weights_filename,
    epochs,
    use_itertools=True,
    restore=False):
    ''' Train model to detect only specified person. '''

    # if retrain model weights
    if restore == True:
        model.load_weights(weights_filename)

    # load images from disk
    class_0, class_1 = get_dataset(dataset)

    # time window
    window = 10

    counter = 1

    if use_itertools == True:
        valid_combinations = np.asarray(list(product(class_0, class_0)))
        np.append(valid_combinations, np.asarray(list(product(class_1, class_1))))

        invalid_combinations = np.asarray(list(product(class_0, class_1)))

    else:
        valid_combinations = []
        for i in range(len(class_0) - window):
            for j in range(1, window):
                valid_combinations.append([class_0[i], class_0[i+j]])

            print("Creating positive pairs of images: {}/{}".format(counter,
                    len(class_0)),
                    end="\r", flush=True)
            counter += 1

        counter = 1
        for i in range(len(class_1) - window):
            for j in range(1, window):
                valid_combinations.append([class_1[i], class_1[i+j]])

            print("Creating positive pairs of images: {}/{}".format(counter,
                    len(class_1)),
                    end="\r", flush=True)
            counter += 1

        invalid_combinations = np.asarray(invalid_combinations)

    # split data to train and validation dataset
    invalid_train, valid_train, invalid_test, valid_test = split_dataset(invalid_combinations, valid_combinations)

    X_test = np.concatenate((invalid_test, valid_test))

    y_invalid_test = np.zeros((invalid_test.shape[0], 1))
    y_valid_test = np.ones((valid_test.shape[0], 1))

    y_test = np.concatenate((y_invalid_test, y_valid_test))

    for i in range(epochs):
        print("[Epoch] {}/{}\n".format(i+1, epochs))

        # balance occurance of classes in training dataset
        X, y = normalize_dataset(invalid_train, valid_train)
        for j in range(len(X) - 1):

            # run training
            model.fit(x=X[j].tolist(),
                y=y[j],
                epochs=1,
                batch_size=1)

        print("Saved weights of epoch {}".format(str(i+1)))
        model.save(weights_filename+"_"+str(i+1))

        sum = 0
        for i in range(len(X_test)):
            y_pred = model.evaluate(X_test[i].tolist(), y_test[i])
            sum += y_pred[1]
        print("Overall result: {}".format(float(sum)/float(len(X_test))))

    model.save(weights_filename+"_final")

    print("Valid combinations evaluation:\n")
    for i in range(len(valid_combinations)):
        print(model.predict([valid_combinations[i][0], valid_combinations[i][1]]))

    print("\nInvalid combinations evaluation:\n")
    for i in range(len(invalid_combinations)):
        print(model.predict([invalid_combinations[i][0], invalid_combinations[i][1]]))


class Detection():
    ''' Class representing detected bounding box. '''
    def __init__(self):
        self.bbox = None
        self.id = None
        self.image = None
        self.color = None
        self.encoded = None


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
        "class: "+str(class_no),
        (bbox[0],bbox[1]-10),
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        color, 
        5)


def check_siamese_model(source,
    model_core,
    model_encoder,
    image_size,
    pretrained_model,
    weights_filename=None,
    memmory_size=1,
    start=0,
    distance_threshold=70.0,
    iou_threshold=0.7,
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

    if pretrained_model == False:

        # check if file with weights of model exists
        if not os.path.isfile(weights_filename):
            print("File with weights does not exist, exiting")
            exit()

        model_encoder.load_weights(weights_filename)

    counter = 0
    id_counter = 0

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
        bbox, label, _ = cv.detect_common_objects(frame, confidence=confidence, model='yolov3')

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
                    desired_size = max(image.shape[0], image.shape[1])
                    tmp_image = np.zeros((desired_size, desired_size, 3))
                    tmp_image[:image.shape[0], :image.shape[1], :] = image
                    image = cv2.resize(tmp_image, dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC)

                # reshape to fit neural network input
                image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

                # create detection object
                detection = Detection()
                detection.bbox = bbox[i]
                detection.image = image
                detection.encoded = model_encoder.predict(image)

                # if no previous detection add as next class
                if len(old_detections_list) == 0:
                    detection.id = id_counter
                    id_counter += 1
                    detection.color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                    detections_list.append(detection)
                    draw_bbox(frame, detection.bbox, detection.id, 1., detection.color)
                else:

                    iou_weight = []

                    # for each past frame wich is stored
                    for j in range(len(old_detections_list)):
                        iou_weight.append(0)

                        # number of bounding boxes wich have maximum IoU of each iterations
                        iou_counter = 1
                        max_iou = 0
                        max_iou_bbox = None
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
                            iou_weight[j] += max_iou

                    # divide weight by number of accumulations
                    iou_weight = np.array(iou_weight)
                    iou_weight = np.divide(iou_weight, iou_counter)
                    iou_index = np.argmax(iou_weight)

                    distances = []

                    # check if is simmilar to existing classes
                    for i in range(len(old_detections_list)):
                        last_index = len(old_detections_list[i]) - 1
                        distances.append(model_core.predict([old_detections_list[i][last_index].encoded, detection.encoded]))

                    dist = np.array(distances)

                    # if the IoU value is greater than threshold
                    if iou_weight[iou_index] > iou_threshold:
                        # include IoU value in the final score, promote best result
                        dist[iou_index] /= 2

                    # find the smallest distance between images
                    index = np.argmin(dist)

                    # check if distance is acceptable
                    if dist[index][0][0] < distance_threshold:
                        detection.id = old_detections_list[index][last_index].id
                        detection.color = old_detections_list[index][last_index].color
                        detections_list.append(detection)
                        draw_bbox(frame, detection.bbox, detection.id, dist[index], detection.color)
                    else:
                        # if not found simmilar class create new one
                        detection.id = id_counter
                        detection.color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                        detection.encoded = model_encoder.predict(image)
                        id_counter += 1
                        detections_list.append(detection)
                        draw_bbox(frame, detection.bbox, detection.id, 1., detection.color)

        # resize frame to fit to the screen
        frame = cv2.resize(frame, dsize=None, fx=scale, fy=scale)

        # display output
        cv2.imshow("Person ReID", frame)

        # add new detections to FIFO
        for i in range(len(detections_list)):
            if len(old_detections_list) < detections_list[i].id:
                old_detections_list[detections_list[i].id].append(detections_list[i])
            else:
                old_detections_list.append([detections_list[i]])

        # if FIFO is full remove first element
        for i in range(len(old_detections_list)):
            if len(old_detections_list[i]) > memmory_size:
                old_detections_list[i].remove(old_detections_list[i][0])

        # press "ESC" to stop
        if cv2.waitKey(0) == 27:
            break

    # release resources
    cv2.destroyAllWindows()

import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import os
import numpy as np
from .utils import rotateImage
from .utils import rectangleContains


# randomized colors of bounding boxes
COLORS = np.random.uniform(0, 255, size=(2, 1))


class Sample(object):
    ''' Class representing image with class. '''

    def __init__(self):
        self.image = None
        self.id = None

    def __str__(self):
        print("Image: {}".format(self.image.shape))
        print("ID: {}".format(self.id))

    def set_image(self, image):
        self.image = image

    def set_id(self, id):
        self.id = id


# coordinates of bounding box
mouse1X = None
mouse1Y = None
mouse2X = None
mouse2Y = None


def mouse_callback(event,x,y,flags,param):
    ''' Create callback to point bounding box of detected person during manual
     labelling. '''

    global mouse1X, mouse1Y, mouse2X, mouse2Y
    if event == cv2 .EVENT_LBUTTONDOWN:
        if mouse1X == None:
            mouse1X, mouse1Y = x, y
        else:
            mouse2X, mouse2Y = x, y


def mouse_callback_yolo(event,x,y,flags,param):
    ''' Create callback to point bounding box of detected person during manual
    labelling. '''

    global mouse1X, mouse1Y, mouse2X, mouse2Y
    if event == cv2 .EVENT_LBUTTONDOWN:
        mouse1X, mouse1Y = x, y


def manual_label_video(source, dataset, start=0, scale=0.3, color=(255,255,255)):
    ''' Manually label object on video to create train dataset.
    1 left mouse click - specify left right corner of bounding box
    2 left mouse click - specify right bottom corner of bounding box
    q - pass previous bounding box coordinates to the next frame
    w - erase bounding box coordinates
    '''

    global mouse1X, mouse1Y, mouse2X, mouse2Y

    # check if video file exists
    if not os.path.isfile(source):
        print("Video file does not exist, exiting")
        exit()

    # collect train data in file
    file = open(dataset, "a")

    # register mouse callback to mark person
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback)

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

    counter = 0

    # until end of video
    while True:
        counter += 1

        status, frame = video.read()

        if not status:
            exit()

        # skip number of frames
        if counter < start:
            continue
        else:

            # rotate frame to vertical position
            frame = rotateImage(frame, 270)

            file.write('Frame {}\n'.format(counter))
            file.flush()

            frame = cv2.resize(frame, dsize=None, fx=scale, fy=scale)
            cv2.imshow('image', frame)

            # press 'w' to erase previous bounding box
            # press 'ESC' to exit
            key = cv2.waitKey(0)
            if key == 27:
                exit()
            elif key == ord('w'):
                text = str(int(mouse1X/scale))+','+ \
                        str(int(mouse1Y/scale))+','+ \
                        str(int(mouse2X/scale))+','+ \
                        str(int(mouse2Y/scale))
                mouse1X = mouse1Y = mouse2X = mouse2Y = None
            else:
                text = ''

                # all coordinates must be valid
                if mouse1X is None or mouse1Y is None or mouse2X is None or mouse2Y is None:
                    text = 'None'
                else:
                    cv2.rectangle(frame, 
                            (mouse1X, mouse1Y), 
                            (mouse2X, mouse2Y), 
                            color, 
                            2)
                    cv2.imshow('image', frame)
                    text = str(int(mouse1X/scale))+','+ \
                            str(int(mouse1Y/scale))+','+ \
                            str(int(mouse2X/scale))+','+ \
                            str(int(mouse2Y/scale))

            file.write(text+'\n')
            file.flush()

    # release resources
    cv2.destroyAllWindows()
    file.close()



def label_detected_person_on_video(source, dataset, confidence=0.25, start=0, scale=0.3):
    ''' Create dataset of detected people using YOLOv3 by pointing at given 
    bounding box. '''

    global mouse1X, mouse1Y, mouse2X, mouse2Y

    # check if video file exists
    if not os.path.isfile(source):
        print("Video file does not exist, exiting")
        exit()

    # collect dataset in file
    file = open(dataset, "a")

    # register mouse callback to mark person
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback_yolo)

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

    counter = 0

    # until end of video
    while True:
        counter += 1

        status, frame = video.read()

        if not status:
            exit()

        # skip number of frames
        if counter < start:
            continue
        else:

            # rotate frame to vertical position
            frame = rotateImage(frame, 270)

            # apply object detection
            bbox, label, conf = cv.detect_common_objects(frame, confidence=confidence, model='yolov3-tiny')

            # storage for person class values
            new_bbox = []
            new_label = []
            new_conf = []

            # filter people class
            for i in range(len(label)):
                if label[i] == 'person':
                    new_bbox.append(bbox[i])
                    new_label.append(label[i])
                    new_conf.append(conf[i])

            # scale down video frame
            for i in range(len(bbox)):
                for j in range(len(bbox[i])):
                    bbox[i][j] = int(bbox[i][j]*scale)
            frame = cv2.resize(frame, dsize=None, fx=scale, fy=scale)

            # draw bounding box over detected objects
            out = draw_bbox(frame, new_bbox, new_label, new_conf)

            # display output
            cv2.imshow('image', out)

            # press "ESC' to exit
            # press 'w' to erase selected point
            key = cv2.waitKey(0)
            if key == 27:
                exit()
            elif key == ord('w'):
                mouse1X = mouse1Y = mouse2X = mouse2Y = None

            file.write('Frame {}\n'.format(counter))
            file.flush()

            # write corner points and class of each detected person's bounding box
            for i in range(len(bbox)):
                text = str(int(bbox[i][0]))+','+ \
                    str(int(bbox[i][1]))+','+ \
                    str(int(bbox[i][2]))+','+ \
                    str(int(bbox[i][3]))+','
                if rectangleContains(bbox[i], mouse1X, mouse1Y):
                    text += '1'
                else:
                    text += '0'

                file.write(text+'\n')
                file.flush()

            # press "q" to skip
            if cv2.waitKey(0) == ord('q'):
                break

    # release resources
    cv2.destroyAllWindows()
    file.close()



def get_dataset(source, dataset):
    ''' Return generator of samples from dataset. '''

    # check if video file exists
    if not os.path.isfile(source):
        print("Video file does not exist, exiting")
        exit()

    # open dataset of bounding boxes with corresponding classes for frames
    file = open(dataset, "r")

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

    line = ' '

    # until end of file
    while line != '':

        line = file.readline()
        status, frame = video.read()

        if not status:
            exit()

        # rotate frame to vertical position
        frame = rotateImage(frame, 270)

        line = file.readline()

        # until end of file
        while line != '':

            # until next frame
            if not 'Frame' in line:
                x1, y1, x2, y2, id = line.split(",")
                x1 = int(int(x1)/0.3)
                y1 = int(int(y1)/0.3)
                x2 = int(int(x2)/0.3)
                y2 = int(int(y2)/0.3)
                id = int(id)

                # extract detected person from video frame
                image = frame[y1:y2, x1:x2]

                # reshape to fit neural network input
                image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
                sample = Sample()
                # one_hot_vector = np.zeros((1,2))
                # one_hot_vector.itemset(id, 1)
                sample.set_id(id)
                # sample.set_id(one_hot_vector)
                sample.set_image(image)
                yield sample
            else:
                break

            line = file.readline()

    # release resources
    cv2.destroyAllWindows()
    file.close()


def check_label_video(source, dataset, start=0, scale=0.3):
    ''' Display video and collected dataset to validate. '''

    # check if video file exists
    if not os.path.isfile(source):
        print("Video file does not exist, exiting")
        exit()

    # collect dataset in file
    file = open(dataset, "r")

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

    counter = 0

    # until end of video
    while True:
        counter =  counter + 1
        index = '0'

        # skip number of frames
        if counter < start:
            line = file.readline()
            line = file.readline()
            status, frame = video.read()
            continue

        # skip number of frames data
        while index != str(counter):
            line = file.readline()
            text, index = line.split()

        status, frame = video.read()

        if not status:
            exit()

        # rotate frame to vertical position
        frame = rotateImage(frame, 270)

        line = file.readline()

        # if no person selected
        if 'None' in line:
            frame = cv2.resize(frame, dsize=None, fx=scale, fy=scale)
            cv2.imshow('image', frame)
        else:
            x1, y1, x2, y2 = line.split(",")
            cv2.rectangle(frame, 
                    (int(x1), int(y1)), 
                    (int(x2), int(y2)), 
                    COLORS[0], 
                    10)
            frame = cv2.resize(frame, dsize=None, fx=scale, fy=scale)
            cv2.imshow('image', frame)

        # press 'ESC' to exit
        key = cv2.waitKey(0)
        if key != 27:
            frame = cv2.resize(frame, dsize=None, fx=scale, fy=scale)
            cv2.imshow('image', frame)
        else:
            exit()

    # release resources
    cv2.destroyAllWindows()
    file.close()


def check_label_video_from_yolo(source, dataset, start=0, scale=0.3):
    ''' Display video and collected dataset to validate. '''

    # check if video file exists
    if not os.path.isfile(source):
        print("Video file does not exist, exiting")
        exit()

    # read collected dataset
    file = open(dataset, "r")

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

    counter = 0

    # until end of video
    while True:
        counter =  counter + 1
        line = ''

        # skip number of frames
        if counter < start:
            line = file.readline()
            while 'Frame' in line:
                line = file.readline()
            continue

        status, frame = video.read()

        if not status:
            exit()

        # rotate frame to vertical position
        frame = rotateImage(frame, 270)

        line = file.readline()

        # until end of file
        while line != '':

            # until next frame
            if not 'Frame' in line:
                x1, y1, x2, y2, id = line.split(",")

                if '1' in id:
                    cv2.rectangle(frame, 
                        (int(int(x1)/scale), int(int(y1)/scale)), 
                        (int(int(x2)/scale), int(int(y2)/scale)), 
                        COLORS[0], 
                        10)
                else:
                    cv2.rectangle(frame, 
                        (int(int(x1)/scale), int(int(y1)/scale)), 
                        (int(int(x2)/scale), int(int(y2)/scale)), 
                        COLORS[1], 
                        10)
            else:
                break

            line = file.readline()

        # press 'ESC' to exit
        key = cv2.waitKey(0)
        if key != 27:
            frame = cv2.resize(frame, dsize=None, fx=scale, fy=scale)
            cv2.imshow('image', frame)
        else:
            exit()

    # release resources
    cv2.destroyAllWindows()
    file.close()


def process_video(source, dataset, confidence, scale=0.3):
    ''' Detect people on video using pretrained YOLOv3 network. '''

    # check if video file exists
    if not os.path.isfile(source):
        print("Video file does not exist, exiting")
        exit()

    # collect dataset in file
    file = open(dataset, "a")

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

    counter = 0

    # until end of video
    while True:
        counter += 1
        status, frame = video.read()

        if not status:
            exit()

        # rotate frame to vertical position
        frame = rotateImage(frame, 270)

        # apply object detection
        bbox, label, conf = cv.detect_common_objects(frame, confidence=confidence, model='yolov3-tiny')

        # storage for person class values
        new_bbox = []
        new_label = []
        new_conf = []

        file.write("Frame {}\n".format(counter))
        file.flush()

        # filter people class
        for i in range(len(label)):
            if label[i] == 'person':
                new_bbox.append(bbox[i])
                file.write(str(bbox[i]))
                file.flush()
                new_label.append(label[i])
                new_conf.append(conf[i])
                file.write(str(conf[i])+'\n')
                file.flush()

        # scale down video frame
        for i in range(len(bbox)):
            for j in range(len(bbox[i])):
                bbox[i][j] = int(bbox[i][j]*scale)
        frame = cv2.resize(frame, dsize=None, fx=scale, fy=scale)

        # draw bounding box over detected objects
        out = draw_bbox(frame, new_bbox, new_label, new_conf)

        # display output
        cv2.imshow("Real-time object detection", out)

        # press 'q' to stop
        if cv2.waitKey(0) == ord('q'):
            break

    # release resources
    cv2.destroyAllWindows()
    file.close()



def detect_faces_on_video(source, scale=0.3):
    ''' Detect people's faces on video using pretrained YOLOv3 network. '''

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

    counter = 0

    # until end of video
    while True:
        counter += 1
        status, frame = video.read()

        if not status:
            exit()

        # rotate frame to vertical position
        frame = rotateImage(frame, 270)

        # apply object detection
        faces, confidences = cv.detect_face(frame)

        # scale down video frame
        for i in range(len(faces)):
            for j in range(len(faces[i])):
                faces[i][j] = int(faces[i][j]*scale)
        frame = cv2.resize(frame, dsize=None, fx=scale, fy=scale)

        # loop through detected faces
        for face, conf in zip(faces,confidences):

            (startX,startY) = face[0],face[1]
            (endX,endY) = face[2],face[3]

            # draw rectangle over face
            cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        # display output
        cv2.imshow("Real-time object detection", frame)

        # press 'q' to stop
        if cv2.waitKey(0) == ord('q'):
            break

    # release resources
    cv2.destroyAllWindows()
import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import sys
import numpy as np
import argparse
import os


COLORS = np.random.uniform(0, 255, size=(80, 3))


def rotateImage(image, angle):
    height = image.shape[0]
    width = image.shape[1]
    height_big = height * 2
    width_big = width * 2
    image_big = cv2.resize(image, (width_big, height_big))
    image_center = (width_big/2, height_big/2)

    # rotation center
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 0.5)
    result = cv2.warpAffine(image_big, rot_mat, (width_big, height_big), flags=cv2.INTER_LINEAR)

    new_height = result.shape[0]
    new_width = result.shape[1]
    new_image_center = (new_height / 2, new_width / 2)
    if height > width:
        result = result[int(new_image_center[0]-(height/2)):int(new_image_center[0]+(height/2)),
                int(new_image_center[1]-(width/2)):int(new_image_center[1]+(width/2))]
    else:
        result = result[int(new_image_center[0]-(width/2)):int(new_image_center[0]+(width/2)),
                int(new_image_center[1]-(height/2)):int(new_image_center[1]+(height/2))]
    return result


def custom_draw_bbox(img, bbox, old_bbox, labels, confidence, color=(255,255,255)):
    ''' Draw bounding box of detected persons. '''
    for i, label in enumerate(labels):
        if old_bbox != []:
            for j in range(len(old_bbox)):
                max = 0
                max_index = 0
                value = overlapped_area(bbox[i], old_bbox[j])
                if value != None and value > max:
                    max = value
                    max_index = j
        label += ' ' + str(format(confidence[i] * 100, '.2f')) + '%'
        cv2.rectangle(img, (bbox[i][0],bbox[i][1]), (bbox[i][2],bbox[i][3]), color, 2)
        cv2.putText(img, label + str(i), (bbox[i][0],bbox[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


mouse1X = None
mouse1Y = None
mouse2X = None
mouse2Y = None


def mouse_callback(event,x,y,flags,param):
    ''' Create callback to point bounding box of object during manual labelling. '''

    global mouse1X, mouse1Y, mouse2X, mouse2Y
    if event == cv2 .EVENT_LBUTTONDOWN:
        if mouse1X == None:
            mouse1X, mouse1Y = x, y
        else:
            mouse2X, mouse2Y = x, y


def manual_label_video(source, dataset, start=0):
    ''' Manually label object on video to create train dataset.
    1 left mouse click - specify left right corner of bounding box
    2 left mouse click - specify right bottom corner of bounding box
    q - pass previous bounding box coordinates to the next frame
    w - erase bounding box coordinates
    '''

    global mouse1X, mouse1Y, mouse2X, mouse2Y
    if not os.path.isfile(source):
        print("File does not exist, exiting")
        exit()

    # collect train data in file
    file = open(dataset, "a")

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback)

    # Read video
    video = cv2.VideoCapture(source)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        exit()

    # read frame from video
    status, frame = video.read()

    if not status:
        exit()

    counter = 0
    while True:
        counter += 1

        status, frame = video.read()

        if not status:
            exit()

        if counter < start:
            continue
        else:

            # rotate frame to vertical position
            frame = rotateImage(frame, 270)

            file.write("Frame {}\n".format(counter))
            file.flush()

            frame = cv2.resize(frame, dsize=None, fx=0.3, fy=0.3)
            cv2.imshow('image', frame)

            # press "q" to retain previous bounding box
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.rectangle(frame, (mouse1X, mouse1Y), (mouse2X, mouse2Y), (255,255,255), 2)
                cv2.imshow('image', frame)
                text = str(int(mouse1X * (1.0/0.3)))+','+str(int(mouse1Y * (1.0/0.3)))+','+str(int(mouse2X * (1.0/0.3)))+','+str(int(mouse2Y * (1.0/0.3)))
                print(text)
                file.write(text+'\n')
                file.flush()

            # press "w" to erase previous bounding box
            if cv2.waitKey(0) & 0xFF == ord('w'):
                text = str(int(mouse1X * (1.0/0.3)))+','+str(int(mouse1Y * (1.0/0.3)))+','+str(int(mouse2X * (1.0/0.3)))+','+str(int(mouse2Y * (1.0/0.3)))
                print(text)
                file.write(text+'\n')
                file.flush()
                mouse1X = mouse1Y = mouse2X = mouse2Y = None

    # release resources
    cv2.destroyAllWindows()
    file.close()


def check_label_video(source, dataset, start=0):
    ''' Display video and collected bounding boxes to check if data is correct. '''

    if not os.path.isfile(source):
        print("File does not exist, exiting")
        exit()

    file = open(dataset, "r")

    # Read video
    video = cv2.VideoCapture(source)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        exit()

    # read frame from video
    status, frame = video.read()

    if not status:
        exit()

    counter = 0
    while True:
        counter =  counter + 1
        index = '0'

        if counter < start:
            line = file.readline()
            line = file.readline()
            status, frame = video.read()
            continue

        while index != str(counter):
            line = file.readline()
            print(line)
            text, index = line.split()

        status, frame = video.read()

        if not status:
            exit()

        # rotate frame to vertical position
        frame = rotateImage(frame, 270)

        line = file.readline()
        if 'None' in line:
            frame = cv2.resize(frame, dsize=None, fx=0.3, fy=0.3)
            cv2.imshow('image', frame)
        else:
            print(line)
            x1, y1, x2, y2 = line.split(",")
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), 20)
            frame = cv2.resize(frame, dsize=None, fx=0.3, fy=0.3)
            cv2.imshow('image', frame)

        # press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            frame = cv2.resize(frame, dsize=None, fx=0.3, fy=0.3)
            cv2.imshow('image', frame)

    # release resources
    cv2.destroyAllWindows()
    file.close()


def process_video(source, confidence):
    ''' Perform people detection on video. '''

    if not os.path.isfile(source):
        print("File does not exist, exiting")
        exit()

    file = open("data_manual.txt", "w")

    # Read video
    video = cv2.VideoCapture(source)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        exit()

    # read frame from video
    status, frame = video.read()

    if not status:
        exit()

    counter = 0
    while True:
        counter += 1
        status, frame = video.read()

        if not status:
            exit()

        # rotate frame to vertical position
        frame = rotateImage(frame, 270)

        # apply object detection
        bbox, label, conf = cv.detect_common_objects(frame, confidence=confidence, model='yolov3-tiny')

        new_bbox = []
        new_label = []
        new_conf = []

        file.write("Frame {}\n".format(counter))
        file.flush()

        # process only detected people
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
                bbox[i][j] = int(bbox[i][j]*0.3)
        frame = cv2.resize(frame, dsize=None, fx=0.3, fy=0.3)

        # draw bounding box over detected objects
        out = draw_bbox(frame, new_bbox, new_label, new_conf)

        # display output
        cv2.imshow("Real-time object detection", out)

        # press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release resources
    cv2.destroyAllWindows()
    file.close()


if __name__ == '__main__':
    # initiate the parser
    parser = argparse.ArgumentParser()

    # add long and short argument
    parser.add_argument("--source",
                    "-s",
                    help="path to video",
                    dest="source",
                    required=True)

    parser.add_argument("--confidence",
                    "-c",
                    help="confidence level",
                    dest="confidence",
                    default=0.2,
                    required=False)

    # read arguments from the command line
    args = parser.parse_args()

    process_video(args.source, float(args.confidence))
    # manual_label_video(args.source, float(args.confidence))
    # check_label_video(args.source, "data_manual.txt")

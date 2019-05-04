import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import sys
import numpy as np
import argparse
import os


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


def process_video(source, confidence):
    if not os.path.isfile(source):
        print("File does not exist, exiting")
        exit()

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

    while True:
        status, frame = video.read()
        frame = rotateImage(frame, 270)
        frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)

        if not status:
            exit()

        # apply object detection
        bbox, label, conf = cv.detect_common_objects(frame, confidence=confidence, model='yolov3-tiny')

        new_bbox = []
        new_label = []
        new_conf = []

        # process only detected people
        for i in range(len(label)):
            if label[i] == 'person':
                new_bbox.append(bbox[i])
                new_label.append(label[i])
                new_conf.append(conf[i])

        # draw bounding box over detected objects
        out = draw_bbox(frame, new_bbox, new_label, new_conf, write_conf=True)

        # display output
        cv2.imshow("Real-time object detection", out)

        # press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release resources
    cv2.destroyAllWindows()


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

import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import sys
import numpy as np
import argparse
import os
from scripts.collect_dataset import get_dataset
<<<<<<< HEAD
=======
from scripts.collect_dataset import create_dataset
from scripts.collect_dataset import split_dataset
from scripts.collect_dataset import check_label_video_from_yolo
>>>>>>> 760aabf... Add function to create and manipulate fixed size image dataset
from scripts.utils import rotateImage
import scripts.model as model


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


#     check_label_video_from_yolo(args.source, 'data/data.txt', start=0, scale=0.3)
#     model.train_model("/home/adam/person_tracking_and_reid/data/dataset", 
#             model.cnn_net2(), 
#             'data/cnn_net2.dh5', 
#             5, 
#             restore=False)
    model.check_model(args.source, model.cnn_net2(), 'data/cnn_net2.dh5', 0.4)

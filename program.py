import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import sys
import numpy as np
import argparse
import os
from scripts.collect_dataset import get_dataset
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

    # check_label_video_from_yolo(args.source, 'data/data.txt', start=0, scale=0.3)
    model.train_model(args.source, 'data/data.txt', 'data/cnn_net2.dh5', 1, restore=True)
    # model.check_model(args.source, 'data/cnn_net2.dh5', 0.4)

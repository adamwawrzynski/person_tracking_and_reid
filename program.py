#!/bin/python
import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import sys
import numpy as np
import argparse
import os
from scripts.collect_dataset import get_dataset
from scripts.collect_dataset import create_dataset
from scripts.collect_dataset import split_dataset
from scripts.collect_dataset import label_detected_person_on_video
from scripts.collect_dataset import check_label_video_from_yolo
from scripts.utils import rotateImage
import scripts.model as model


# initiate the parser
parser = argparse.ArgumentParser()

# add long and short argument
parser.add_argument("--source",
                "-s",
                help="path to video",
                dest="source",
                required=True)

parser.add_argument("--weight_file",
                "-w",
                help="path to weight file",
                dest="weight_file",
                required=True)

# read arguments from the command line
args = parser.parse_args()

model.check_model(source=args.source,
        model=model.cifar_10_cnn((128,128,3)),
        weights_filename=args.weight_file,
        image_size=128,
        start=0,
        threshold=0.5)

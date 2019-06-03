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

# read arguments from the command line
args = parser.parse_args()

# model.train_siamese_net(dataset="data/dataset_128",
#         model=model.siamese_network((128,128,3)),
#         weights_filename=args.weight_file,
#         epochs=20,
#         restore=True)

model.check_siamese_model(source=args.source,
        model_encoder=model.siamese_encoder((128,128,3)),
        model_core=model.siamese_network_core((1280,)),
        pretrained_model=True,
        image_size=128,
        start=0)

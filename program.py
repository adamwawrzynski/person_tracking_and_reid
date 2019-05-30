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
import scripts.collect_dataset as c_d


# initiate the parser
parser = argparse.ArgumentParser()

# add long and short argument
parser.add_argument("--source",
                "-s",
                help="path to video",
                dest="source",
                required=True)

parser.add_argument("--dataset",
                "-d",
                help="path to dataset",
                dest="dataset",
                required=True)

parser.add_argument("--weight_file",
                "-w",
                help="path to weight file",
                dest="weight_file",
                required=True)

# read arguments from the command line
args = parser.parse_args()

#c_d.create_dataset(source=args.source, destination="data/", dataset=args.dataset)

# c_d.manual_label_video(source=args.source,
#                        dataset="data/data_manual_video2_Chybicki.txt")

# c_d.clean_manual_data(dataset="data/data_manual_video2_Chybicki.txt", id=2)

c_d.create_dataset_from_manual_data(source=args.source, destination="data/class_2/",
                        dataset="data/data_manual_video2_Chybicki.txt")

# model.train_model(dataset="data/",
#         model=model.cifar_10_cnn((128,128,3)),
#         weights_filename=args.weight_file,
#         iterations=10,
#         epochs=1,
#         restore=False)

# model.check_model(source=args.source,
#         model=model.cifar_10_cnn((128,128,3)),
#         weights_filename=args.weight_file,
#         start=0,
#         threshold=0.5)

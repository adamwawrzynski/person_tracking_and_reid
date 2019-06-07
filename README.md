# person_tracking_and_reid

## About

This repository contains application to tracking and reidentification people on
video recording based on Siamese network.

## Quick links

* [Requirements](#requirements)
* [Getting started](#getting-started)
* [Architecture](#architecture)

## Requirements

Program was tested on Ubuntu 18.04 LTS.

To run the **program.py** you need Python3.x (recommended version is
3.6) and modules listed in **requirements.txt** file.

## Getting started

### Installation (Ubuntu)

To install all dependencies you have to run the following commands:

```bash
# install Python3.x nad pip3
sudo apt-get install -y python3 python3-dev python3-pip

# install all dependencies
pip3 install -r requirements.txt
```

Or you can run *setup.sh* script with the following command:

```bash
sudo ./setup.sh
```

### Usage of tool

Program, thanks to **argparse** module, offers simple manual and parameters
validation. You will menu if you give wrong parameters or run program with
**-h** flag:

```bash
./program.py -h
```

And in result you will see this:

```bash
Using TensorFlow backend.
usage: program.py [-h] --source SOURCE
program.py: error: the following arguments are required: --source/-s
```

## Architecture

Based on repository user chunhanl: [ElanGuard_Public](https://github.com/chunhanl/ElanGuard_Public/blob/master/README.md?fbclid=IwAR0VhEa3itqefYbx-h-CYMWeWzfLKuFoKoZGhSixf_56F9b7W7D9xHvhB6Y)

System is based on Siamese network to calculate distance between frames with
detected people. The smalles distance which meets threshold requirements is
classified. Also intersect over union metrics is used to predict class.

As encoder pretrained MobileNetV2 network is used (but can be replaced by any
other network). Decoded images are compared using Euclidean distance and this
score is used to predict appropriate class.

To optimize classification proces, images are decoded only once, and those
vectors are stored in Detection class object.
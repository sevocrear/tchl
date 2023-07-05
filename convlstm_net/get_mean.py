from resnet_plus_lstm import resnet18rnn
from utilities.tee import Tee
from kitti_horizon.kitti_horizon_torch import KITTIHorizon
from utilities.losses import calc_horizon_leftright
from utilities.auc import *
import torch
from torchvision import transforms
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import platform
import logging
logger = logging.getLogger('matplotlib.animation')
logger.setLevel(logging.DEBUG)
hostname = platform.node()
import argparse
import contextlib
import math
import cv2
np.seterr(all='raise')
from glob import glob

parser = argparse.ArgumentParser(description='')
parser.add_argument('--image_width', default=625, type=int, help='image width')
parser.add_argument('--image_height', default=190, type=int, help='image height')
parser.add_argument('--source_path', default="images", type=str, help='source path. Dir for images; video file')
parser.add_argument('--input_src', default="image", type=str, help='video or image')
parser.add_argument('--max_length', default =100, type=int, help='max length of source to be processed')
args = parser.parse_args()

imgs = glob(os.path.join(args.source_path, "*.png")) + glob(os.path.join(args.source_path, "*.jpg"))
imgs = sorted(imgs)

# Choose sequence_length (should be less than number of images)
seq_length = args.max_length

if args.input_src == "image":
        imgs = glob(os.path.join(args.source_path, "*.png")) + glob(os.path.join(args.source_path, "*.jpg"))
        imgs = sorted(imgs)
        
        pixel_mean = np.zeros((seq_length, 3))
        print(pixel_mean)        
        # Choose sequence_length (should be less than number of images)
        seq_length = min(len(imgs), seq_length)

        for idx, image in enumerate(imgs[:seq_length]):
            image = cv2.imread(image)
            
            image = cv2.resize(image, (args.image_width, args.image_height), interpolation =1).astype(np.float32)
            image /= 255.
            pixel_mean[idx, :] = [np.mean(image[:, :, x]) for x in range(image.shape[2])]
            
elif args.input_src == "video":
    cap = cv2.VideoCapture(args.source_path)
    rep, image = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    pixel_mean = np.zeros((seq_length, 3))

    seq_length = min(seq_length, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    idx = 0
    while (cap.isOpened() and idx  < seq_length):
        ret, image = cap.read()
        if not ret:
            break
        image_shape_orig = image.shape
        
        image = cv2.resize(image, (args.image_width, args.image_height), interpolation =1).astype(np.float32)
        image /= 255.
        pixel_mean[idx, :] = [np.mean(image[:, :, x]) for x in range(image.shape[2])]
        
        idx += 1
        
print('pixel mean for the images:')    
print(np.mean(pixel_mean, axis = 0))

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
parser.add_argument('--load', default=None, type=str, help='path to NN model weights')
parser.add_argument('--seqlength', default=10000, type=int, help='maximum frames per sequence')
parser.add_argument('--skip', dest='skip', action='store_true', help='use ConvLSTM with skip connection')
parser.add_argument('--fc', dest='fc', action='store_true', help='FC layers instead of ConvLSTM')
parser.add_argument('--lstm_state_reduction', default=4., type=float, help='')
parser.add_argument('--lstm_depth', default=2, type=int, help='number of stacked ConvLSTM cells')
parser.add_argument('--cpu', dest='cpu', action='store_true', help='use CPU only')
parser.add_argument('--gpu', default='0', type=str, help='which GPU to use')
parser.add_argument('--convlstm', dest='convlstm', action='store_true', help='use ConvLSTM')
parser.add_argument('--meanmodel', dest='meanmodel', action='store_true', help='')
parser.add_argument('--simple_skip', dest='simple_skip', action='store_true', help='naive skip connection')
parser.add_argument('--image_width', default=625, type=int, help='image width')
parser.add_argument('--image_height', default=190, type=int, help='image height')
parser.add_argument('--image_path', default="image.png", type=str, help='image path')
parser.add_argument('--whole', dest='whole_sequence', action='store_true', help='process whole sequence at once')
parser.add_argument('--res_dir', default = "results", type=str, help='dir to save imgs and video to')
parser.add_argument('--fps', default =5, type=int, help='fps of the video to be saved')
args = parser.parse_args()

# CREATE DIRECTORY FOR RESULTS
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir)
    
# SELECT DEVICE
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
if args.cpu:
    device = torch.device('cpu', 0)
else:
    device = torch.device('cuda', 0)


seq_length = args.seqlength
whole_sequence = args.whole_sequence

# LOAD MODEL
baseline_angle = 0.013490
baseline_offset = -0.036219

if args.load is not None:
    model = resnet18rnn(use_fc=args.fc,
                        use_convlstm=args.convlstm,
                        lstm_skip=args.skip,
                        lstm_depth=args.lstm_depth,
                        lstm_state_reduction=args.lstm_state_reduction,
                        lstm_simple_skip=args.simple_skip).to(device)

    print("load weights from ", args.load)
    checkpoint = torch.load(args.load, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
else:
    model = None

# IMG PIXEL MEAN
pixel_mean = [0.362365, 0.377767, 0.366744]


# FUNCTION TO CALCULATE HORIZON LINE Y-s
calc_hlr = calc_horizon_leftright(args.image_width, args.image_height)

with torch.no_grad():
    imgs = glob(os.path.join(args.image_path, "*.png")) + glob(os.path.join(args.image_path, "*.jpg"))
    imgs = sorted(imgs)
    
    # Choose sequence_length (should be less than number of images)
    seq_length = min(len(imgs), seq_length)
    
    # Pack images into Numpy Array
    images = np.zeros((seq_length, 3, args.image_height, args.image_width)).astype(np.float32)
    for idx, image in enumerate(imgs[:seq_length]):
        image = cv2.imread(image)
        image_shape_orig = image.shape
        image = cv2.resize(image, (args.image_width, args.image_height), interpolation =1)
        image = np.transpose(image, [2, 0, 1])/255.
        images[idx,:,:,:] = image
    
    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(os.path.join(args.res_dir,'result.mp4'), fourcc, args.fps, (image_shape_orig[1], image_shape_orig[0]))
    
    # Process images
    images = torch.tensor(images).unsqueeze(0)
    if whole_sequence:
        output_offsets, output_angles = model(images.to(device))
        output_offsets = output_offsets.detach()
        output_angles = output_angles.detach()

    # Calculate horizons for the sequence
    for si in range(images.shape[1]):

        image = images.numpy()[0,si,:,:,:].transpose((1,2,0))
        image_draw = image.copy()
        image_draw[:,:,0] += pixel_mean[0]
        image_draw[:,:,1] += pixel_mean[1]
        image_draw[:,:,2] += pixel_mean[2]
        image_draw *= 255.
        
        width = image.shape[1]
        height = image.shape[0]

        if args.load is not None or args.meanmodel:
                if args.load is not None:
                    if not whole_sequence:

                        output_offsets, output_angles = model(images[:,si,:,:,:].unsqueeze(1).to(device))
                        offset_estm = output_offsets[0,0].cpu().detach().numpy().squeeze()
                        angle_estm = output_angles[0,0].cpu().detach().numpy().squeeze()

                        yle, yre = calc_hlr(output_offsets[0, 0], output_angles[0, 0])
                    else:

                        yle, yre = calc_hlr(output_offsets[0, si], output_angles[0, si])

                        offset_estm = output_offsets[0,si].cpu().detach().numpy().squeeze().copy()
                        angle_estm = output_angles[0,si].cpu().detach().numpy().squeeze().copy()
                else:
                    offset_estm = baseline_offset
                    angle_estm = baseline_angle
                    yle, yre = calc_hlr(torch.from_numpy(np.array([offset_estm])), torch.from_numpy(np.array([angle_estm])))
                
                offset_estm += 0.5
                offset_estm *= height

                estm_mp = np.array([width/2., offset_estm])
                estm_nv = np.array([np.sin(angle_estm), np.cos(angle_estm)])
                estm_hl = np.array([estm_nv[0], estm_nv[1], -np.dot(estm_nv, estm_mp)])
                estm_h1 = np.cross(estm_hl, np.array([1, 0, 0]))
                estm_h2 = np.cross(estm_hl, np.array([1, 0, -width]))
                estm_h1 /= estm_h1[2]
                estm_h2 /= estm_h2[2]
                estm_h1 = estm_h1.astype(int)
                estm_h2 = estm_h2.astype(int)
                cv2.line(image_draw, (estm_h1[0], estm_h1[1]), (estm_h2[0], estm_h2[1]), color = (255, 255, 255), thickness = 3)
                img_to_save = cv2.resize(image_draw, (image_shape_orig[1], image_shape_orig[0]), interpolation = 1)
                cv2.imwrite(os.path.join(args.res_dir,f'out_{si}.png'), img_to_save)
                out_video.write(img_to_save.astype(np.uint8))
                print(estm_h1, estm_h2)
                print(yle,yre)
out_video.release()

                
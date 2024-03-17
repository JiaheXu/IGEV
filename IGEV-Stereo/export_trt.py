# import torch_tensorrt
# torch_tensorrt.logging.set_reportable_log_level(torch_tensorrt.logging.Level.Debug)
import sys
sys.path.append('core')
# DEVICE = 'cpu'
DEVICE = 'cuda'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from igev_stereo import IGEVStereo
from utils.utils import InputPadder
from PIL import Image


# from PIL import Image
from matplotlib import pyplot as plt
import os
import cv2
import time

# ros things
import message_filters
import rospy

from cv_bridge import CvBridge
bridge = CvBridge()

from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct

from lightning import Trainer, LightningModule
# from dsta_mvs.test.utils import *

import deform_conv2d_onnx_exporter
deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()

from torch2trt import torch2trt

from igev_stereo import IGEVStereo

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):

    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))
    model = model.module
    model.eval().cuda()
    # model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    left_images = sorted(glob.glob(args.left_imgs, recursive=True))
    right_images = sorted(glob.glob(args.right_imgs, recursive=True))
    # print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

    image1 = load_image(left_images[0])
    image2 = load_image(right_images[0])

    padder = InputPadder(image1.shape, divis_by=32)
    image1, image2 = padder.pad(image1, image2)
    # print("image1: ", image1.shape)
    # print("image2: ", image2)
    # with torch.no_grad():
    #     disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
    #     disp = disp.cpu().numpy()
    #     disp = padder.unpad(disp)
    with torch.no_grad():
        model_trt = torch2trt(model, [image1,image2])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/sceneflow/sceneflow.pth')
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')

    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="./demo-imgs/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="./demo-imgs/*/im1.png")

    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/data/Middlebury/trainingH/*/im0.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/data/Middlebury/trainingH/*/im1.png")
    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/data/ETH3D/two_view_training/*/im0.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/data/ETH3D/two_view_training/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="./demo-output/")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    
    args = parser.parse_args()

    Path(args.output_directory).mkdir(exist_ok=True, parents=True)

    demo(args)

# polygraphy surgeon sanitize IGEV_stereo.onnx --fold-constants -o step2.onnx
# polygraphy convert step2.onnx --fp16 -o IGEV_stereo.engine
# export CUDA_HOME=/usr/local/cuda-12.3
# export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.3
# export CMAKE_CUDA_COMPILER=/usr/local/cuda-12.3/bin/nvcc

# cmake -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.3 -D CMAKE_CUDA_COMPILER=/usr/local/cuda-12.3/bin/nvcc -B build .
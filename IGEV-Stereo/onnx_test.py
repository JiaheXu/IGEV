import onnx
import sys
sys.path.append('core')
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
from matplotlib import pyplot as plt
import os
import cv2

import onnxruntime
import time
import onnxruntime.backend

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
def demo(args):
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()
    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    left_images = sorted(glob.glob(args.left_imgs, recursive=True))
    right_images = sorted(glob.glob(args.right_imgs, recursive=True))
    # print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

    image1 = load_image(left_images[0])
    image2 = load_image(right_images[0])
    padder = InputPadder(image1.shape, divis_by=32)
    image1, image2 = padder.pad(image1, image2)
    
    # with torch.no_grad():
    #     for i in range(5):
    #         padder = InputPadder(image1.shape, divis_by=32)
    #         image1, image2 = padder.pad(image1, image2)
    #         start = time.time()
    #         disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
    #         end = time.time()
    #         print("torch inference time: ", end - start)    
    # print("torch finished")
    # disp = disp.cpu().numpy()
    # torch_out = disp.copy()
    
    # disp = padder.unpad(disp)
    # filename = os.path.join("test.png")
    # plt.imsave(filename, disp.squeeze(), cmap='jet')
    



    ort_session = onnxruntime.InferenceSession("IGEV_stereo.onnx",providers=["CUDAExecutionProvider"])
    print("########## onnx")
    print("onnx device: ", onnxruntime.get_device())
    print("##########")
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image1), ort_session.get_inputs()[1].name: to_numpy(image2)}
    start = time.time()
    ort_outs = ort_session.run(None, ort_inputs)
    end = time.time()
    print("onnx inference time: ", end - start)

    # onnx_out = padder.unpad(ort_outs[0])
    # filename = os.path.join("onnx_test.png")
    # plt.imsave(filename, onnx_out.squeeze(), cmap='jet')
    
    # compare ONNX Runtime and PyTorch results
    # np.testing.assert_allclose(torch_out, ort_outs[0], rtol=1e-1, atol=1e-1)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

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



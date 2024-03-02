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
# from PIL import Image
from matplotlib import pyplot as plt
import os
import cv2
import time

# ros things
import message_filters
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
bridge = CvBridge()

from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct

class igev_stereo():
    def __init__(self, args):

        self.args = args
        self.model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
        self.model.load_state_dict(torch.load(args.restore_ckpt))

        self.model = self.model.module
        self.model.to(DEVICE)
        self.model.eval()

        cameraMatrix1 = np.array([
            [1078.591064453125, 0., 1098.9085693359375],
            [0. ,1078.591064453125,  640.240966796875],
            [0., 0., 1.0]
        ])
        distCoeffs1 = np.array( [0., 0., 0., 0., 0.] )

        cameraMatrix2 = np.array([
            [1078.591064453125, 0., 1098.9085693359375],
            [0. ,1078.591064453125,  640.240966796875],
            [0., 0., 1.0]
        ])
        distCoeffs2 = np.array( [0., 0., 0., 0., 0.] )

        imageSize = (1242, 2208)
        R = np.array([ 
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],            
        ])
        T = np.array([ 
            [-0.12],
            [0.],
            [0.]            
        ])
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T)
        self.Q = Q
        print("Q: ", Q)
        self.current_time = time.time()
        self.previous_time = time.time() - 1.0
        #Todo: feed a startup all zero image to the network
        self.cam1_sub = message_filters.Subscriber(args.left_topic, Image)
        self.cam2_sub = message_filters.Subscriber(args.right_topic, Image)


        self.point_cloud_pub = rospy.Publisher("zed2/point_cloud2", PointCloud2, queue_size=2)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.cam1_sub, self.cam2_sub], 10, 1, allow_headerless=True)
        self.ts.registerCallback(self.callback)

    def load_image(self, img):
        img = img.astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(DEVICE)

    def callback(self, cam1_msg, cam2_msg):
        with torch.no_grad():
            image1 = bridge.imgmsg_to_cv2(cam1_msg)
            image1_np = np.array(image1[:,:,0:3])
            image2 = bridge.imgmsg_to_cv2(cam2_msg)
            image2_np = np.array(image2[:,:,0:3])

            if(self.args.downsampling > 1):
                image1_np = np.array(image1[::self.args.downsampling, ::self.args.downsampling, :])
                image2_np = np.array(image2[::self.args.downsampling, ::self.args.downsampling, :])
                print("downsampled:")
            print("image1.shape: ", image1_np.shape)
            print("image2.shape: ", image2_np.shape)

            
            image1= self.load_image(image1_np)
            image2= self.load_image(image2_np)
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            disp = self.model(image1, image2, iters=args.valid_iters, test_mode=True)
            disp = disp.cpu().numpy()
            disp = padder.unpad(disp)
            # print("disp: ", disp.shape)
            disp = np.squeeze(disp)
            image_3d = cv2.reprojectImageTo3D(disp, self.Q)
            print("image_3d: ", image_3d.shape)            
            # rgb point cloud, reference : https://gist.github.com/lucasw/ea04dcd65bc944daea07612314d114bb
            points = []
            lim = 8
            for i in range( image_3d.shape[0] ):
                for j in range(image_3d.shape[1]):
                        x = image_3d[i][j][0]
                        y = image_3d[i][j][1]
                        z = image_3d[i][j][2]
                        b = image1_np[i][j][0]
                        g = image1_np[i][j][1]
                        r = image1_np[i][j][2]
                        a = 255
                        # print r, g, b, a
                        rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
                        # print hex(rgb)
                        pt = [x, y, z, rgb]
                        points.append(pt)
            print("finished")   
            fields = [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    # PointField('rgb', 12, PointField.UINT32, 1),
                    PointField('rgba', 12, PointField.UINT32, 1),
                    ]

            header = Header()
            header.frame_id = "map"
            
            pc2 = point_cloud2.create_cloud(header, fields, points)
            pc2.header.stamp = rospy.Time.now()
            self.point_cloud_pub.publish(pc2)




    def run(self):
        rospy.spin()  



def demo(args):
    rospy.init_node("igev_node")
    igev_stereo_node = igev_stereo(args)
    igev_stereo_node.run()



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

    parser.add_argument('--left_topic', type=str, default="/zedA/zed_node_A/left/image_rect_color", help="left cam topic")
    parser.add_argument('--right_topic', type=str, default="/zedA/zed_node_A/right/image_rect_color", help="right cam topic")
    parser.add_argument('--downsampling', type=int, default=1, help="downsampling image dimension")
    args = parser.parse_args()

    Path(args.output_directory).mkdir(exist_ok=True, parents=True)

    demo(args)

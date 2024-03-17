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
        
        self.cameraMatrix1 = []
        self.cameraMatrix2 = []
        
        self.distCoeffs1 = []
        self.distCoeffs2 = []
        # 2k setting, need to modify into yaml file in the future 1242 * 2208
        self.cameraMatrix1.append( np.array([
            [1056.7081298828125, 0., 1102.6148681640625],
            [0. ,1056.7081298828125,  612.7953491210938],
            [0., 0., 1.0]
        ]) )
        self.distCoeffs1.append( np.array( [0., 0., 0., 0., 0.] ) )
        self.cameraMatrix2.append( np.array([
            [1056.7081298828125, 0., 1102.6148681640625],
            [0. ,1056.7081298828125,  612.7953491210938],
            [0., 0., 1.0]
        ]) )
        self.distCoeffs2.append( np.array( [0., 0., 0., 0., 0.] ) )

        # 1080p setting, need to modify into yaml file in the future 1080*1920
        self.cameraMatrix1.append( np.array([
            [1064.29833984375, 0., 958.1749267578125],
            [0. ,1064.29833984375,  532.0037231445312],
            [0., 0., 1.0]
        ]) )
        self.distCoeffs1.append( np.array( [0., 0., 0., 0., 0.] ) )
        self.cameraMatrix2.append( np.array([
            [1064.29833984375, 0., 958.1749267578125],
            [0. ,1064.29833984375,  532.0037231445312],
            [0., 0., 1.0]
        ]) )
        self.distCoeffs2.append( np.array( [0., 0., 0., 0., 0.] ) )

        # 720p setting, need to modify into yaml file in the future 720*1280
        self.cameraMatrix1.append( np.array([
            [525.5062255859375, 0., 637.7863159179688],
            [0. ,525.5062255859375,  354.01727294921875],
            [0., 0., 1.0]
        ]) )
        self.distCoeffs1.append( np.array( [0., 0., 0., 0., 0.] ) )
        self.cameraMatrix2.append( np.array([
            [525.5062255859375, 0., 637.7863159179688],
            [0. ,525.5062255859375,  354.01727294921875],
            [0., 0., 1.0]
        ]) )
        self.distCoeffs2.append( np.array( [0., 0., 0., 0., 0.] ) )

        self.imageSize = []
        self.imageSize.append( (1242, 2208) )
        self.imageSize.append( (1080, 1920) )
        self.imageSize.append( (720, 1280) )
        self.R = np.array([ 
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],            
        ])
        self.T = np.array([ 
            [-0.12],
            [0.],
            [0.]            
        ])

        self.Q = []
        for i in range ( len( self.imageSize ) ):
            R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(self.cameraMatrix1[i], self.distCoeffs1[i], self.cameraMatrix2[i], self.distCoeffs2[i], self.imageSize[i], self.R, self.T)
            self.Q.append(Q)
 
        #Todo: feed a startup all zero image to the network
        self.cam1_sub = message_filters.Subscriber(args.left_topic, Image)
        self.cam2_sub = message_filters.Subscriber(args.right_topic, Image)
        self.depth_sub = message_filters.Subscriber(args.depth_topic, Image)

        self.disparity_pub = rospy.Publisher("zed2/disparity", PointCloud2, queue_size=1)

        self.point_cloud_pub = rospy.Publisher("zed2/point_cloud2", PointCloud2, queue_size=1)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.cam1_sub, self.cam2_sub, self.depth_sub], 10, 1, allow_headerless=True)
        self.ts.registerCallback(self.callback)

    def load_image(self, img):
        img = img.astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(DEVICE)

    def disparity_to_depth(self, disparity):
        focal_length = 525.5062255859375
        
        if(disparity.shape[0] == 1080):
            focal_length = 1064.29833984375
        if(disparity.shape[0] == 1242):
            focal_length = 1056.7081298828125
        depth = (0.12 * 525.5062255859375) / disparity
        return depth

    def compare_depth(self, depth1, depth2):
        if(depth1.shape != depth2.shape):
            print("different img dims!!!!!")
            return None
        np_nan = np.isnan(depth1)
        # print("np.isnan(depth1): ", np.logical_not(np_nan) )
        depth1_valid =  np.logical_and( np.logical_not(np.isnan(depth1)), np.logical_not(np.isinf(depth1)) )
        depth2_valid =  np.logical_and( np.logical_not(np.isnan(depth2)), np.logical_not(np.isinf(depth2)) )
        common_ele = np.logical_and( depth1_valid, depth2_valid )

        print("mean: ", np.mean( np.abs(depth1[common_ele] - depth2[common_ele])) )
        diff = np.zeros( depth2.shape )
        diff[common_ele] = np.abs(depth1[common_ele] - depth2[common_ele])
        return diff

    def callback(self, cam1_msg, cam2_msg, depth_msg):

        print("callback")
        with torch.no_grad():
            image1 = bridge.imgmsg_to_cv2(cam1_msg)
            image1_np = np.array(image1[:,:,0:3])
            image2 = bridge.imgmsg_to_cv2(cam2_msg)
            image2_np = np.array(image2[:,:,0:3])

            image_depth = bridge.imgmsg_to_cv2(depth_msg)
            image_depth_np = np.array(image_depth)
            depth_valid =  np.logical_and( np.logical_not(np.isnan(image_depth_np)), np.logical_not(np.isinf(image_depth_np)) )

            if(self.args.downsampling > 1):
                image1_np = np.array(image1[::self.args.downsampling, ::self.args.downsampling, :])
                image2_np = np.array(image2[::self.args.downsampling, ::self.args.downsampling, :])
                print("downsampled:")

            # print("image1.shape: ", image1_np.shape)
            # print("image2.shape: ", image2_np.shape)
            cv2.imwrite('img1.png', image1_np)
            cv2.imwrite('img2.png', image2_np)

            image1= self.load_image(image1_np)
            image2= self.load_image(image2_np)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            start = time.time()
            disp = self.model(image1, image2, iters=args.valid_iters, test_mode=True)
            end = time.time()
            print("torch inference time: ", end - start)
            
            disp = disp.cpu().numpy()
            disp = padder.unpad(disp)
            disp = disp.squeeze()
            plt.imsave("IGEV.png", disp.squeeze(), cmap='jet')
            
            # stereo = cv2.StereoBM_create(numDisparities=96, blockSize=15)
            # image1_gray = cv2.cvtColor(image1_np, cv2.COLOR_RGB2GRAY)
            # image2_gray = cv2.cvtColor(image2_np, cv2.COLOR_RGB2GRAY)
            # disparity = stereo.compute(image1_gray, image2_gray)
            # plt.imsave("opencv_disp.png",disparity,cmap='jet')

            igev_depth = self.disparity_to_depth(disp)
            diff  = self.compare_depth(igev_depth, image_depth_np)

            diff_grayscale = ( diff - np.min(diff) ) / ( np.max(diff) - np.min(diff)) * 255
            diff_grayscale[ diff>0.01 ] = 255
            diff_grayscale[ diff<=0.01 ] = 0
            plt.imsave("diff.png", diff_grayscale, cmap='gray')

            

            # gray_frame = diff   #cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            # fig, ax = plt.subplots()
            # cs = ax.add_image(gray_frame)
            # fig.colorbar(cs)
            # plt.savefig("diff.png")

            # print("difference: ", diff )
            # np.testing.assert_allclose(igev_depth, image_depth_np[0], rtol=10, atol=0.1)

            # print("disparity shape: ", disp.shape)
            # print("image_depth shape:", image_depth_np.shape)
            # plt.imsave("IGEV.png", disp.squeeze(), cmap='jet')

            # image_3d = cv2.reprojectImageTo3D(disp, self.Q)
            # print("image_3d: ", image_3d.shape)            
            # # rgb point cloud, reference : https://gist.github.com/lucasw/ea04dcd65bc944daea07612314d114bb
            # points = []
            # lim = 8
            # for i in range( image_3d.shape[0] ):
            #     for j in range(image_3d.shape[1]):
            #             x = image_3d[i][j][0]
            #             y = image_3d[i][j][1]
            #             z = image_3d[i][j][2]
            #             b = image1_np[i][j][0]
            #             g = image1_np[i][j][1]
            #             r = image1_np[i][j][2]
            #             a = 255
            #             # print r, g, b, a
            #             rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
            #             # print hex(rgb)
            #             pt = [x, y, z, rgb]
            #             points.append(pt)
            # print("finished")   
            # fields = [PointField('x', 0, PointField.FLOAT32, 1),
            #         PointField('y', 4, PointField.FLOAT32, 1),
            #         PointField('z', 8, PointField.FLOAT32, 1),
            #         # PointField('rgb', 12, PointField.UINT32, 1),
            #         PointField('rgba', 12, PointField.UINT32, 1),
            #         ]

            # header = Header()
            # header.frame_id = "map"
            
            # pc2 = point_cloud2.create_cloud(header, fields, points)
            # pc2.header.stamp = rospy.Time.now()
            # self.point_cloud_pub.publish(pc2)




    def run(self):
        rospy.spin()  



def demo(args):
    rospy.init_node("igev_node")
    igev_stereo_node = igev_stereo(args)
    igev_stereo_node.run()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/sceneflow/sceneflow.pth')
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/middlebury/middlebury.pth')
    # parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/eth3d/eth3d.pth')

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

    parser.add_argument('--left_topic', type=str, default="/zed2/zed_node/left/image_rect_color", help="left cam topic")
    parser.add_argument('--right_topic', type=str, default="/zed2/zed_node/right/image_rect_color", help="right cam topic")
    parser.add_argument('--depth_topic', type=str, default="/zed2/zed_node/depth/depth_registered", help="depth cam topic")
    parser.add_argument('--downsampling', type=int, default=1, help="downsampling image dimension")
    args = parser.parse_args()

    Path(args.output_directory).mkdir(exist_ok=True, parents=True)

    demo(args)

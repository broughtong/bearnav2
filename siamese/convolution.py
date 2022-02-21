import rospy
from std_msgs.msg import Float32MultiArray
from stroll_bearnav.msg import FeatureArray
import cv2
import os
import numpy as np
from torchvision import transforms
from model import Siamese, load_model, get_custom_CNN
import torch as t


CUTS = True
CUTS_NUM = 3


device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")


class Node:
    def __init__(self):
        rospy.init_node("repr_convolution", anonymous=True)
        # init neural network
        backbone = get_custom_CNN()
        model = Siamese(backbone).to(device)
        file_path = os.path.dirname(os.path.abspath(__file__))
        self.model = load_model(model, os.path.join(file_path, "../models/model_47.pt")).to(device)
        self.model.eval()
        self.map_tensor = None

        # Publishers
        self.pub = rospy.Publisher('/heading_histogram', Float32MultiArray, queue_size=1)
        if CUTS:
            self.pub_cuts = rospy.Publisher('/cuts_value', Float32, queue_size=1)

        # Subscribers
        rospy.Subscriber("/features", FeatureArray, self.callback)
        rospy.Subscriber("/localMap", FeatureArray, self.get_map)

        rospy.loginfo("Convolving of representations started")

    def callback(self, msg):
        with t.no_grad():
            if self.map_tensor is not None:
                rospy.loginfo('Repr pair received ...')
                curr_repr = self.msg_to_tensor(msg)
                
                if CUTS:
                    hist, crop_size, crops_idx = self.histogram_cuts(self.map_tensor, curr_repr)
                    combine_cuts(hist, crop_size, crops_idx)
                    # TODO publish result here

                convolved_tensor = self.model.conv_repr(self.map_tensor, curr_repr).squeeze(0).squeeze(0).squeeze(0)
                out = list(convolved_tensor.cpu().numpy())
                out = Float32MultiArray(data=out)
                self.pub.publish(out)

    def get_map(self, msg):
        with t.no_grad():
            self.map_tensor = self.msg_to_tensor(msg)

    def msg_to_tensor(self, msg):
        feature = msg.feature[0]
        target_shape = (1, int(feature.x), int(feature.y), int(feature.size))
        tensor = t.tensor(feature.descriptor, device=device).reshape(target_shape)
        return tensor

    def histogram_cuts(src, tgt): 
        """
        method only used for cuts
        """
        crop_size = src.size(-1) / CUTS_NUM
        crops_idx = np.linspace(0, src.size(-1) - crop_size, CUTS_NUM, dtype=int)
        target_crops = []
        for crop_idx in crops_idx:
            target_crops.append(tgt[..., crop_idx:crop_idx + CROP_SIZE])
        target_crops = t.cat(target_crops, dim=0)
        batched_source = src.repeat(crops_num, 1, 1, 1)
        
        histograms = model(batched_source, target_crops)
        # histogram = histogram * MASK
        # histogram = t.sigmoid(histogram)
        std, mean = t.std_mean(histograms, dim=-1, keepdim=True)
        histograms = (histogram - mean) / std
        histograms = t.softmax(histograms, dim=1)
        return histogram, crop_size, crops_idx

    def combine_cuts(histogram, crop_size, crops_idx):
        # TODO: implement this !!!


if __name__ == '__main__':
    n = Node()
    rospy.spin()

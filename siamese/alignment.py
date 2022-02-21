import rospy
from std_msgs.msg import Float32MultiArray
from stroll_bearnav.msg import FeatureArray
import cv2
import os
import numpy as np
from torchvision import transforms
from model import Siamese, load_model, get_custom_CNN
import torch as t
from bearnav2.msg import Alignment, IntList


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
        self.to_tensor = transforms.ToTensor()

        # Publishers
        self.pub_hist = rospy.Publisher('/heading_histogram', IntList, queue_size=1)
        self.pub_align = rospy.Publisher('/alignment/output', Alignment)

        # Subscribers
        rospy.Subscriber("/bearnav2/alignment/imageB", FeatureArray, self.callback)
        rospy.Subscriber("/bearnav2/alignment/imageA", FeatureArray, self.get_map)

        rospy.loginfo("Convolving of representations started")

    def callback(self, msg):
        with t.no_grad():
            if self.map_tensor is not None:
                rospy.loginfo('Image pair received ...')
                curr_tensor = self.image_to_tensor(msg)
                print("Passing tensors:", self.map_tensor.shape, curr_tensor.shape)
                conv_tensor = self.model(self.map_tensor, curr_tensor)
                print("Outputed histogram", conv_tensor.shape)
                out = list(conv_tensor.cpu().numpy())
                m1 = IntList(data=out)
                m2 = Alignment(alignment=0, uncertainty=0)
                self.pub_align.publish(m2)
                self.pub_hist.publish(m1)


    def get_map(self, msg):
        with t.no_grad():
            self.map_tensor = self.image_to_tensor(msg)

    def image_to_tensor(self, msg):
        im = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        image_tensor = self.to_tensor(im).unsqueeze(0).to(device)
        return image_tensor


if __name__ == '__main__':
    n = Node()
    rospy.spin()

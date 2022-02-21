import rospy
from sensor_msgs.msg import Image
from stroll_bearnav.msg import FeatureArray, Feature
import os
import numpy as np
from torchvision import transforms
from model import Siamese, load_model, get_custom_CNN
import torch as t
import os


device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
# device = t.device("cpu")

class Node:
    def __init__(self):
        rospy.init_node("feature_extraction", anonymous=True)

        # init neural network
        backbone = get_custom_CNN()
        model = Siamese(backbone).to(device)
        file_path = os.path.dirname(os.path.abspath(__file__))
        self.model = load_model(model, os.path.join(file_path, "../models/model_47.pt")).to(device)
        self.model.eval()

        # Transforms
        self.to_tensor = transforms.ToTensor()  # TODO: maybe add resize transform

        # Publishers
        self.pub = rospy.Publisher('/features', FeatureArray, queue_size=1)
        # self.pub_test = rospy.Publisher('/localMap', FeatureArray, queue_size=1)

        # Subscribers
        rospy.Subscriber("/camera/color/image_raw", Image, self.callback)

        rospy.loginfo("Representation processing started!")

    def callback(self, msg):
        with t.no_grad():
            rospy.loginfo('Image received')
            im = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            image_tensor = self.to_tensor(im).unsqueeze(0).to(device)
            repr = self.model.get_repr(image_tensor)
            repr_message = FeatureArray()
            repr_message.feature.append(self.tensor_to_feature(repr))
            self.pub.publish(repr_message)
            # self.pub_test.publish(repr_message)

    def tensor_to_feature(self, repr):
        feature = Feature()
        feature.x = repr.size(-3)
        feature.y = repr.size(-2)
        feature.size = repr.size(-1)
        feature.descriptor = repr.flatten().cpu().numpy()
        return feature


if __name__ == '__main__':
    n = Node()
    rospy.spin()

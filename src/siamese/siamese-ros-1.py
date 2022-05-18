#!/usr/bin/env python
import sys
import rospy
import cv2
import siamese
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from bearnav2.msg import Alignment, IntList, Histogram
from bearnav2.srv import SiameseNet, SiameseNetResponse

pub = None
pub_hist = None
aligner = None
br = CvBridge()
imgABuf = None


def process_imgs(req):
    imgs1 = req.map_images.data
    imgs2 = req.live_images.data
    net_out = siam.forward(imgs1, imgs2)
    hists = [Histogram(hist) for hist in net_out]
    return SiameseNetResponse(hists)


if __name__ == "__main__":
    rospy.init_node("siamese")
    siam = siamese.SiameseNetwork()
    s = rospy.Service('siamese_network', SiameseNet, process_imgs)
    rospy.logwarn("The siamese network is ready!")
    rospy.spin()

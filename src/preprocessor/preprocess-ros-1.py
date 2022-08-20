#!/usr/bin/env python
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from dynamic_reconfigure.server import Server
from bearnav2.cfg import PreprocessorConfig

pub = None
br = CvBridge()
hist_equal = False

def callback(msg):
    img = br.imgmsg_to_cv2(msg)
    if hist_equal:
        img= cv2.equalizeHist(img)
    msg = br.cv2_to_imgmsg(img)
    publisher.publish(msg)

def config_cb(config, level):
    global hist_equal
    hist_equal = config.hist_equal
    return config

if __name__ == "__main__":

    rospy.init_node("preprocessor")
    srv = Server(PreprocessorConfig, config_cb)
    pub = rospy.Publisher("preprocess/output", Image, queue_size=0)
    rospy.Subscriber("preprocess/input", Image, callback)
    rospy.spin()

#!/usr/bin/env python
import rospy
import sys
import cv2
import preprocess
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

pub = None
p = None
br = CvBridge()

def callback(msg):
    img = br.imgmsg_to_cv2(msg)
    img = p.process(img)
    msg = br.cv2_to_imgmsg(img)
    publisher.publish(msg)

if __name__ == "__main__":

    configName = sys.argv[1]
    p = preprocess.Preprocessor(configName)

    rospy.init_node("preprocessor")
    pub = rospy.Publisher("preprocess/output", Image, queue_size=0)
    rospy.Subscriber("preprocess/input", Image, callback)
    rospy.spin()

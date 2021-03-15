#!/usr/bin/env python
import rospy
import cv2
import alignment
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from bearnav2.msg import Alignment

pub = None
a = alignment.Alignment("./config.yaml")
br = CvBridge()
imgABuf = None

def callbackA(msg):
        global imgABuf
        imgABuf = br.imgmsg_to_cv2(msg)

def callbackB(msg):
        global imgABuf
    
        if imgABuf is None:
                print("Still haven't rec'd cam!!")
        return

        imgB = br.imgmsg_to_cv2(msg)
        alignment, uncertainty = a.process(imgABuf, imgB)
        m = Alignment()
        m.alignment = alignment
        m.uncertainty = uncertainty
        pub.publish(m)

if __name__ == "__main__":

        rospy.init_node("alignment")
        pub = rospy.Publisher("alignment/output", Alignment, queue_size=0)
        rospy.Subscriber("alignment/inputA", Image, callbackA)
        rospy.Subscriber("alignment/inputB", Image, callbackB)
        rospy.spin()

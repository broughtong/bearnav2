#!/usr/bin/env python
import sys
import rospy
import cv2
import alignment
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from bearnav2.msg import Alignment, IntList
from dynamic_reconfigure.server import Server
from bearnav2.cfg import AlignmentConfig

pub = None
pub_hist = None
aligner = None
br = CvBridge()
imgBuf = None

def callbackB(msg):
    global imgBuf

    rospy.logwarn("Img rece")

    if imgBuf is None:
        rospy.logwarn("Saving")
        imgBuf = br.imgmsg_to_cv2(msg)

    imgB = br.imgmsg_to_cv2(msg)
    alignment, uncertainty, hist = aligner.process(imgBuf, imgB)
    m = Alignment()
    m.alignment = alignment
    m.uncertainty = uncertainty
    pub.publish(m)

    hm = IntList()
    hm.data = hist
    pub_hist.publish(hm)

if __name__ == "__main__":

    rospy.init_node("alignment")
    aligner = alignment.Alignment()

    pub = rospy.Publisher("alignment/output", Alignment, queue_size=0)
    pub_hist = rospy.Publisher("histogram", IntList, queue_size=0)

    rospy.logwarn("subscibing")
    rospy.Subscriber("/camera_front/image_color", Image, callbackB, queue_size=1)

    rospy.logwarn("Aligner Ready...")
    rospy.spin()

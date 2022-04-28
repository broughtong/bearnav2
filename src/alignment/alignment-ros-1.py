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
imgABuf = None

def callbackA(msg):
    global imgABuf
    imgABuf = br.imgmsg_to_cv2(msg)

def callbackB(msg):

    if imgABuf is None:
        rospy.logwarn("Aligner still awaiting cam A!")
        return

    imgB = br.imgmsg_to_cv2(msg)
    alignment, uncertainty, hist = aligner.process(imgABuf, imgB)
    m = Alignment()
    m.alignment = alignment
    m.uncertainty = uncertainty
    pub.publish(m)

    hm = IntList()
    hm.data = hist
    pub_hist.publish(hm)

def config_cb(config, level):
    global aligner
     aligner.method = config.feature_type
    #aligner.method = "SIAM"
    return config

if __name__ == "__main__":

    rospy.init_node("alignment")
    aligner = alignment.Alignment()
    srv = Server(AlignmentConfig, config_cb)

    pub = rospy.Publisher("alignment/output", Alignment, queue_size=0)
    pub_hist = rospy.Publisher("histogram", IntList, queue_size=0)

    rospy.Subscriber("alignment/inputA", Image, callbackB, queue_size=1)
    rospy.Subscriber("alignment/inputB", Image, callbackA, queue_size=1)

    rospy.logdebug("Aligner Ready...")
    rospy.spin()

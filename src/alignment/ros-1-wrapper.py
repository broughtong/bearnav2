#!/usr/bin/env python3
import rospy
import cv2
import alignment
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

pub = None
a = alignment.Alignment("./config.yaml")
br = CvBridge()
imgABuf = None

def callbackA(msg):
	global imgABuf
	imgABuf = br.imgmsg_to_cv2(msg)

def callbackB(msg):
	global imgABuf
	imgB = br.imgmsg_to_cv2(msg)
	alignment, uncertainty = a.process(imgABuf, imgB)
	publisher.publish(msg)

if __name__ == "__main__":

	rospy.init_node("alignment")
	pub = rospy.Publisher("alignment/output", Image, queue_size=0)
	rospy.Subscriber("alignment/inputA", Image, callbackA)
	rospy.Subscriber("alignment/inputB", Image, callbackB)
	rospy.spin()

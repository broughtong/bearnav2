#!/usr/bin/env python
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from bearnav2.msg import FeatureLocations

#pubs and cv bridge
train_pub = None
current_pub = None
both_pub = None
br = CvBridge()

#img bufs
imgA = None
imgB = None

#timestamp bufs
imgATS = None
imgBTS = None

#oldest imgs we can draw these feats on (will prob be instant tho)
imgTimeout = rospy.Duration.from_sec(0.5)

#draw params
matchCol = [0, 255, 0]
nomatchCol = [255, 0, 0]
boxSize = 10

def cbImgA(msg):
        global imgA
        imgA = br.imgmsg_to_cv2(msg)
        imgATS = rospy.Time.now()

def cbImgB(msg):
        global imgB
        imgB = br.imgmsg_to_cv2(msg)
        imgBTS = rospy.Time.now()

def cbFL(msg):
        if imgA is None or imgB is None:
                print("Waiting for images...")

        #check imgs are current
        time = rospy.Time.now()
        aDiff = time - imgATS
        if ADiff < imgTimeout:
                print("Not drawing feats on img A, too old")
        bDiff = time - imgBTS
        if BDiff < imgTimeout:
                print("Not drawing feats on img B, too old")

        for idx in range(len(msg.xa)):
                if msg.matched[idx]:
                        cv2.rectangle(imgA, (xa-boxSize, ya-boxSize), (xa+boxSize, ya+boxSize), matchCol, 1)
                else:
                        cv2.rectangle(imgA, (xa-boxSize, ya-boxSize), (xa+boxSize, ya+boxSize), nomatchCol, 1)

def callback(msg):

        msg = br.cv2_to_imgmsg(img, encoding="rgb8")
        pub.publish(msg)

if __name__ == "__main__":

        rospy.init_node("feature_viz")
        train_pub = rospy.Publisher("/image_viz/train", Image, queue_size=0)
        current_pub = rospy.Publisher("/image_viz/current", Image, queue_size=0)
        both_pub = rospy.Publisher("/image_viz/alignment", Image, queue_size=0)
        rospy.Subscriber("/alignment/inputA", Image, cbImgA)
        rospy.Subscriber("/alignment/inputB", Image, cbImgB)
        rospy.Subscriber("/alignment/featureLocations", FeatureLocations, cbFL)
        print("Feature viz ready...")
        rospy.spin()

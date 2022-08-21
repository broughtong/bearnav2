#!/usr/bin/env python

import rospy
from bearnav2.srv import Alignment, AlignmentResponse, Representations, RepresentationsResponse
from sensor_processing import BearnavClassic, PF2D, VisualOnly
from backends.odometry.odom_dist import OdometryAbsolute, OdometryRelative
from backends.siamese.siamese import SiameseCNN
from backends.crosscorrelation.crosscorr import CrossCorrelation
from sensor_msgs.msg import Image
from bearnav2.msg import Features, ImageList
import ros_numpy


# Network hyperparameters
PAD = 32
NETWORK_DIVISION = 8
RESIZE_W = 512


def parse_camera_msg(msg):
    img = ros_numpy.numpify(msg)
    if "bgr" in msg.encoding:
        img = img[..., ::-1]  # switch from bgr to rgb
    img_msg = ros_numpy.msgify(Image, img, "rgb8")
    return img_msg


def produce_representationCB(image):
    img = parse_camera_msg(image)
    msg = ImageList([img])
    features = align_abs._to_feature(msg)
    pub.publish(features[0])


if __name__ == '__main__':
    rospy.init_node("sensor_processing")
    rospy.loginfo("Sensor processing started!")
    camera_topic = rospy.get_param("~camera_topic")

    # Choose sensor method
    align_abs = SiameseCNN(padding=PAD, resize_w=RESIZE_W)
    pub = rospy.Publisher("live_representation", Features, queue_size=1)
    sub = rospy.Subscriber(camera_topic, Image,
                           produce_representationCB, queue_size=1, buff_size=50000000)
    rospy.spin()

#!/usr/bin/env python

import rospy
from bearnav2.srv import Alignment, AlignmentResponse, Representations, RepresentationsResponse
from sensor_processing import BearnavClassic, PF2D, VisualOnly
from backends.odometry.odom_dist import OdometryAbsolute, OdometryRelative
from backends.siamese.siamese import SiameseCNN
from backends.crosscorrelation.crosscorr import CrossCorrelation
from sensor_msgs.msg import Image
from bearnav2.msg import Features, ImageList, SensorsInput
import ros_numpy
import torch as t


# Network hyperparameters
PAD = 32
NETWORK_DIVISION = 8
RESIZE_W = 512


current_map = None
map_tensor = None
last_live_rep = None


def parse_camera_msg(msg):
    img = ros_numpy.numpify(msg)
    if "bgr" in msg.encoding:
        img = img[..., ::-1]  # switch from bgr to rgb
    img_msg = ros_numpy.msgify(Image, img, "rgb8")
    return img_msg


def produce_histogramsCB(image):
    global last_live_rep, current_map, map_tensor
    img = parse_camera_msg(image)
    msg = ImageList([img])
    curr_img_tensor = align_abs._to_feature(msg, pytorch=True)
    if current_map is not None and map_tensor is not None and last_live_rep is not None:
        extended_map_tensor = t.stack([map_tensor, last_live_rep])
        histograms = align_abs.forward(extended_map_tensor, curr_img_tensor, pytorch=True)
        current_map.live_features.values = histograms.flatten()
        current_map.live_features.shape = histograms.shape
        current_map.header = image.header
        pub.publish(current_map)
    last_live_rep = curr_img_tensor


def fetch_mapCB(map_msg):
    global current_map, map_tensor
    current_map = map_msg
    map_tensor = align_abs._from_feature(current_map.map_features)


if __name__ == '__main__':
    rospy.init_node("sensor_processing")
    rospy.loginfo("Sensor processing started!")
    camera_topic = rospy.get_param("~camera_topic")

    # Choose sensor method
    align_abs = SiameseCNN(padding=PAD, resize_w=RESIZE_W)
    pub = rospy.Publisher("sensors_output", SensorsInput, queue_size=1)
    sub_camera = rospy.Subscriber(camera_topic, Image,
                                  produce_histogramsCB, queue_size=1, buff_size=10000000)
    sub_map = rospy.Subscriber("sensors_input", SensorsInput,
                               fetch_mapCB, queue_size=1, buff_size=10000000)


    rospy.spin()

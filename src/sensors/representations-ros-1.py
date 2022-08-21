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
from message_filters import ApproximateTimeSynchronizer, Subscriber


# Network hyperparameters
PAD = 32
NETWORK_DIVISION = 8
RESIZE_W = 512


last_live_rep = None


def parse_camera_msg(msg):
    img = ros_numpy.numpify(msg)
    if "bgr" in msg.encoding:
        img = img[..., ::-1]  # switch from bgr to rgb
    img_msg = ros_numpy.msgify(Image, img, "rgb8")
    return img_msg


def produce_histogramsCB(image, map_msg):
    global last_live_rep
    current_map = map_msg
    map_tensor = align_abs._from_feature(current_map.map_features)
    img = parse_camera_msg(image)
    msg = ImageList([img])
    curr_img_tensor = align_abs._to_feature(msg, pytorch=True)
    if current_map is not None and map_tensor is not None and last_live_rep is not None:
        extended_map_tensor = t.cat([map_tensor, last_live_rep])
        histograms = align_abs.forward(extended_map_tensor, curr_img_tensor, pytorch=True)
        ret_feature = Features()
        ret_feature.shape = histograms.shape
        ret_feature.values = histograms.flatten()
        current_map.live_features = [ret_feature]
        current_map.header = image.header
        pub.publish(current_map)
    last_live_rep = curr_img_tensor


if __name__ == '__main__':
    rospy.init_node("sensor_processing")
    rospy.loginfo("Sensor processing started!")
    camera_topic = rospy.get_param("~camera_topic")

    # Choose sensor method
    align_abs = SiameseCNN(padding=PAD, resize_w=RESIZE_W)
    # pub = rospy.Publisher("sensors_output", SensorsInput, queue_size=1)
    # sub_camera = rospy.Subscriber(camera_topic, Image,
    #                               produce_histogramsCB, queue_size=1, buff_size=10000000)
    # sub_map = rospy.Subscriber("sensors_input", SensorsInput,
    #                            fetch_mapCB, queue_size=1, buff_size=10000000)


    cam_sub = Subscriber(camera_topic, Image)
    map_sub = Subscriber("sensors_input", SensorsInput)
    synced_topics = ApproximateTimeSynchronizer([cam_sub, map_sub], queue_size=1, slop=0.25)
    synced_topics.registerCallback(produce_histogramsCB)

    rospy.spin()

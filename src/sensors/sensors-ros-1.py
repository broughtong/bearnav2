#!/usr/bin/env python

import rospy
from bearnav2.msg import SensorsOutput, SensorsInput, ImageList
from std_msgs.msg import Float32
from bearnav2.srv import Alignment, AlignmentResponse
from nav_msgs.msg import Odometry
from sensor_processing import BearnavClassic, PF2D
from backends.odometry.odom_dist import OdometryAbsolute, OdometryRelative
from backends.siamese.siamese import SiameseCNN
from backends.crosscorrelation.crosscorr import CrossCorrelation


def start_subscribes(fusion_class,
                     abs_align_topic, abs_dist_topic, rel_dist_topic, prob_dist_topic,
                     rel_align_service_name):
    # subscribers for images and other topics used for alignment and distance estimation
    if fusion_class.abs_align_est is not None and len(abs_align_topic) > 0:
        rospy.Subscriber(abs_align_topic, fusion_class.abs_align_est.supported_message_type,
                         fusion_class.process_abs_alignment, queue_size=1)
    if fusion_class.abs_dist_est is not None and len(abs_dist_topic) > 0:
        rospy.Subscriber(abs_dist_topic, fusion_class.abs_dist_est.supported_message_type,
                         fusion_class.process_abs_distance, queue_size=1)
    if fusion_class.rel_dist_est is not None and len(rel_dist_topic) > 0:
        rospy.Subscriber(rel_dist_topic, fusion_class.rel_dist_est.supported_message_type,
                         fusion_class.process_rel_distance, queue_size=1)
    if fusion_class.prob_dist_est is not None and len(prob_dist_topic) > 0:
        rospy.Subscriber(prob_dist_topic, fusion_class.prob_dist_est.supported_message_type,
                         fusion_class.process_prob_distance, queue_size=1)
    # service for rel alignment
    if fusion_class.rel_align_est is not None and len(rel_align_service_name) > 0:
        relative_image_service = rospy.Service(fusion_class.type_prefix + "/" + rel_align_service_name,
                                               Alignment, fusion_class.process_rel_alignment)
        return relative_image_service
    return None


if __name__ == '__main__':
    rospy.init_node("sensor_processing")
    rospy.loginfo("Sensor processing started!")

    # Choose sensor method
    align_abs = SiameseCNN()
    align_rel = CrossCorrelation()
    dist_abs = OdometryAbsolute()
    dist_rel = OdometryRelative()

    # Set here fusion method
    teach_fusion = BearnavClassic("teach", align_abs, dist_abs)
    repeat_fusion = PF2D("repeat", 500, 0.05, 0.5, 0.025, 0.3, 2, True,
                         align_abs, align_rel, dist_rel)

    # Start listening to topics and service for teacher (mapmaker)
    teach_handler = start_subscribes(teach_fusion,
                                     "", "/husky_velocity_controller/odom", "", "",
                                     "")
    # Start listening to topics and service for repeater
    repeat_handler = start_subscribes(repeat_fusion,
                                      "sensors_input", "", "/husky_velocity_controller/odom", "",
                                      "local_alignment")

    rospy.spin()

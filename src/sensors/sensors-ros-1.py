#!/usr/bin/env python

import rospy
from bearnav2.srv import Alignment, AlignmentResponse, Representations, RepresentationsResponse
from sensor_processing import BearnavClassic, PF2D, VisualOnly
from backends.odometry.odom_dist import OdometryAbsolute, OdometryRelative
from backends.siamese.siamese import SiameseCNN
from backends.crosscorrelation.crosscorr import CrossCorrelation


def start_subscribes(fusion_class,
                     abs_align_topic, abs_dist_topic, rel_dist_topic, prob_dist_topic,
                     rel_align_service_name, repr_service_name):
    # --------------- TOPICS ------------
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
    # -------------- SERVICES -------------
    # service for rel alignment
    relative_image_service = None
    if fusion_class.rel_align_est is not None and len(rel_align_service_name) > 0:
        relative_image_service = rospy.Service(rel_align_service_name,
                                               Alignment, fusion_class.process_rel_alignment)
    # service for representations
    representation_service = None
    if fusion_class.repr_creator is not None and len(repr_service_name) > 0:
        representation_service = rospy.Service(repr_service_name,
                                               Representations, fusion_class.create_representations)

    return relative_image_service, representation_service


if __name__ == '__main__':
    rospy.init_node("sensor_processing")
    rospy.loginfo("Sensor processing started!")

    # Choose sensor method
    align_abs = SiameseCNN()
    align_rel = CrossCorrelation()
    dist_abs = OdometryAbsolute()
    dist_rel = OdometryRelative()

    # Set here fusion method for teaching phase -------------------------------------------
    teach_fusion = BearnavClassic("teach", align_abs, dist_abs, align_abs)
    teach_handlers = start_subscribes(teach_fusion,
                                      "", "/husky_velocity_controller/odom", "", "",
                                      "", "")

    # TODO: set representations here

    # Set here fusion method for repeating phase ------------------------------------------
    # 1) Bearnav classic - this method also needs publish span 0 in the repeater !!!
    # repeat_fusion = BearnavClassic("repeat", align_abs, dist_abs)
    # repeat_handlers = start_subscribes(repeat_fusion,
    #                                    "sensors_input", "/husky_velocity_controller/odom", "", "",
    #                                    "", "")
    # 2) Particle filter 2D - parameters are really important
    repeat_fusion = PF2D("repeat", 500, 0.25, 1.0, 0.03, 0.3, 2, True,
                         align_abs, align_rel, dist_rel, align_abs)
    repeat_handlers = start_subscribes(repeat_fusion,
                                       "sensors_input", "", "/husky_velocity_controller/odom", "",
                                       "local_alignment", "get_repr")
    # 3) Visual Only
    # repeat_fusion = VisualOnly("repeat", align_abs, align_abs, align_abs)
    # repeat_handler = start_subscribes(repeat_fusion,
    #                                   "sensors_input", "", "", "sensors_input",
    #                                   "", "")

    rospy.spin()

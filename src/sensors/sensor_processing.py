import numpy as np
from base_classes import DisplacementEstimator, RelativeDistanceEstimator, AbsoluteDistanceEstimator, SensorFusion
import rospy

"""
Here should be placed all classes for fusion of sensor processing
"""


class BearnavClassic(SensorFusion):

    def __init__(self, abs_align_est: DisplacementEstimator, abs_dist_est: AbsoluteDistanceEstimator):
        super().__init__()
        self.abs_align_est = abs_align_est
        self.abs_dist_est = abs_dist_est

    def process_rel_alignment(self, msg):
        rospy.logwarn("This function is not available for this fusion class")
        raise Exception("Bearnav Classic does not support relative alignment")

    def process_abs_alignment(self, msg):
        msg.map_images = [msg.map_images[len(msg.map_images) // 2]]     # choose only the middle image
        histogram = self.abs_align_est.displacement_message_callback(msg)
        self.alignment = np.argmax(histogram) - np.size(histogram)//2
        self.publish_data()

    def process_rel_distance(self, msg):
        rospy.logwarn("This function is not available for this fusion class")
        raise Exception("Bearnav Classic does not support relative distance")

    def process_abs_distance(self, msg):
        self.distance = self.abs_dist_est.abs_dist_message_callback(msg)
        self.publish_data()

    def process_prob_distance(self, msg):
        rospy.logwarn("This function is not available for this fusion class")
        raise Exception("Bearnav Classic does not support probability of distances")
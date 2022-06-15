import rospy
from bearnav2.msg import SensorsOutput, SensorsInput, ImageList
from std_msgs.msg import Float32
from bearnav2.srv import LocalAlignment, LocalAlignmentResponse
from nav_msgs.msg import Odometry
from src.sensors.sensor_processing import BearnavClassic
from src.sensors.backends.odometry.odom_dist import OdometryAbsolute, OdometryRelative


def relative_alignment_service(msg):
    """
    Comparing images without any changes - very close locally and timewise (in same map for example)
    """
    fusion.process_rel_alignment(msg)


def absolute_alignment_callback(msg):
    """
    Comparing current camera view vs map
    """
    fusion.process_abs_alignment(msg)


def absolute_distance_callback(msg):
    """
    this is message with distance float
    """
    fusion.process_abs_distance(msg)


def relative_distance_callback(msg):
    """
    this is message with distance float
    """
    fusion.process_rel_distance(msg)


if __name__ == '__main__':
    rospy.init_node("sensor_processing")
    rospy.loginfo("Sensor processing started!")

    # TODO: choose sensor method
    dist_abs = OdometryAbsolute()
    align_abs = None    # TODO implement this

    # TODO set here fusion method
    fusion = BearnavClassic(align_abs, dist_abs)

    # service for relative alignment
    if fusion.rel_align_est is not None:
        # TODO: do this also with varying message (service) type
        relative_image_service = rospy.Service('relative_alignment', LocalAlignment, relative_alignment_service)
    # subscribers for images and other topics used for alignment and distance estimation
    if fusion.abs_align_est is not None:
        rospy.Subscriber("alignment_absolute", fusion.abs_align_est.supported_message_type,
                         absolute_alignment_callback, queue_size=1)
    if fusion.abs_dist_est is not None:
        rospy.Subscriber("distance_absolute", fusion.abs_dist_est.supported_message_type,
                         absolute_distance_callback, queue_size=1)
    if fusion.rel_dist_est is not None:
        rospy.Subscriber("distance_relative", fusion.rel_dist_est.supported_message_type,
                         relative_distance_callback, queue_size=1)

    rospy.spin()

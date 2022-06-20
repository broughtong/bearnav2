import rospy
from bearnav2.msg import SensorsOutput, SensorsInput, ImageList
from std_msgs.msg import Float32
from bearnav2.srv import LocalAlignment, LocalAlignmentResponse
from nav_msgs.msg import Odometry
from src.sensors.sensor_processing import BearnavClassic
from src.sensors.backends.odometry.odom_dist import OdometryAbsolute, OdometryRelative
from src.sensors.backends.siamese.siamese import SiameseCNN


def start_subscribes(fusion_class):
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


def start_services(fusion_class):
    if fusion_class.rel_align_est is not None and len(rel_align_service) > 0:
        # TODO: do this also with varying message (service) type - LocalAlignment is not universal
        relative_image_service = rospy.Service(rel_align_service, LocalAlignment, fusion_class.process_rel_alignment)
        return relative_image_service
    return None


if __name__ == '__main__':
    rospy.init_node("sensor_processing")
    rospy.loginfo("Sensor processing started!")

    # Choose sensor method
    align_abs = SiameseCNN()
    dist_abs = OdometryAbsolute()

    # Set here fusion method
    fusion = BearnavClassic(align_abs, dist_abs)

    # TODO: set the topics to subscribe here
    rel_align_service = "blah_blah"
    abs_align_topic = "blah_blah"
    rel_dist_topic = "blah_blah"
    abs_dist_topic = "blah_blah"
    prob_dist_topic = "blah_blah"

    # Start listening to topics, and initialize services
    start_subscribes(fusion)
    service_handle = start_services(fusion)

    rospy.spin()

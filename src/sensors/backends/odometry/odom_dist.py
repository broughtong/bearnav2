from base_classes import RelativeDistanceEstimator, AbsoluteDistanceEstimator
from nav_msgs.msg import Odometry
import rospy


class OdometryAbsolute(AbsoluteDistanceEstimator):

    def __init__(self):
        super(OdometryAbsolute, self).__init__()
        self.supported_message_type = Odometry
        self.last_odom = None
        rospy.loginfo("Odometry absolute distance estimator successfully initialized!")

    def _abs_dist_message_callback(self, msg: Odometry) -> float:
        if self.last_odom is None:
            self.last_odom = msg
            return self._distance
        dx = self.last_odom.pose.pose.position.x - msg.pose.pose.position.x
        dy = self.last_odom.pose.pose.position.y - msg.pose.pose.position.y
        dz = self.last_odom.pose.pose.position.z - msg.pose.pose.position.z
        self._distance += (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
        self.last_odom = msg
        self.header = msg.header
        return self._distance

    def _set_dist(self, dist) -> float:
        self.last_odom = None
        self._distance = 0.0
        return dist

    def health_check(self) -> bool:
        return True


class OdometryRelative(RelativeDistanceEstimator):

    def __init__(self):
        super(OdometryRelative, self).__init__()
        self.supported_message_type = Odometry
        self.last_odom = None
        rospy.loginfo("Odometry relative distance estimator successfully initialized!")

    def _rel_dist_message_callback(self, msg: Odometry) -> float:
        if self.last_odom is None:
            self.last_odom = msg
            return None
        dx = self.last_odom.pose.pose.position.x - msg.pose.pose.position.x
        dy = self.last_odom.pose.pose.position.y - msg.pose.pose.position.y
        dz = self.last_odom.pose.pose.position.z - msg.pose.pose.position.z
        self.last_odom = msg
        return (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5

    def health_check(self) -> bool:
        return True

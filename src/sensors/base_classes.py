from abc import ABC, abstractmethod
import numpy as np
from std_msgs.msg import Float32
from sensor_msgs.msg import Image, SensorsOutput
import rospy
from bearnav2.srv import SetDist, SetDistResponse


"""
These are base classes for sensor modules in this package
"""


class DisplacementEstimator(ABC):
    """
    Base class for displacement estimator
    Extend this to add new estimator, the main method which must be implemented is "_displacement_message_callback"
    """

    def __init__(self):
        self.supported_message_type = None  # this attrubute must be set
        if not self.health_check():
            rospy.logwarn("Displacement estimator health check was not successful")
            raise Exception("Displacement Estimator health check failed")

    def displacement_message_callback(self, msg: object) -> list[np.ndarray]:
        if not isinstance(msg, self.supported_message_type) or self.supported_message_type is None:
            rospy.logwarn("Incorrect type of message in displacement estimator" +
                          str(type(msg)) + " vs " + str(self.supported_message_type))
            raise Exception("Wrong message type")
        return self._displacement_message_callback(msg)

    @abstractmethod
    def _displacement_message_callback(self, msg: object) -> list[np.ndarray]:
        """
        returns list of histograms (displacement probabilities) -> there could be one histogram or multiple
        """
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        raise NotImplementedError


class RelativeDistanceEstimator(ABC):
    """
    Abtract method for estimating the relative distance traveled from last measurement
    Extend this to add new estimator, the main method which must be implemented is "_rel_dist_message_callback"
    """

    def __init__(self):
        self.supported_message_type = None
        if not self.health_check():
            rospy.logwarn("Relative distance estimator health check was not successful")
            raise Exception("Rel Dist Estimator health check failed")

    def rel_dist_message_callback(self, msg: object) -> float:
        if not isinstance(msg, self.supported_message_type) or self.supported_message_type is None:
            rospy.logwarn("Incorrect type of message in relative distance estimator" +
                          str(type(msg)) + " vs " + str(self.supported_message_type))
            raise Exception("Wrong message type")
        return self._rel_dist_message_callback(msg)

    @abstractmethod
    def _rel_dist_message_callback(self, msg: object) -> float:
        """
        returns float value which tells by how much the robot moved
        """
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        raise NotImplementedError


class AbsoluteDistanceEstimator(ABC):
    """
    Abstract method for estimating the absolute distance traveled.
    Extend this to add new estimator, the main method which must be implemented is "_abs_dist_message_callback"
    """

    def __init__(self):
        self.supported_message_type = None
        self._distance = None
        self.set_distance = rospy.Service('set_distance', SetDist, self.set_distance)
        if not self.health_check():
            rospy.logwarn("Absolute distance estimator health check was not successful")
            raise Exception("Abs Dist Estimator health check failed")

    def abs_dist_message_callback(self, msg: object) -> float:
        if self._distance is None:
            rospy.logwarn("If you want to use absolute distance sensor - you have to set the distance first!")
            raise Exception("The distance must be set first")
        if not isinstance(msg, self.supported_message_type) or self.supported_message_type is None:
            rospy.logwarn("Incorrect type of message in absolute distance estimator" +
                          str(type(msg)) + " vs " + str(self.supported_message_type))
            raise Exception("Wrong message type")
        return self._abs_dist_message_callback(msg)

    def set_distance(self, msg: SetDist) -> SetDistResponse:
        self._distance = msg.dist
        return SetDistResponse()

    @abstractmethod
    def _abs_dist_message_callback(self, msg: object) -> float:
        """
        returns floats -> distance absolute value
        """
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        raise NotImplementedError


class ProbabilityDistanceEstimator(ABC):
    """
    Abstract method for estimating the absolute distance traveled.
    Extend this to add new estimator, the main method which must be implemented is "_abs_dist_message_callback"
    """

    def __init__(self):
        self.supported_message_type = None
        if not self.health_check():
            rospy.logwarn("Absolute distance estimator health check was not successful")
            raise Exception("Abs Dist Estimator health check failed")

    def prob_dist_message_callback(self, msg: object) -> list[float]:
        if not isinstance(msg, self.supported_message_type) or self.supported_message_type is None:
            rospy.logwarn("Incorrect type of message in probabilistic distance estimator" +
                          str(type(msg)) + " vs " + str(self.supported_message_type))
            raise Exception("Wrong message type")
        return self._prob_dist_message_callback(msg)

    @abstractmethod
    def _prob_dist_message_callback(self, msg: object) -> list[float]:
        """
        returns list of floats -> probability of traveled distance
        """
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        raise NotImplementedError


class SensorFusion(ABC):
    """
    Abstract method for the sensor fusion!
    """

    def __init__(self):
        self.output_pub = rospy.Publisher("output", SensorsOutput, queue_size=1)
        self.distance = 0
        self.alignment = 0
        self.distance_std = 0
        self.alignment_std = 0
        self.abs_dist_est = None
        self.rel_dist_est = None
        self.prob_dist_est = None
        self.abs_align_est = None
        self.rel_align_est = None

    def publish_data(self):
        out = SensorsOutput()
        out.alignment = self.alignment
        out.alignment_uncertainty = self.alignment_std
        out.distance = self.distance
        out.distance_uncertainty = self.distance_std
        self.output_pub.publish(out)

    @abstractmethod
    def process_rel_alignment(self, msg):
        raise NotImplementedError

    @abstractmethod
    def process_abs_alignment(self, msg):
        raise NotImplementedError

    @abstractmethod
    def process_rel_distance(self, msg):
        raise NotImplementedError

    @abstractmethod
    def process_abs_distance(self, msg):
        raise NotImplementedError

    @abstractmethod
    def process_prob_distance(self, msg):
        raise NotImplementedError


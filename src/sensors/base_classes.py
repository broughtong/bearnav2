from abc import ABC, abstractmethod
import numpy as np
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from bearnav2.msg import SensorsOutput
import rospy
from bearnav2.srv import SetDist, SetDistResponse
from typing import List


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

    def displacement_message_callback(self, msg: object) -> List[np.ndarray]:
        if not isinstance(msg, self.supported_message_type) or self.supported_message_type is None:
            rospy.logwarn("Incorrect type of message in displacement estimator" +
                          str(type(msg)) + " vs " + str(self.supported_message_type))
            raise Exception("Wrong message type")
        return self._displacement_message_callback(msg)

    @abstractmethod
    def _displacement_message_callback(self, msg: object) -> List[np.ndarray]:
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

    def set_dist(self, dist):
        self._distance = dist

    @abstractmethod
    def _abs_dist_message_callback(self, msg: object) -> float:
        """
        increment the absolute distance self._distance in here
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
        if self.health_check():
            rospy.logwarn("Absolute distance estimator health check was not successful")
            raise Exception("Abs Dist Estimator health check failed")

    def prob_dist_message_callback(self, msg: object) -> List[float]:
        if self._distance is None:
            rospy.logwarn("If you want to use absolute distance sensor - you have to set the distance first!")
            raise Exception("The distance must be set first")
        if not isinstance(msg, self.supported_message_type) or self.supported_message_type is None:
            rospy.logwarn("Incorrect type of message in probabilistic distance estimator" +
                          str(type(msg)) + " vs " + str(self.supported_message_type))
            raise Exception("Wrong message type")
        return self._prob_dist_message_callback(msg)

    def set_dist(self, dist):
        self._distance = dist

    @abstractmethod
    def _prob_dist_message_callback(self, msg: object) -> List[float]:
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
        self.output_pub = rospy.Publisher("correction_cmd", SensorsOutput, queue_size=1)
        self.distance = None
        self.alignment = None
        self.distance_std = None
        self.alignment_std = None
        self.abs_dist_est = None
        self.rel_dist_est = None
        self.prob_dist_est = None
        self.abs_align_est = None
        self.rel_align_est = None
        self.set_distance = rospy.Service('set_dist', SetDist, self.set_distance)
        self.set_alignment = rospy.Service('set_align', SetDist, self.set_alignment)
    
    def publish_data(self):
        out = SensorsOutput()
        if self.alignment is not None:
            out.alignment = self.alignment
            out.alignment_uncertainty = self.alignment_std
        else:
            out.alignment = 0.0
            out.alignment_uncertainty = -1.0
        if self.distance is not None:
            out.distance = self.distance
            out.distance_uncertainty = self.distance_std
        else:
            out.distance = 0.0
            out.distance_uncertainty = -1.0
        # rospy.logwarn(str(out))
        self.output_pub.publish(out)

    def set_distance(self, msg: SetDist) -> SetDistResponse:
        self.distance = msg.dist
        self.distance_std = 0.0
        if self.abs_dist_est is not None:
            self.abs_dist_est.set_dist(self.distance)
        if self.prob_dist_est is not None:
            self.prob_dist_est.set_dist(self.distance)
        return SetDistResponse()

    def set_alignment(self, msg: SetDist) -> SetDistResponse:
        self.alignment = msg.dist
        self.alignment_std = 0.0
        return SetDistResponse()
        
    def process_rel_alignment(self, msg):
        if self.alignment is not None:
            self._process_rel_alignment(msg)

    def process_abs_alignment(self, msg):
        if self.alignment is not None:
            self._process_abs_alignment(msg)

    def process_rel_distance(self, msg):
        if self.distance is not None:
            self._process_rel_distance(msg)

    def process_abs_distance(self, msg):
        if self.distance is not None:
            self._process_abs_distance(msg)

    def process_prob_distance(self, msg):
        if self.distance is not None:
            self._process_prob_distance(msg)

    @abstractmethod
    def _process_rel_alignment(self, msg):
        raise NotImplementedError

    @abstractmethod
    def _process_abs_alignment(self, msg):
        raise NotImplementedError

    @abstractmethod
    def _process_rel_distance(self, msg):
        raise NotImplementedError

    @abstractmethod
    def _process_abs_distance(self, msg):
        raise NotImplementedError

    @abstractmethod
    def _process_prob_distance(self, msg):
        raise NotImplementedError


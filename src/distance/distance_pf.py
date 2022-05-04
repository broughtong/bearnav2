#!/usr/bin/env python
import rospy
import numpy as np
from scipy import interpolate
from bearnav2.srv import SiameseNet


def numpy_softmax(arr):
    return np.exp(arr) / np.sum(np.exp(arr))


class DistancePF:
    def __init__(self, use_twist):
        self.particles_num = 100
        self.odom_var = 0.05
        self.interp_coef = 10
        self.particles_frac = 2

        self.motion_step = False
        self.sensor_step = False

        self.last_odom = None
        self.last_time = None
        self.particles = None

        rospy.wait_for_service('siamese_network')
        self.nn_service = rospy.ServiceProxy('siamese_network', SiameseNet)

    def set(self, dst):
        self.particles = np.ones(self.particles_num) * dst
        self.motion_step = True
        self.last_odom = None
        self.last_time = None
        return dst

    def get_position(self):
        assert self.particles is not None
        return np.mean(self.particles)

    def process_images(self, imgsA, imgsB):
        try:
            resp1 = self.nn_service(imgsA, imgsB)
            return resp1
        except rospy.ServiceException as e:
            rospy.logwarn("Service call failed: %s" % e)
            return None

    def processT(self, msg):
        return None, False

    def processO(self, msg):
        if self.motion_step:
            dx = self.last_odom.pose.pose.position.x - msg.pose.pose.position.x
            dy = self.last_odom.pose.pose.position.y - msg.pose.pose.position.y
            dz = self.last_odom.pose.pose.position.z - msg.pose.pose.position.z
            self.last_odom = msg
            # measured distance
            dist_diff = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
            # adding new particles with uncertainty
            self.particles = np.stack([self.particles + dist_diff +
                                       np.random.normal(loc=0, scale=self.odom_var * dist_diff, size=self.particles_num)
                                       for _ in range(self.particles_frac)])
            self.motion_step = False
            self.sensor_step = True
        return self.get_position(), True

    def processS(self, msg):
        if self.sensor_step:
            imgsA = msg.imgsA
            imgsB = msg.imgsB
            dists = msg.distances

            # clip particles to available image span
            self.particles = np.clip(self.particles, min(dists), max(dists))
            # get time histogram
            time_hist = self.process_images(imgsA, imgsB)
            # normalize
            prob_time_hist = numpy_softmax((time_hist - np.mean(time_hist)))
            # interpolate
            prob_interp = interpolate.interp1d(dists, prob_time_hist, kind="cubic")
            # get probabilites of particles
            particle_prob = prob_interp(self.particles)
            # choose best candidates and reduce the number of particles
            self.particles = np.random.choice(self.particles, int(self.particles_num/self.particles_frac),
                                              p=particle_prob)
            self.sensor_step = False
            self.motion_step = True

        return self.get_position(), True

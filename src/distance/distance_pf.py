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
        self.odom_var = 0.1
        self.interp_coef = 10
        self.particles_frac = 2

        self.motion_step = False
        self.sensor_step = False

        self.last_odom = None
        self.last_time = None
        self.particles = None

        # for debug
        self.raw_odom = None

        rospy.wait_for_service('/siamese_network')
        self.nn_service = rospy.ServiceProxy('/siamese_network', SiameseNet)

    def set(self, dst, var=1):
        self.particles = np.ones(self.particles_num) * dst.dist +\
                         np.random.normal(loc=0, scale=var, size=self.particles_num)
        print(str(self.particles.size), "particles initialized at position", str(dst))
        self.raw_odom = dst.dist
        self.motion_step = True
        self.sensor_step = True
        self.last_odom = None
        self.last_time = None
        return dst.dist

    def get_position(self):
        assert self.particles is not None
        rospy.logwarn("Outputted position: " + str(np.mean(self.particles)) + " +- " + str(np.std(self.particles)) + " vs raw odom: " + str(self.raw_odom))
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
        if self.last_odom is None:
            self.last_odom = msg
            return None, False
        if self.motion_step:
            dx = self.last_odom.pose.pose.position.x - msg.pose.pose.position.x
            dy = self.last_odom.pose.pose.position.y - msg.pose.pose.position.y
            dz = self.last_odom.pose.pose.position.z - msg.pose.pose.position.z
            self.last_odom = msg
            # measured distance
            dist_diff = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
            self.raw_odom += dist_diff
            # adding new particles with uncertainty
            self.particles = np.concatenate([self.particles + dist_diff +
                                             np.random.normal(loc=0, scale=self.odom_var * dist_diff, size=self.particles.size)
                                             for _ in range(self.particles_frac)])
            self.motion_step = False
            self.sensor_step = True
            rospy.logwarn("waiting for sensor model ...")

        return self.get_position(), True

    def processS(self, msg):
        if self.sensor_step:
            imgsA = msg.map_images
            imgsB = msg.live_images
            dists = msg.distances

            # clip particles to available image span
            self.particles = np.clip(self.particles, min(dists), max(dists))
            # get time histogram
            hists = self.process_images(imgsA, imgsB)
            hists = np.array([hist.data for hist in hists.histograms])
            time_hist = np.max(hists, axis=-1)
            # print("Time histogram", time_hist)
            # interpolate
            rospy.logwarn("time histogram: " + str(time_hist))
            rospy.logwarn("with distances: " + str(dists))
            prob_interp = interpolate.interp1d(dists, time_hist, kind="linear")
            # get probabilites of particles
            particle_prob = numpy_softmax(prob_interp(self.particles))
            # choose best candidates and reduce the number of particles
            # rospy.logwarn(particle_prob)
            self.particles = np.random.choice(self.particles, int(self.particles_num/self.particles_frac),
                                              p=particle_prob)
            self.sensor_step = False
            self.motion_step = True
            rospy.logwarn("waiting for robot model ...")

        return self.get_position(), True

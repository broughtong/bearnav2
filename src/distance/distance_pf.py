#!/usr/bin/env python
import rospy
import numpy as np
from scipy import interpolate
from bearnav2.srv import SiameseNet


def numpy_softmax(arr):
    return np.exp(arr) / np.sum(np.exp(arr), axis=-1, keepdims=True)


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

        self.odom_only = False
        self.visual_only = False

        # for debug
        self.raw_odom = None

        rospy.wait_for_service('siamese_network')
        self.nn_service = rospy.ServiceProxy('/siamese_network', SiameseNet, persistent=True)

    def set(self, dst, var=0.5):
        self.particles = np.ones(self.particles_num) * dst.dist +\
                         np.random.normal(loc=0, scale=var, size=self.particles_num)
        #print(str(self.particles.size), "particles initialized at position", str(dst))
        self.raw_odom = dst.dist
        self.motion_step = True
        self.sensor_step = True
        self.last_odom = None
        self.last_time = None
        return dst.dist

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
        if self.last_odom is None or self.visual_only:
            self.last_odom = msg
            return None, False
        if self.odom_only:
            dx = self.last_odom.pose.pose.position.x - msg.pose.pose.position.x
            dy = self.last_odom.pose.pose.position.y - msg.pose.pose.position.y
            dz = self.last_odom.pose.pose.position.z - msg.pose.pose.position.z
            self.last_odom = msg
            # measured distance
            dist_diff = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
            self.raw_odom += dist_diff
            return self.raw_odom, True
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
            rospy.logwarn("Outputted position: " + str(np.mean(self.particles)) + " +- " + str(np.std(self.particles)) + " vs raw odom: " + str(self.raw_odom))

        return self.get_position(), True

    def processS(self, msg):
        if self.odom_only:
            return None, False

        if self.visual_only:
            imgsA = msg.map_images
            imgsB = msg.live_images
            dists = msg.distances
            hists = self.process_images(imgsA, imgsB)
            hists = np.array([hist.data for hist in hists.histograms])
            time_hist = np.max(hists, axis=-1)
            prob_interp = interpolate.interp1d(dists, time_hist, kind="quadratic")
            interp_list = np.linspace(dists[0], dists[-1], dists*8)  # 8 is magic number
            interp_out = prob_interp(interp_list)
            return interp_list[np.argmax(interp_out)], True

        if self.sensor_step:
            imgsA = msg.map_images
            imgsB = msg.live_images
            dists = msg.distances

            rospy.logwarn("curr img times:" + str(imgsA.data[0].header.stamp.secs) + ", " + str(imgsB.data[0].header.stamp.secs))
            # clip particles to available image span
            self.particles = np.clip(self.particles, min(dists), max(dists))
            # get time histogram
            hists = self.process_images(imgsA, imgsB)
            hists = np.array([hist.data for hist in hists.histograms])
            # hists = numpy_softmax(hists)
            time_hist = np.max(hists, axis=-1)
            # print("Time histogram", time_hist)
            # interpolate
            #rospy.logwarn("time histogram: " + str(time_hist))
            #rospy.logwarn("with distances: " + str(dists))
            prob_interp = interpolate.interp1d(dists, time_hist, kind="quadratic")
            # get probabilites of particles
            particle_prob = numpy_softmax(prob_interp(self.particles))
            # rospy.logwarn(particle_prob)
            # rospy.logwarn(self.particles)
            # choose best candidates and reduce the number of particles
            # rospy.logwarn(particle_prob)
            self.particles = np.random.choice(self.particles, int(self.particles_num/self.particles_frac),
                                              p=particle_prob)
            self.sensor_step = False
            self.motion_step = True
            rospy.logwarn("waiting for robot model ...")

        return self.get_position(), True

#!/usr/bin/env python
import rospy
import numpy as np
from scipy import interpolate
from bearnav2.srv import SiameseNet
from bearnav2.msg import FloatList


def numpy_softmax(arr):
    tmp = np.exp(arr) / np.sum(np.sum(np.exp(arr)))
    return tmp


class DistancePF:
    def __init__(self):
        self.particles_num = 200
        self.odom_var = 0.1
        self.displac_var = 4  # in pixels
        self.interp_coef = 10
        self.particles_frac = 2

        self.motion_step = False
        self.sensor_step = False

        self.last_image = None
        self.last_odom = None
        self.particles = None

        # This must be set to odom only during recording otherwise no images are going to be saved
        self.odom_only = False
        self.visual_only = False

        # for debug only
        self.raw_odom = None

        rospy.wait_for_service('siamese_network')
        self.nn_service = rospy.ServiceProxy('siamese_network', SiameseNet, persistent=True)
        self.particles_pub = rospy.Publisher("particles", FloatList, queue_size=1)

    def set(self, dst, var=(0.5, 100)):
        self.particles = np.transpose(np.ones((2, self.particles_num)).transpose() * np.array((dst.dist, 0)) +\
                                      np.random.normal(loc=(0, 0), scale=var, size=(self.particles_num, 2)))
        #print(str(self.particles.size), "particles initialized at position", str(dst))
        self.raw_odom = dst.dist
        self.motion_step = True
        self.sensor_step = True
        self.last_odom = None
        self.last_image = None
        return dst.dist

    def get_position(self):
        assert self.particles is not None
        return np.mean(self.particles, axis=1)

    def process_images(self, imgsA, imgsB):
        try:
            resp1 = self.nn_service(imgsA, imgsB)
            return resp1
        except rospy.ServiceException as e:
            rospy.logwarn("Service call failed: %s" % e)
            return None

    def processOS(self, odom_msg, img_msg):
        # get data
        if self.last_odom is None or self.particles is None or self.last_image is None:
            self.last_odom = odom_msg
            self.last_image = img_msg.live_images
            return None, False

        imgsA = img_msg.map_images
        imgsB = img_msg.live_images
        dists = img_msg.distances
        dx = self.last_odom.pose.pose.position.x - odom_msg.pose.pose.position.x
        dy = self.last_odom.pose.pose.position.y - odom_msg.pose.pose.position.y
        dz = self.last_odom.pose.pose.position.z - odom_msg.pose.pose.position.z
        dist_diff = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
        self.raw_odom += dist_diff

        # TODO: visual only and odom only needs to output alignment too!
        if self.visual_only:
            hists = self.process_images(imgsA, imgsB)
            hists = np.array([hist.data for hist in hists.histograms])
            time_hist = np.max(hists, axis=-1)
            prob_interp = interpolate.interp1d(dists, time_hist, kind="quadratic")
            interp_list = np.linspace(dists[0], dists[-1], len(dists)*8)  # 8 is magic number
            interp_out = prob_interp(interp_list)
            output = interp_list[np.argmax(interp_out)]
            rospy.logwarn("Estimated distance: " + str(output))
            return output, True

        if self.odom_only:
            # measured distance
            self.last_odom = odom_msg
            return self.raw_odom, True

        # measured distance
        dist_diff = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
        rospy.logwarn("curr img times:" + str(imgsA.data[0].header.stamp.secs) + ", " + str(imgsB.data[0].header.stamp.secs))
        # clip particles to available image span
        self.particles[0] = np.clip(self.particles[0], min(dists), max(dists))
        self.particles[1] = np.clip(self.particles[1], -255, 255)
        # get time histogram
        imgsA.data.append(self.last_image.data[0])

        # sensor step ---
        service_out = self.process_images(imgsA, imgsB)
        hists = np.array([hist.data for hist in service_out.histograms[:-1]])
        # interpolate
        hist_width = np.shape(hists)[1]
        xx, yy = np.meshgrid(dists, np.linspace(-255, 255, hist_width))
        prob_interp = interpolate.interp2d(xx, yy, np.transpose(hists), kind="linear")
        # get probabilites of particles
        # particle_prob = numpy_softmax(prob_interp(self.particles[0], self.particles[1]))
        particle_prob = prob_interp(self.particles[0], self.particles[1])
        # choose best candidates and reduce the number of particles
        part_indices = np.arange(self.particles_num)
        chosen_indices = np.random.choice(part_indices, int(self.particles_num/self.particles_frac),
                                          p=particle_prob[0]/np.sum(particle_prob[0]))
        self.particles = self.particles[:, chosen_indices]
        print(self.particles.shape)

        for i in range(100):
            rospy.logwarn(self.particles[:, i])
            rospy.logwarn(particle_prob[0, i]/np.sum(particle_prob[0]))

        # motion step
        hist_diff = np.argmax(np.array(service_out.histograms[-1].data)) - hist_width//2

        self.particles = np.concatenate([np.transpose(self.particles) + np.array((dist_diff, hist_diff)) +
                                         np.random.normal(loc=(0, 0),
                                                          scale=(self.odom_var * dist_diff, self.displac_var),
                                                          size=(self.particles.shape[1], 2))
                                         for _ in range(self.particles_frac)]).transpose()
        rospy.logwarn(np.array((dist_diff, hist_diff)))
        particles_out = self.particles.flatten()
        self.particles_pub.publish(particles_out)
        rospy.logwarn("Outputted position: " + str(np.mean(self.particles)) + " +- " + str(np.std(self.particles)) + " vs raw odom: " + str(self.raw_odom))
        self.last_odom = odom_msg
        self.last_image = imgsB
        return self.get_position(), True

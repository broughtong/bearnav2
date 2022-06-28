import numpy as np
from base_classes import DisplacementEstimator, RelativeDistanceEstimator, AbsoluteDistanceEstimator, SensorFusion
import rospy
from bearnav2.srv import Alignment, AlignmentResponse, SetDist, SetDistResponse
from bearnav2.msg import FloatList, SensorsInput, ImageList
from scipy import interpolate

"""
Here should be placed all classes for fusion of sensor processing
"""


class BearnavClassic(SensorFusion):

    def __init__(self, abs_align_est: DisplacementEstimator, abs_dist_est: AbsoluteDistanceEstimator):
        super().__init__(abs_align_est=abs_align_est, abs_dist_est=abs_dist_est)

    def _process_rel_alignment(self, msg):
        rospy.logwarn("This function is not available for this fusion class")
        raise Exception("Bearnav Classic does not support relative alignment")

    def _process_abs_alignment(self, msg):
        # This is not ideal since we assume certain message type beforhand - however this class should be message agnostic!
        # msg.map_images.data = [msg.map_images.data[len(msg.map_images.data) // 2]]     # choose only the middle image
        histogram = self.abs_align_est.displacement_message_callback(msg)
        self.alignment = (np.argmax(histogram) - np.size(histogram)//2) / (np.size(histogram)//2)
        rospy.logwarn("Current displacement: " + str(self.alignment))
        self.publish_align()

    def _process_rel_distance(self, msg):
        rospy.logwarn("This function is not available for this fusion class")
        raise Exception("Bearnav Classic does not support relative distance")

    def _process_abs_distance(self, msg):
        self.distance = self.abs_dist_est.abs_dist_message_callback(msg)
        self.publish_dist()

    def _process_prob_distance(self, msg):
        rospy.logwarn("This function is not available for this fusion class")
        raise Exception("Bearnav Classic does not support probability of distances")


class PF2D(SensorFusion):

    # TODO: everything here must be changed from pixelwise to relative image width

    def __init__(self, particles_num: int, odom_error: float, odom_init_std: float,
                 align_error: float, align_init_std: float, particles_frac: int, debug: bool,
                 abs_align_est: DisplacementEstimator, rel_align_est: DisplacementEstimator,
                 rel_dist_est: RelativeDistanceEstimator):
        super(PF2D, self).__init__(abs_align_est=abs_align_est, rel_align_est=rel_align_est, rel_dist_est=rel_dist_est)

        self.odom_error = odom_error
        self.align_error = align_error
        self.odom_init_std = odom_init_std
        self.align_init_std = align_init_std

        self.particles_num = particles_num
        self.particles_frac = particles_frac
        self.last_image = None
        self.last_odom = None
        self.particles = None
        self.traveled_dist = 0

        # For debugging
        self.debug = debug
        if debug:
            self.particles_pub = rospy.Publisher("particles", FloatList, queue_size=1)

    def set_distance(self, msg: SetDist) -> SetDistResponse:
        ret = super(PF2D, self).set_distance(msg)
        var = (self.odom_init_std, self.align_init_std)
        dst = self.distance
        self.particles = np.transpose(np.ones((2, self.particles_num)).transpose() * np.array((dst, 0)) +\
                                      np.random.normal(loc=(0, 0), scale=var, size=(self.particles_num, 2)))
        self.last_image = None
        rospy.logwarn("Particles reinitialized at position " + str(dst) + "m")
        return ret

    def _process_rel_alignment(self, msg):
        histogram = self.rel_align_est.displacement_message_callback(msg)
        out = AlignmentResponse()
        out.histograms = histogram
        return out

    def _process_abs_alignment(self, msg):
        # get everything
        hists = np.array(self.abs_align_est.displacement_message_callback(msg))
        trans = np.array(msg.map_transitions)
        dists = np.array(msg.map_distances)
        traveled = self.traveled_dist

        # sensor step --------------------------------------------------------------------
        # interpolate
        hist_width = np.shape(hists)[1]
        xs, ys = np.meshgrid(dists, np.linspace(-255, 255, hist_width))
        positions = np.vstack([xs.ravel(), ys.ravel()])
        idx, idy = np.meshgrid(np.arange(hists.shape[0]), np.arange(hists.shape[1]))
        indices = np.vstack([idx.ravel(), idy.ravel()])
        particle_prob = interpolate.griddata(np.transpose(positions),
                                             hists[indices[0], indices[1]],
                                             (self.particles[0], self.particles[1]),
                                             method="nearest")
        # get probabilites of particles
        particle_prob = self._numpy_softmax(particle_prob)
        # choose best candidates and reduce the number of particles
        part_indices = np.arange(self.particles_num)
        chosen_indices = np.random.choice(part_indices, int(self.particles_num/self.particles_frac),
                                          p=particle_prob/np.sum(particle_prob))
        self.particles = self.particles[:, chosen_indices]

        # motion step --------------------------------------------------------------------------------
        if self.last_image is not None:
            rel_msg = SensorsInput()
            rel_msg.live_images = msg.live_images
            last_img_msg = ImageList()
            last_img_msg.data = [self.last_image]
            rel_msg.map_images = last_img_msg
            hists = self.rel_align_est.displacement_message_callback(rel_msg)
            curr_img_diff = (np.argmax(np.array(hists[0])) - hist_width//2) * 8
        else:
            curr_img_diff = 0

        # get map transition for each particle
        mat_dists = np.transpose(np.matrix(dists))
        p_distances = np.matrix(self.particles[0, :])
        # rospy.logwarn(np.argmin(np.abs(mat_dists - p_distances)))
        closest_transition = np.transpose(np.clip(np.argmin(np.abs(mat_dists - p_distances), axis=0), 0, len(dists) - 2))
        dists_diffs = np.diff(dists)
        traveled_fracs = traveled / dists_diffs
        # rospy.logwarn(str(np.mean(closest_transition)) + " +- " + str(np.std(closest_transition)) + " " + str(np.shape(closest_transition)))
        trans_cumsum_per_particle = trans[closest_transition]
        frac_per_particle = traveled_fracs[closest_transition]
        # generate new particles
        out = []
        trans_diff = None
        for _ in range(self.particles_frac):
            rolls = np.random.rand(self.particles.shape[1])
            indices = self._first_nonzero(np.matrix(trans_cumsum_per_particle) >= np.transpose(np.matrix(rolls)), 1)
            trans_diff = np.array(((indices - hist_width//2) * 8) * frac_per_particle)
            particle_shifts = np.concatenate((np.ones(trans_diff.shape) * traveled, curr_img_diff - trans_diff), axis=1)
            moved_particles = np.transpose(self.particles) + particle_shifts + \
                              np.random.normal(loc=(0, 0),
                                               scale=(self.odom_error * traveled, self.align_error),
                                               size=(self.particles.shape[1], 2))
            out.append(moved_particles)
        self.particles = np.concatenate(out).transpose()

        # rospy.logwarn(np.array((dist_diff, hist_diff)))
        particles_out = self.particles.flatten()
        self.particles_pub.publish(particles_out)
        rospy.logwarn("Outputted position: " + str(np.mean(self.particles[0, :])) + " +- " + str(np.std(self.particles[0, :])) + " vs raw odom: " + str(self.raw_odom))
        rospy.logwarn("Outputted alignment: " + str(np.mean(self.particles[1, :])) + " +- " + str(np.std(self.particles[1, :])) + " with transition: " + str(np.mean(trans_diff)))
        self.last_image = msg.live_images

    def _process_rel_distance(self, msg):
        # only increment the distance
        dist = self.rel_dist_est.rel_dist_message_callback(msg)
        self.traveled_dist += dist
        self.particles[0] += dist

    def _process_abs_distance(self, msg):
        rospy.logwarn("This function is not available for this fusion class")
        raise Exception("PF2D does not support absolute distance")

    def _process_prob_distance(self, msg):
        rospy.logwarn("This function is not available for this fusion class")
        raise Exception("PF2D does not support distance probabilities")

    def _numpy_softmax(self, arr):
        tmp = np.exp(arr) / np.sum(np.exp(arr))
        return tmp

    def _first_nonzero(self, arr, axis, invalid_val=-1):
        mask = arr != 0
        return np.array(np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val))
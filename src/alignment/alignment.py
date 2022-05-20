#!/usr/bin/env python
import yaml
import histogram
import numpy as np
import os
import rospy
from torchvision import transforms
from scipy import interpolate
import time
from backends import traditional
from std_msgs.msg import Header, Float32
from bearnav2.srv import SiameseNet
from bearnav2.msg import ImageList


PEAK_MULT = 1.0
NEWTORK_DIVISION = 8.0
RESIZE_W = 512


def numpy_softmax(arr):
    return np.exp(arr) / np.sum(np.exp(arr))


class Alignment:

    def __init__(self):

        self.method = "SIAM"
        self.traditionalMethods = ["SIFT", "SURF", "KAZE", "AKAZE", "BRISK", "ORB"]
        rospy.logwarn("Alignment method:" + self.method)
        if self.method == "SIAM":
            rospy.wait_for_service('siamese_network')
            self.nn_service = rospy.ServiceProxy('siamese_network', SiameseNet)

    def process(self, imgA, imgB):

        rospy.logwarn("recieved pair of imgs")
        peak, uncertainty = 0, 0
        hist = []

        if self.method in self.traditionalMethods: 
            from backends import traditional
            kpsA, desA = traditional.detect(imgA, self.method)
            kpsB, desB = traditional.detect(imgB, self.method)

            if kpsA is None or kpsB is None:
                return peak, 0, hist

            displacements = traditional.match(kpsA, desA, kpsB, desB)
            displacements = [int(x) for x in displacements]

            hist = histogram.slidingHist(displacements, 10)
            peak, n = histogram.getHistPeak(hist)

            h = {}
            for i in hist:
                h.update(i)

            yVals = []
            for x in range(min(h), max(h) + 1):
                yVals.append(h[x])
            hist = yVals

            print(peak, n)
            rospy.logwarn("===")
            rospy.logwarn(peak)
            rospy.logwarn(n)

            if n < 10:
                peak = 0

        elif self.method == "VGG":
            from backends import vgg

            print(imgA.shape, imgB.shape, "SHAPRES")

            if imgA.shape[-1] == 4:
                print("WARNING 4D image!")
                imgA = imgA[:,:,:3]
            
            peak, val, hist = vgg.align(imgA, imgB)
            print(peak, val)

        elif self.method == "SIAM":
            rospy.loginfo('Image pair received ...')
            start = time.time()
            # rospy.loginfo("Passing tensors:", map_tensor.shape, curr_tensor.shape)
            net_in1 = ImageList([imgA])
            net_in2 = ImageList([imgB])
            try:
                response = self.nn_service(net_in1, net_in2)
            except rospy.ServiceException as e:
                rospy.logwarn("Service call failed: %s" % e)
                return None
            hist = response.data[0]
            rospy.loginfo("images has been aligned with histogram:")
            rospy.loginfo(str(hist))
            f = interpolate.interp1d(np.linspace(0, RESIZE_W, hist.size), hist.histograms[0].data, kind="cubic")
            interp_hist = f(np.arange(0, RESIZE_W))
            peak = (np.argmax(interp_hist) - interp_hist.size/2.0) * PEAK_MULT
            rospy.logwarn("Peak is: " + str(peak))
            end = time.time()
            rospy.logwarn("The alignment took: " + str(end - start))
            # rospy.loginfo("Outputed histogram", hist.shape)
            return peak, 0, [] 
        rospy.logwarn("No image matching scheme selected! Not correcting heading!")
        return peak, 0, hist

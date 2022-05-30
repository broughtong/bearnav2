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

PAD = 32
PEAK_MULT = 0.5
NEWTORK_DIVISION = 8.0
RESIZE_H = 320
RESIZE_W = 512


class Alignment:

    def __init__(self):

        self.method = "SIAM"
        self.traditionalMethods = ["SIFT", "SURF", "KAZE", "AKAZE", "BRISK", "ORB"]
        rospy.logwarn(self.method) 
        if self.method == "SIAM":
            from backends.siam_model import Siamese, load_model, get_parametrized_model
            import torch as t
            self.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
            # init neural network
            model = get_parametrized_model(False, 3, 256, 0, 3, self.device)
            file_path = os.path.dirname(os.path.abspath(__file__))
            self.model = load_model(model, os.path.join(file_path, "backends/model_eunord.pt")).to(self.device)
            self.model.eval()
            self.to_tensor = transforms.ToTensor()
            self.resize = transforms.Resize(RESIZE_H)
            rospy.logwarn("Neural network sucessfully initialized!")

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
            import torch as t
            with t.no_grad():
                rospy.loginfo('Image pair received ...')
                start = time.time()
                curr_tensor = self.image_to_tensor(imgB)
                map_tensor = self.image_to_tensor(imgA)
                # rospy.loginfo("Passing tensors:", map_tensor.shape, curr_tensor.shape)
                hist = self.model(map_tensor, curr_tensor, padding=PAD)
                hist_out = t.softmax(hist, dim=-1)
                hist = hist.cpu().numpy()
                #rospy.logwarn(str(hist_out))
                #rospy.logwarn("images has been aligned with histogram:")
                # rospy.logwarn(str(list(hist_out)))
                # TODO: interpolate the histogram!
                f = interpolate.interp1d(np.linspace(0, RESIZE_W, hist.size), hist, kind="cubic")
                interp_hist = f(np.arange(0, RESIZE_W))
                peak = (np.argmax(interp_hist) - interp_hist.size/2.0) * PEAK_MULT
                #rospy.logwarn("Peak is: " + str(peak))
                end = time.time()
                #rospy.logwarn("The alignment took: " + str(end - start))
                # rospy.loginfo("Outputed histogram", hist.shape)
            return peak, 0, [] 
        rospy.logwarn("No image matching scheme selected! Not correcting heading!")
        return peak, 0, hist

    def image_to_tensor(self, msg):
        msg = np.array(msg)
        image_tensor = self.resize(self.to_tensor(msg).unsqueeze(0)).to(self.device)
        return image_tensor

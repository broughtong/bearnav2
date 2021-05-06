#!/usr/bin/env python
import yaml
import histogram
import numpy as np

class Alignment:

    def __init__(self):

        self.method = "BRISK"
        self.traditionalMethods = ["SIFT", "SURF", "KAZE", "AKAZE", "BRISK", "ORB"]
               
    def process(self, imgA, imgB):

        peak, uncertainty = 0, 0
        hist = []

        if self.method in self.traditionalMethods: 
            from backends import traditional
            kpsA, desA = traditional.detect(imgA, self.method)
            kpsB, desB = traditional.detect(imgB, self.method)

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

        return peak, 0, hist

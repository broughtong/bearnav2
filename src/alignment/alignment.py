#!/usr/bin/env python
import yaml
import histogram
from backends import traditional, vgg
import numpy as np

class Alignment:

        def __init__(self, configFilename):

                self.readConfig(configFilename)
                self.method = "VGG"
                self.traditionalMethods = ["SIFT", "SURF", "KAZE", "AKAZE", "BRISK", "ORB"]

        def readConfig(self, filename):

                data = {}
                with open(filename, "r") as file:
                        data = yaml.safe_load(file)

                for key in data.keys():
                        self.key = data[key]
                        
        def process(self, imgA, imgB):
        
                peak, uncertainty = 0, 0

                if self.method in self.traditionalMethods: 
                        print("Using sift for trad align")
                        kpsA, desA = traditional.detect(imgA, self.method)
                        kpsB, desB = traditional.detect(imgB, self.method)

                        displacements = traditional.match(kpsA, desA, kpsB, desB)
                        displacements = [int(x) for x in displacements]

                        hist = histogram.slidingHist(displacements, 10)
                        peak, n = histogram.getHistPeak(hist)
                        print(peak, n)

                elif self.method == "VGG":
            
                        print(imgA.shape, imgB.shape, "SHAPRES")

                        if imgA.shape[-1] == 4:
                                print("WARNING 4D image!")
                                imgA = imgA[:,:,:3]
                        
                        peak, val = vgg.align(imgA, imgB)
                        print(peak, val)

                return peak, 0

if __name__ == "__main__":
        print(traditional)



                

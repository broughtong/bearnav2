#!/usr/bin/env python
import yaml
import histogram
from backends import traditional, vgg

class Alignment:

        def __init__(self, configFilename):

                self.readConfig(configFilename)
        self.method = "SIFT"
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
                        kpsA, desA = traditional.detect(imgA, self.method)
                        kpsB, desB = traditional.detect(imgB, self.method)

                        displacements = traditional.match(kpsA, desA, kpsB, desB)
                        displacements = [int(x) for x in displacements]

                        hist = histogram.slidingHist(displacements, 10)
                        peak = histogram.getHistPeak(hist)

                elif self.method == "VGG":
            
                        peak = vgg.align(imgA, imgB)

                return peak, 0

if __name__ == "__main__":
        print(traditional)



                

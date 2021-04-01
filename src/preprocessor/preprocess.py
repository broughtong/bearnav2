#!/usr/bin/env python
import cv2
import yaml

class Preprocessor:

    def __init__(self, configFilename):

        self.readConfig(configFilename)

        self.preprocess_image = True
        self.use_hist_equalisation = True

    def readConfig(self, filename):

        data = {}
        with open(filename, "r") as file:
            data = yaml.safe_load(file)

        for key in data.keys():
            self.key = data[key]
            
    def process(self, img):

        if self.preprocess_img == False:
            return img

        if self.use_hist_equalisation:
            img = cv2.equalizeHist(img)

        return img
        


#!/usr/bin/env python3
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

		self.preprocess_image = data["preprocess_image"]
		self.use_hist_equalisation = data["use_hist_equalisation"]

			
	def process(self, img):

		if self.preprocess_img == False:
			return img

		if self.use_hist_equalisation:
			img = cv2.equalizeHist(img)

		return img
		


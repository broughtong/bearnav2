#!/usr/bin/env python3
import yaml
import histogram
from backend import traditional

class Alignment:

	def __init__(self, configFilename):

		self.readConfig(configFilename)

	def readConfig(self, filename):

		data = {}
		with open(filename, "r") as file:
			data = yaml.safe_load(file)

		for key in data.keys():
			self.key = data[key]
			
	def process(self, imgA, imgB):
	
		kpsA, desA = traditional.detect(imgA, "SIFT")
		kpsB, desB = traditional.detect(imgB, "SIFT")

		displacements = traditional.match(kpsA, desA, kpsB, desB)

		hist = histogram.slidingHist(displacements, 10)

		peak = histogram.getHistPeak(hist)

		return peak, 1



		

#!/usr/bin/env python
import yaml
import histogram
from backends import traditional

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
                displacements = [int(x) for x in displacements]

		hist = histogram.slidingHist(displacements, 10)
		peak = histogram.getHistPeak(hist)

		return peak, 0

if __name__ == "__main__":
        print(traditional)



		

#!/usr/bin/env python

class Navigator:

	def __init__(self, ):
        self alignment = None
        self uncertainty = None
        self useUncertainty = True
        self turnGain = 0.1 #turn 0.1 rad per each pixel of error
        self velocityGain = 1 # 1 is same speed as thought map, less is slower more is faster
	def process(self, msg):

        correction = self.alignment * turnGain # angle = px * angle/pixel
        if useUncertainty:
            correction = correction * (1 - self.uncertainty)
        out = Twist()
        out.linear.x = msg.linear.x * velocityGain
        out.linear.y = msg.linear.y * velocityGain
        out.linear.z = msg.linear.z * velocityGain
        out.angular.x = msg.angular.x * velocityGain 
        out.angular.y = msg.angular.y * velocityGain
        out.angular.z = msg.angular.z * velocityGain + correction
		return out

    def reconfig(cfg):
        self.useUncertainty = cfg['use_uncertainty']
        self.turnGain = cfg['turn_gain']
        self.velocityGain = cfg['velocity_gain']

    def correction(msg):
		self.alignment = msg.alignment #Px
        self.uncertainty = msg.uncertainty


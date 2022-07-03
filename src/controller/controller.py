#!/usr/bin/env python
from geometry_msgs.msg import Twist

class Controller:

    def __init__(self):
        self.alignment = 0
        self.uncertainty = 0
        self.useUncertainty = True
        self.turnGain = 0.5  #turn 0.1 rad per each pixel of error
        self.velocityGain = 1 # 1 is same speed as thought map, less is slower more is faster

    def process(self, msg):
        correction = self.alignment * self.turnGain # angle = px * angle/pixel
        if self.useUncertainty:
            correction = correction * (1 - self.uncertainty)
        out = Twist()
        out.linear.x = msg.linear.x * self.velocityGain
        out.linear.y = msg.linear.y * self.velocityGain
        out.linear.z = msg.linear.z * self.velocityGain
        out.angular.x = msg.angular.x * self.velocityGain 
        out.angular.y = msg.angular.y * self.velocityGain
        out.angular.z = msg.angular.z * self.velocityGain + correction
        return out

    def reconfig(self,cfg):
        self.useUncertainty = cfg['use_uncertainty']
        self.turnGain = cfg['turn_gain']
        self.velocityGain = cfg['velocity_gain']

    def correction(self,msg):
        self.alignment = msg.output #Px
        # self.uncertainty = msg.uncertainty


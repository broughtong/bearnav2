#!/usr/bin/env python3
import rospy
import navigator
from geometry_msgs.msg import Twist
from bearnav2.msg import Alignment
from bearnav2.cfg import NavigatorConfig
from dynamic_reconfigure.server import Server

pub = None
n = navigator.Navigator()

def callbackVel(msg):
	driven = n.process(msg)
	pub.publish(driven)

def callbackCorr(msg):
    n.correction(msg)

def callbackReconfigure(config,level):
    n.reconfig(config)
    return config

if __name__ == "__main__":
    rospy.init_node("navigator")
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=0)
    rospy.Subscriber("/map_vel", Twist, callbackVel)
    rospy.Subscriber("/correction_cmd", Alignment, callbackCorr)
    srv = Server(NavigatorConfig, callbackReconfigure)
    rospy.spin()

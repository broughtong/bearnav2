#!/usr/bin/env python3
import rospy
import navigator
from geometry_msgs.msg import Twist
from bearnav2.msg import Alignment
from bearnav2.cfg import NavigatorConfig
pub = None
n = navigator.Navigator("./config.yaml")

def callbackVel(msg):
	driven = n.process(msg)
	publisher.publish(driven)

def callbackCorr(msg):
    n.correction(msg)

def callbackReconfigure(config):
    n.reconfig(config)

if __name__ == "__main__":
    rospy.init_node("navigator")
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=0)
    rospy.Subscriber("/map_vel", Twist, callback)
    rospy.Subscriber("/correction_cmd", Alignment, callback)
    srv = Server(NavigatorConfig, callbackReconfigure)
    rospy.spin()

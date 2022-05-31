#!/usr/bin/env python
import rospy
import navigator
from geometry_msgs.msg import Twist
from bearnav2.msg import Alignment
from bearnav2.cfg import NavigatorConfig
from bearnav2.srv import SetClockGain, SetClockGainResponse
from dynamic_reconfigure.server import Server

pub = None
n = navigator.Navigator()
gainSrv = None

def callbackVel(msg):
    driven = n.process(msg)
    pub.publish(driven)

def callbackCorr(msg):
    n.correction(msg)

def callbackReconfigure(config,level):
    n.reconfig(config)

    #if velocity is increased, time at that velocity needs
    #to be decreased, so we invert the clock gain
    try:
        gain = 1/config["velocity_gain"]
        if type(gain) == float: 
            gainSrv(gain)
        else:
            gainSrv(1.0)
    except:
        rospy.logwarn("Unable to set clock gain")
        
    return config

if __name__ == "__main__":
    rospy.init_node("navigator")
    rospy.wait_for_service('set_clock_gain')
    gainSrv = rospy.ServiceProxy('set_clock_gain', SetClockGain)
    cmd_vel_topic = rospy.get_param("~cmd_vel_topic")
    pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=0)
    rospy.Subscriber("map_vel", Twist, callbackVel)
    rospy.Subscriber("correction_cmd", Alignment, callbackCorr)
    srv = Server(NavigatorConfig, callbackReconfigure)
    rospy.spin()

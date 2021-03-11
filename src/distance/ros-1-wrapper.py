#!/usr/bin/env python3
import rospy
import distance
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odom
from bearnav_2.srv import SetDist

pub = None


def callbackTwist(msg):
    driven, use = d.processT(msg)
    if use:
        publisher.publish(driven)

def callbackOdom(msg):
    driven, use = d.processO(msg)
    if use:
        publisher.publish(driven)

def handle_set_dist(dst):
    d.set(dst)


if __name__ == "__main__":

    rospy.init_node("distance")
    use_twist = rospy.get_param("use_twist",'False')
    d = distance.Distance(use_twist)
    pub = rospy.Publisher("/distance", Float64, queue_size=0) 
    rospy.Subscriber("/odom", Odom, callbackOdoom)
    rospy.Subscriber("/cmd_vel",Twist , callbackTwist)
    s = rospy.Service('set_dist', SetDist, handle_add_two_ints)
    rospy.spin()

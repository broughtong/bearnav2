#!/usr/bin/env python
import rospy
import distance
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from bearnav2.srv import SetDist, SetDistResponse

pub = None


def callbackTwist(msg):
    driven, use = d.processT(msg)
    if use:
        pub.publish(driven)

def callbackOdom(msg):
    driven, use = d.processO(msg)
    if use:
        pub.publish(driven)

def handle_set_dist(dst):
    driven = d.set(dst)
    print("Distance set to " + str(driven))
    pub.publish(driven)
    return SetDistResponse()


if __name__ == "__main__":

    rospy.init_node("distance")
    use_twist = rospy.get_param("use_twist",'False')
    d = distance.Distance(use_twist)
    pub = rospy.Publisher("/distance", Float64, queue_size=0) 
    rospy.Subscriber("/odometry/filtered", Odometry, callbackOdom)
    rospy.Subscriber("/cmd_vel",Twist , callbackTwist)
    s = rospy.Service('set_dist', SetDist, handle_set_dist)
    rospy.spin()

#!/usr/bin/env python
import rospy
import distance
import distance_pf
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from bearnav2.srv import SetDist, SetDistResponse
from bearnav2.msg import PFInput

pub = None

def callbackTwist(msg):
    driven, use = d.processT(msg)
    if use:
        pub.publish(driven)

def callbackOdom(msg):
    driven, use = d.processO(msg)
    if use:
        pub.publish(driven)

def callbackCamera(msg):
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
    cmd_vel_topic = rospy.get_param("~cmd_vel_topic")
    odom_topic = rospy.get_param("~odom_topic")
    # d = distance.Distance(use_twist)
    d = distance_pf.DistancePF(False)
    pub = rospy.Publisher("distance", Float64, queue_size=1)
    rospy.Subscriber(odom_topic, Odometry, callbackOdom, queue_size=1)
    rospy.Subscriber(cmd_vel_topic, Twist, callbackTwist, queue_size=1)
    rospy.Subscriber("pf_img_input", PFInput, callbackCamera)  # TODO: make name of topic as argument
    s = rospy.Service('set_dist', SetDist, handle_set_dist)
    rospy.spin()

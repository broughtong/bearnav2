#!/usr/bin/env python
import rospy
import distance
import distance_pf
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from bearnav2.srv import SetDist, SetDistResponse
from bearnav2.msg import PFInput, Alignment, FloatList
import message_filters

pub = None

def callbackOdomCamera(odom_msg, img_msg):
    particle, use = d.processOS(odom_msg, img_msg)
    if use:
        rospy.logwarn(particle)
        distance, displac = particle[0], particle[1]
        pub.publish(distance)
        m = Alignment()
        m.alignment = displac
        m.uncertainty = 0
        pub_align.publish(m)

def handle_set_dist(dst):
    driven = d.set(dst)
    print("Distance set to " + str(driven))
    return SetDistResponse()

if __name__ == "__main__":

    rospy.init_node("distance")
    use_twist = rospy.get_param("use_twist",'False')
    cmd_vel_topic = rospy.get_param("~cmd_vel_topic")
    odom_topic = rospy.get_param("~odom_topic")
    d = distance_pf.DistancePF()
    pub = rospy.Publisher("distance", Float32, queue_size=1)
    sub1 = message_filters.Subscriber(odom_topic, Odometry)
    sub2 = message_filters.Subscriber("pf_img_input", PFInput)
    ts = message_filters.ApproximateTimeSynchronizer([sub1, sub2], 1, 1)
    ts.registerCallback(callbackOdomCamera)
    pub_align = rospy.Publisher("alignment/output", Alignment, queue_size=1)
    s = rospy.Service('set_dist', SetDist, handle_set_dist)
    rospy.spin()

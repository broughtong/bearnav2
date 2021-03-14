#!/usr/bin/env python
from bearnav.srv import mapmaker
import rospy

def add_two_ints_server():
    rospy.init_node('add_two_ints_server')
    s = rospy.Service('add_two_ints', AddTwoInts, handle_add_two_ints)
    print("Ready to add two ints.")
    rospy.spin()

if __name__ == "__main__":
    rospy.init_node("mapmaker_server")
    s = rospy.Service("makemaker", mapmaker, callback)
    print("Mapmaker running, awaiting instructions")
    rospy.spin()


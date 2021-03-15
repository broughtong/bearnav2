#!/usr/bin/env python
import rospy
import os
import actionlib
import cv2
import rosbag
from sensor_msgs.msg import Image, Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from bearnav2.msg import MapRepeaterAction, MapRepeaterFeedback
from bearnav2.srv import SetDist
from cv_bridge import CvBridge

class ActionServer():
    #_feedback = bearnav2.msg.MapMakerFeedback()
    #_result = bearnav2.msg.MapMakerResult()

    def __init__(self):

        #some vars
        self.br = CvBridge()
        self.img = None
        self.mapName = ""
        self.mapStep = 20
        self.nextStep = 0
        self.bag = None

        print("Waiting for services to become available...")
        rospy.wait_for_service("set_dist")

        print("Resetting distance node")
        self.distance_reset_srv = rospy.ServiceProxy("set_dist", SetDist)
        self.distance_reset_srv(0)
        self.distance_sub = rospy.Subscriber("/distance", Float64, self.distanceCB)

        print("Subscibing to cameras")
        self.cam_sub = rospy.Subscriber("/camera_2/image_rect_color", Image, self.imageCB)

        print("Setting up published for commands")
        self.joy_topic = "cmd_vel"
        self.joy_pub = rospy.Publisher(self.joy_topic, Twist, queue_size=0)

        print("Starting mapmaker server")
        self.server = actionlib.SimpleActionServer("replayer", MapRepeaterAction, execute_cb=self.actionCB, auto_start=False)
        self.server.start()
        print("Server started, awaiting goal")

    def imageCB(self, msg):
        return
        self.img = self.br.imgmsg_to_cv2(msg)
        self.checkShutdown()

    def distanceCB(self, msg):


        return
        dist = msg.data
        if dist >= self.nextStep:
            print("Triggered wp")
            self.nextStep += self.mapStep
            print(self.mapName)
            print(str(dist))
            filename = os.path.join(self.mapName, str(int(dist)) + ".jpg")
            cv2.imwrite(filename, self.img)

        self.checkShutdown()

    def joyCB(self, msg):
        return
        if self.isMapping:
            print("Adding joy")
            self.bag.write(self.joy_topic, msg) 

    def actionCB(self, goal):

        print(goal)

        if goal.mapName == "":
            print("Missing mapname")

        if not os.path.isdir(goal.mapName):
            print("Can't find map directory")

        if not os.path.isfile(os.path.join(goal.mapName, goal.mapName + ".bag")):
            print("Can't find commands")

        print("Starting repeat")
        self.bag = rosbag.Bag(os.path.join(goal.mapName, goal.mapName + ".bag"), "r")
        self.mapName = goal.mapName

        #set distance to zero
        self.distance_reset_srv(0)

        #replay bag
        start = rospy.Time.now()
        sim_start = None
        print("Starting repeat")
        for topic, message, ts in self.bag.read_messages():
            now = rospy.Time.now()
            if sim_start is None:
                sim_start = ts
            else:
                real_time = now - start
                sim_time = ts - sim_start
                if sim_time > real_time:
                    rospy.sleep(sim_time - real_time)
            self.joy_pub.publish(message)
            if rospy.is_shutdown():
                break

        print("Complete")
         
    def checkShutdown(self):
        if self.server.is_preempt_requested():
            self.shutdown()

    def shutdown(self):
        self.bag.close()

if __name__ == '__main__':
    rospy.init_node("replayer_server")
    server = ActionServer()
    rospy.spin()
    server.shutdown()

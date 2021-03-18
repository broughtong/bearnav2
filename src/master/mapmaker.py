#!/usr/bin/env python
import rospy
import os
import actionlib
import cv2
import rosbag
from sensor_msgs.msg import Image, Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from bearnav2.msg import MapMakerAction, MapMakerFeedback
from bearnav2.srv import SetDist
from cv_bridge import CvBridge

class ActionServer():
    #_feedback = bearnav2.msg.MapMakerFeedback()
    #_result = bearnav2.msg.MapMakerResult()

    def __init__(self):

        #some vars
        self.br = CvBridge()
        self.isMapping = False
        self.img = None
        self.mapName = ""
        self.mapStep = 0.3
        self.nextStep = 0
        self.bag = None
        self.lastDistance = None

        print("Waiting for services to become available...")
        rospy.wait_for_service("set_dist")

        print("Resetting distance node")
        self.distance_reset_srv = rospy.ServiceProxy("set_dist", SetDist)
        self.distance_reset_srv(0)
        self.distance_sub = rospy.Subscriber("/distance", Float64, self.distanceCB)

        print("Subscibing to cameras")
        self.cam_sub = rospy.Subscriber("/camera_2/image_rect_color", Image, self.imageCB)

        print("Subscibing to commands")
        #self.joy_topic = "joy_teleop/joy"
        #self.joy_sub = rospy.Subscriber(self.joy_topic, Joy, self.joyCB)
        self.joy_topic = "cmd_vel"
        self.joy_topic = "/husky_velocity_controller/cmd_vel"
        self.joy_sub = rospy.Subscriber(self.joy_topic, Twist, self.joyCB)

        print("Starting mapmaker server")
        self.server = actionlib.SimpleActionServer("mapmaker", MapMakerAction, execute_cb=self.actionCB, auto_start=False)
        self.server.start()
        print("Server started, awaiting goal")

    def imageCB(self, msg):

        self.img = self.br.imgmsg_to_cv2(msg)
        self.checkShutdown()

    def distanceCB(self, msg):

        if self.isMapping == False or self.img is None:
            return

        dist = msg.data
        self.lastDistance = dist
        if dist >= self.nextStep:
            if self.img is None:
                print("Warning: no image received!")

            print("Triggered wp")
            self.nextStep += self.mapStep
            print(self.mapName)
            print(str(dist))
            filename = os.path.join(self.mapName, str(dist) + ".jpg")
            cv2.imwrite(filename, self.img)

        self.checkShutdown()

    def joyCB(self, msg):

        if self.isMapping:
            print("Adding joy")
            self.bag.write(self.joy_topic, msg) 

    def actionCB(self, goal):

        print(goal)

        if self.img is None:
            print("WARNING: NO IMAGE INPUT RECEIVED")

        if goal.mapName == "":
            print("Missing mapname")

        if goal.start == True:
            print("Starting mapping")
            try:
                os.mkdir(goal.mapName)
                with open(goal.mapName + "/params", "w") as f:
                    f.write("stepSize: " + str(self.mapStep))
            except:
                pass
            self.bag = rosbag.Bag(os.path.join(goal.mapName, goal.mapName + ".bag"), "w")
            self.mapName = goal.mapName
            self.isMapping = True

        else:
            print("Creating final wp")
            filename = os.path.join(self.mapName, str(self.lastDistance) + ".jpg")
            cv2.imwrite(filename, self.img)
            print("Stopping Mapping")
            self.isMapping = False
            self.bag.close()

        #set distance to zero
        self.distance_reset_srv(0)
         
    def checkShutdown(self):
        if self.server.is_preempt_requested():
            self.shutdown()

    def shutdown(self):
        self.isMapping = False
        self.bag.close()
        
        
    
"""

    def execute_cb(self, goal):
        success = True
        
        # append the seeds for the fibonacci sequence
        self._feedback.sequence = []
        self._feedback.sequence.append(0)
        self._feedback.sequence.append(1)
        
        # start executing the action
        for i in range(1, goal.order):
            # check that preempt has not been requested by the client
            if self._as.is_preempt_requested():
                rospy.loginfo('%s: Preempted' % self._action_name)
                self._as.set_preempted()
                success = False
                break
            self._feedback.sequence.append(self._feedback.sequence[i] + self._feedback.sequence[i-1])
            # publish the feedback
            self._as.publish_feedback(self._feedback)
          
        if success:
            self._result.sequence = self._feedback.sequence
            self._as.set_succeeded(self._result)
   """


if __name__ == '__main__':
    rospy.init_node("mapmaker_server")
    server = ActionServer()
    rospy.spin()
    server.shutdown()

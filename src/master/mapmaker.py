#!/usr/bin/env python
import rospy
import os
import actionlib
import cv2
from sensor_msgs.msg import Image
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
        self.mapStep = 20
        self.nextStep = 0

        print("Waiting for services to become available...")
        rospy.wait_for_service("set_dist")

        print("Resetting distance node")
        self.distance_reset_srv = rospy.ServiceProxy("set_dist", SetDist)
        self.distance_reset_srv(0)
        self.distance_sub = rospy.Subscriber("/distance", Float64, self.distanceCB)

        print("Subscibing to cameras")
        self.cam_sub = rospy.Subscriber("/camera_2/image_rect_color", Image, self.imageCB)

        print("Starting mapmaker server")
        self.server = actionlib.SimpleActionServer("mapmaker", MapMakerAction, execute_cb=self.actionCB, auto_start=False)
        self.server.start()
        print("Server started, awaiting goal")

    def imageCB(self, msg):

        self.img = self.br.imgmsg_to_cv2(msg)
        self.checkShutdown()

    def distanceCB(self, msg):

        if not self.isMapping:
            return

        dist = msg.data
        if dist >= self.nextStep:
            print("Triggered wp")
            self.nextStep += self.mapStep
            print(self.mapName)
            print(str(dist))
            filename = os.path.join(self.mapName, str(dist) + ".jpg")

            cv2.imwrite(filename, self.img)

        self.checkShutdown()

    def checkShutdown(self):
        if self.server.is_preempt_requested():
            self.isMapping = False

    def actionCB(self, goal):

        print(goal)

        if goal.start == True:
            print("Starting mapping")
            self.mapName = goal.mapName
            self.isMapping = True

        else:
            print("Stopping Mapping")
            self.isMapping = False

        #set distance to zero
        self.distance_reset_srv(0)
         
        
        
    
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

#!/usr/bin/env python
import time
import rospy
import rostopic
import os
import actionlib
import cv2
import rosbag
import roslib
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from bearnav2.msg import MapMakerAction, MapMakerResult 
from bearnav2.srv import SetDist
from cv_bridge import CvBridge


class ActionServer:

    def __init__(self):

        #some vars
        self.br = CvBridge()
        self.isMapping = False
        self.img = None
        self.mapName = ""
        self.mapStep = 1.0
        self.nextStep = 0
        self.bag = None
        self.lastDistance = None

        self.additionalTopics = rospy.get_param("~additional_record_topics")
        self.additionalTopics = self.additionalTopics.split(" ")
        self.additionalTopicSubscribers = []
        if self.additionalTopics[0] != "":
            rospy.logwarn("Recording the following additional topics: " + str(self.additionalTopics))
            for topic in self.additionalTopics:
                msgType = rostopic.get_topic_class(topic)[0]
                s = rospy.Subscriber(topic, msgType, self.miscCB, queue_size=1)
                self.additionalTopicSubscribers.append(s)
    
        rospy.loginfo("Waiting for services to become available...")
        rospy.wait_for_service("set_dist")
        rospy.loginfo("Starting...")

        rospy.logdebug("Resetting distance node")
        self.distance_reset_srv = rospy.ServiceProxy("set_dist", SetDist)
        self.distance_reset_srv(0)
        self.distance_sub = rospy.Subscriber("distance", Float32, self.distanceCB, queue_size=1)

        rospy.logdebug("Subscibing to cameras")
        self.camera_topic = rospy.get_param("~camera_topic")
        self.cam_sub = rospy.Subscriber(self.camera_topic, Image, self.imageCB, queue_size=1)

        rospy.logdebug("Subscibing to commands")
        self.joy_topic = rospy.get_param("~cmd_vel_topic")
        self.joy_sub = rospy.Subscriber(self.joy_topic, Twist, self.joyCB, queue_size=1)

        rospy.logdebug("Starting mapmaker server")
        self.server = actionlib.SimpleActionServer("mapmaker", MapMakerAction, execute_cb=self.actionCB, auto_start=False)
        self.server.start()
        rospy.loginfo("Server started, awaiting goal")

    def miscCB(self, msg, args):
        if self.isMapping:
            topicName = args
            rospy.logdebug("Adding misc from %s" % (topicName))
            self.bag.write(topicName, msg) 

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
                rospy.logwarn("Warning: no image received!")
            rospy.logdebug("Hit waypoint")
            self.nextStep += self.mapStep
            filename = os.path.join(self.mapName, str(dist) + ".jpg")
            cv2.imwrite(filename, self.img)
            rospy.logwarn("Image saved %s" % (filename))

        self.checkShutdown()

    def joyCB(self, msg):
        if self.isMapping:
            rospy.logdebug("Adding joy")
            self.bag.write(self.joy_topic, msg) 

    def actionCB(self, goal):

        if self.img is None:
            rospy.logerr("WARNING: no image coming through, ignoring")
            result = MapMakerResult()
            result.success = False
            self.server.set_succeeded(result)
            return

        if goal.mapName == "":
            rospy.logwarn("Missing mapname, ignoring")
            result = MapRepeaterResult()
            result.success = False
            self.server.set_succeeded(result)
            return

        if goal.start == True:
            self.isMapping = False
            self.img = None
            try:
                os.mkdir(goal.mapName)
                with open(goal.mapName + "/params", "w") as f:
                    f.write("stepSize: %s\n" % (self.mapStep))
                    f.write("odomTopic: %s\n" % (self.joy_topic))
            except:
                rospy.logwarn("Unable to create map directory, ignoring")
                result = MapRepeaterResult()
                result.success = False
                self.server.set_succeeded(result)
                return
            rospy.loginfo("Starting mapping")
            self.bag = rosbag.Bag(os.path.join(goal.mapName, goal.mapName + ".bag"), "w")
            self.mapName = goal.mapName
            self.nextStep = 0
            self.lastDistance = None
            self.distance_reset_srv(0)
            self.isMapping = True
        else:
            rospy.logdebug("Creating final wp")
            filename = os.path.join(self.mapName, str(self.lastDistance) + ".jpg")
            cv2.imwrite(filename, self.img)
            rospy.loginfo("Stopping Mapping")
            time.sleep(2)
            self.isMapping = False
            self.bag.close()
         
    def checkShutdown(self):
        if self.server.is_preempt_requested():
            self.shutdown()

    def shutdown(self):
        self.isMapping = False
        if self.bag is not None: 
            self.bag.close()
       
if __name__ == '__main__':
    rospy.init_node("mapmaker_server")
    server = ActionServer()
    rospy.spin()
    server.shutdown()

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
from std_msgs.msg import Float32, Header
from bearnav2.msg import MapMakerAction, MapMakerResult, SensorsOutput, SensorsInput, ImageList, DistancedTwist
from bearnav2.srv import SetDist, Alignment
from cv_bridge import CvBridge
import numpy as np
from copy import deepcopy

TARGET_WIDTH = 512
br = CvBridge()


# TODO: save representations and send features for major speedup


def save_img(img_msg, filename):
    img = br.imgmsg_to_cv2(img_msg)
    (h, w) = img.shape[:2]
    r = TARGET_WIDTH / float(w)
    dim = (TARGET_WIDTH, int(h * r))
    img = cv2.resize(img, dim)
    cv2.imwrite(filename, img)


class ActionServer:

    def __init__(self):

        #some vars
        self.isMapping = False
        self.img_msg = None
        self.last_img_msg = None
        self.mapName = ""
        self.mapStep = 1.0
        self.nextStep = 0
        self.bag = None
        self.lastDistance = 0.0
        self.visual_turn = True
        self.max_trans = 0.1
        self.curr_trans = 0.0
        self.last_saved_dist = None

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
        rospy.wait_for_service("teach/set_dist")
        rospy.loginfo("Starting...")

        rospy.logdebug("Resetting distance node")
        self.distance_reset_srv = rospy.ServiceProxy("teach/set_dist", SetDist)
        self.align_reset_srv = rospy.ServiceProxy("repeat/set_align", SetDist)
        self.distance_reset_srv(0.0)
        self.align_reset_srv(0.0)
        self.distance_sub = rospy.Subscriber("teach/output_dist", SensorsOutput, self.distanceCB, queue_size=1)

        rospy.logdebug("Subscibing to commands")
        self.joy_topic = rospy.get_param("~cmd_vel_topic")
        self.joy_sub = rospy.Subscriber(self.joy_topic, Twist, self.joyCB, queue_size=1)

        rospy.logdebug("Starting mapmaker server")
        self.server = actionlib.SimpleActionServer("mapmaker", MapMakerAction, execute_cb=self.actionCB, auto_start=False)
        self.server.start()

        if self.visual_turn:
            rospy.wait_for_service("teach/local_alignment")
            self.local_align = rospy.ServiceProxy("teach/local_alignment", Alignment)
            rospy.logwarn("Local alignment service available for mapmaker")

        rospy.logdebug("Subscibing to cameras")
        self.camera_topic = rospy.get_param("~camera_topic")
        self.cam_sub = rospy.Subscriber(self.camera_topic, Image, self.imageCB, queue_size=1)

        rospy.logwarn("Mapmaker started, awaiting goal")

    def miscCB(self, msg, args):
        if self.isMapping:
            topicName = args
            rospy.logdebug("Adding misc from %s" % (topicName))
            self.bag.write(topicName, msg) 

    def imageCB(self, msg):
        # save image on image shift
        self.img_msg = msg

        if self.visual_turn and self.last_img_msg is not None and self.isMapping:
            # create message
            srv_msg = SensorsInput()
            srv_msg.map_images = ImageList([self.last_img_msg])
            srv_msg.live_images = ImageList([self.img_msg])

            try:
                resp1 = self.local_align(srv_msg)
                hist = resp1.histograms[0].data
                half_size = np.size(hist)/2.0
                self.curr_trans = float(np.argmax(hist) - (np.size(hist)//2.0)) / half_size  # normalize -1 to 1
                if abs(self.curr_trans) > self.max_trans and self.last_saved_dist != self.lastDistance:
                    rospy.logdebug("Hit waypoint turn")
                    self.nextStep = self.lastDistance + self.mapStep
                    filename = os.path.join(self.mapName, str(self.lastDistance) + "_" + str(self.curr_trans) + ".jpg")
                    save_img(self.img_msg, filename)  # with resizing
                    self.last_img_msg = self.img_msg
            except Exception as e:
                rospy.logwarn("Service call failed: %s" % e)

        self.checkShutdown()

    def distanceCB(self, msg):
        # save image after traveled distance

        if self.isMapping == False or self.img_msg is None:
            return
        dist = msg.output
        self.lastDistance = dist
        if dist >= self.nextStep:
            self.last_saved_dist = dist
            if self.img_msg is None:
                rospy.logwarn("Warning: no image received!")
            rospy.logdebug("Hit waypoint distance")
            self.nextStep = self.lastDistance + self.mapStep
            filename = os.path.join(self.mapName, str(self.lastDistance) + "_" + str(self.curr_trans) + ".jpg")
            save_img(self.img_msg, filename)  # with resizing
            self.last_img_msg = self.img_msg
            # cv2.imwrite(filename, self.img)
            rospy.logwarn("Image saved %s" % (filename))

        self.checkShutdown()

    def joyCB(self, msg):
        if self.isMapping:
            rospy.logdebug("Adding joy")
            save_msg = DistancedTwist()
            save_msg.twist = msg
            save_msg.distance = self.lastDistance
            self.bag.write("recorded_actions", save_msg)

    def actionCB(self, goal):

        if self.img_msg is None:
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
            self.img_msg = None
            self.last_img_msg = None
            self.distance_reset_srv(0.0)
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
            self.nextStep = 0.0
            self.lastDistance = 0.0
            self.isMapping = True
        else:
            rospy.logdebug("Creating final wp")
            # filename = os.path.join(self.mapName, str(self.lastDistance) + ".jpg")
            # cv2.imwrite(filename, self.img)
            filename = os.path.join(self.mapName, str(self.lastDistance) + "_" + str(self.curr_trans) + ".jpg")
            save_img(self.img_msg, filename)  # with resizing
            rospy.logwarn("Stopping Mapping")
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

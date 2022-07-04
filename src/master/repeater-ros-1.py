#!/usr/bin/env python
import time
import rospy
import roslib
import os
import actionlib
import cv2
import rosbag
import threading
import queue
from sensor_msgs.msg import Image, Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from bearnav2.msg import MapRepeaterAction, MapRepeaterResult, SensorsInput, SensorsOutput, ImageList
from bearnav2.srv import SetDist, SetClockGain, SetClockGainResponse, Alignment
from cv_bridge import CvBridge
import numpy as np


BR = CvBridge()


def numpy_softmax(arr):
    tmp = np.exp(arr) / np.sum(np.exp(arr))
    return tmp


def load_map(mappath, images, distances, trans_hists):
    tmp = []
    for file in list(os.listdir(mappath)):
        if file.endswith(".jpg"):
            tmp.append(file[:-4].plit("_"))
    rospy.logwarn(str(len(tmp)) + " images found in the map")
    tmp.sort(key=lambda x: float(x[0]))
    for idx, dist_turn in enumerate(tmp):

        distances.append(dist_turn[0])
        images.append(BR.cv2_to_imgmsg(cv2.imread(os.path.join(mappath, dist_turn[0] + "_" + dist_turn[1] + ".jpg"))))
        rospy.loginfo("Loaded map image: " + dist_turn[0] + "_" + dist_turn[1] + str(".jpg"))
        if idx > 0:
            trans_hists.append(dist_turn[1])

    rospy.logwarn("Whole map sucessfully loaded")


class ActionServer():

    def __init__(self):

        #some vars
        self.br = CvBridge()
        self.img = None
        self.mapName = ""
        self.mapStep = None
        self.nextStep = 0
        self.bag = None
        self.isRepeating = False
        self.fileList = []
        self.endPosition = None
        self.clockGain = 1.0
        self.curr_dist = 0.0
        self.map_images = []
        self.map_distances = []
        self.action_dists = None
        self.actions = []
        # TODO: this is very important parameter - think about it!
        self.map_publish_span = 2
        self.map_transitions = []

        rospy.logdebug("Waiting for services to become available...")
        rospy.wait_for_service("repeat/set_dist")
        rospy.wait_for_service("repeat/set_align")
        rospy.Service('set_clock_gain', SetClockGain, self.setClockGain)

        rospy.logdebug("Resetting distance node")
        self.distance_reset_srv = rospy.ServiceProxy("repeat/set_dist", SetDist)
        self.align_reset_srv = rospy.ServiceProxy("repeat/set_align", SetDist)
        # self.distance_reset_srv(0.0)
        self.distance_sub = rospy.Subscriber("repeat/output_dist", SensorsOutput, self.distanceCB, queue_size=1)

        rospy.logdebug("Subscibing to cameras")
        self.camera_topic = rospy.get_param("~camera_topic")
        self.cam_sub = rospy.Subscriber(self.camera_topic, Image, self.pubSensorsInput, queue_size=1)
        rospy.logwarn(self.camera_topic)

        rospy.logdebug("Connecting to sensors module")
        self.sensors_pub = rospy.Publisher("sensors_input", SensorsInput, queue_size=1)

        rospy.logdebug("Setting up published for commands")
        self.joy_topic = "map_vel"
        self.joy_pub = rospy.Publisher(self.joy_topic, Twist, queue_size=1)

        rospy.logdebug("Starting repeater server")
        self.server = actionlib.SimpleActionServer("repeater", MapRepeaterAction, execute_cb=self.actionCB, auto_start=False)
        self.server.start()

        rospy.logwarn("Repeater started, awaiting goal")

    def setClockGain(self, req):
        self.clockGain = req.gain 
        return SetClockGainResponse()

    def pubSensorsInput(self, img_msg):
        self.img = img_msg
        if len(self.map_images) > 0:
            # rospy.logwarn(self.map_distances)
            nearest_map_idx = np.argmin(abs(self.curr_dist - np.array(self.map_distances)))
            lower_bound = max(0, nearest_map_idx - self.map_publish_span)
            upper_bound = min(nearest_map_idx + self.map_publish_span + 1, len(self.map_distances))
            map_imgs = ImageList(self.map_images[lower_bound:upper_bound])
            distances = self.map_distances[lower_bound:upper_bound]
            if len(self.map_transitions) > 0:
                transitions = self.map_transitions[lower_bound:upper_bound - 1]
            else:
                transitions = []
            live_imgs = ImageList([img_msg])
            # rospy.logwarn(distances)
            sns_in = SensorsInput()
            sns_in.header = img_msg.header
            sns_in.map_images = map_imgs
            sns_in.live_images = live_imgs
            sns_in.map_distances = distances
            sns_in.map_transitions = transitions
            self.sensors_pub.publish(sns_in)

            # DEBUGGING
            # self.debug_map_img.publish(self.map_images[nearest_map_idx])

    def distanceCB(self, msg):
        if self.isRepeating == False:
            return
        
        if self.img is None:
            rospy.logwarn("Warning: no image received")

        self.curr_dist = msg.output

        if self.endPosition != 0 and self.curr_dist >= self.endPosition:
            self.isRepeating = False

        # TODO: create a method here - replay distancewise
        self.play_closest_action()

        self.checkShutdown()

    def goalValid(self, goal):
        
        if goal.mapName == "":
            rospy.logwarn("Goal missing map name")
            return False
        if not os.path.isdir(goal.mapName):
            rospy.logwarn("Can't find map directory")
            return False
        if not os.path.isfile(os.path.join(goal.mapName, goal.mapName + ".bag")):
            rospy.logwarn("Can't find commands")
            return False
        if not os.path.isfile(os.path.join(goal.mapName, "params")):
            rospy.logwarn("Can't find params")
            return False
        if goal.startPos < 0:
            rospy.logwarn("Invalid (negative) start position). Use zero to start at the beginning") 
            return False
        if goal.startPos > goal.endPos:
            rospy.logwarn("Start position greater than end position")
            return False
        return True

    def actionCB(self, goal):

        rospy.loginfo("New goal received")
        
        if self.goalValid(goal) == False:
            rospy.logwarn("Ignoring invalid goal")
            result = MapRepeaterResult()
            result.success = False
            self.server.set_succeeded(result)
            return

        self.parseParams(os.path.join(goal.mapName, "params"))

        map_loader = threading.Thread(target=load_map, args=(goal.mapName, self.map_images, self.map_distances,
                                                             self.map_transitions))
        map_loader.start()

        #set distance to zero
        rospy.logdebug("Resetting distnace and alignment")
        self.distance_reset_srv(goal.startPos)
        self.align_reset_srv(0.0)
        self.endPosition = goal.endPos
        self.nextStep = 0

        rospy.logwarn("Starting repeat")
        self.bag = rosbag.Bag(os.path.join(goal.mapName, goal.mapName + ".bag"), "r")
        self.mapName = goal.mapName
    
        #create publishers
        additionalPublishers = {}
        rospy.logwarn(self.savedOdomTopic)
        for topic, message, ts in self.bag.read_messages():
            if topic is not self.savedOdomTopic:
                topicType = self.bag.get_type_and_topic_info()[1][topic][0]
                topicType = roslib.message.get_message_class(topicType)
                additionalPublishers[topic] = rospy.Publisher(topic, topicType, queue_size=1) 

        time.sleep(2)       # waiting till some map images are parsed

        self.parse_rosbag()
        self.isRepeating = True
        # kick into the robot at the beggining:
        self.play_closest_action()
        # self.replay_timewise(additionalPublishers)    # for timewise repeating

        rospy.loginfo("Goal Complete!")
        self.shutdown()
        result = MapRepeaterResult()
        result.success = True
        self.server.set_succeeded(result)
         
    def parseParams(self, filename):

        with open(filename, "r") as f:
            data = f.read()
        data = data.split("\n")
        data = filter(None, data)
        for line in data:
            line = line.split(" ")
            if "stepSize" in line[0]:
                rospy.logdebug("Setting step size to: %s" % (line[1]))
                self.mapStep = float(line[1])
            if "odomTopic" in line[0]:
                rospy.logdebug("Saved odometry topic is: %s" % (line[1]))
                self.savedOdomTopic = line[1] 

    def checkShutdown(self):
        if self.server.is_preempt_requested():
            self.shutdown()

    def shutdown(self):
        self.isRepeating = False
        if self.bag is not None:
            self.bag.close()

    def replay_timewise(self, additionalPublishers):
        #replay bag
        rospy.logwarn("Starting")
        previousMessageTime = None
        expectedMessageTime = None
        start = rospy.Time.now()
        for topic, message, ts in self.bag.read_messages():
            #rosbag virtual clock
            now = rospy.Time.now()
            if previousMessageTime is None:
                previousMessageTime = ts
                expectedMessageTime = now
            else:
                simulatedTimeToGo = ts - previousMessageTime
                correctedSimulatedTimeToGo = simulatedTimeToGo * self.clockGain
                error = now - expectedMessageTime
                sleepTime = correctedSimulatedTimeToGo - error
                expectedMessageTime = now + sleepTime
                rospy.sleep(sleepTime)
                previousMessageTime = ts
            #publish
            if topic == self.savedOdomTopic:
                self.joy_pub.publish(message.twist)
            else:
                additionalPublishers[topic].publish(message)
            msgBuf = (topic, message)
            if self.isRepeating == False:
                rospy.loginfo("stopped!")
                break
            if rospy.is_shutdown():
                rospy.loginfo("Node Shutdown")
                result = MapRepeaterResult()
                result.success = False
                self.server.set_succeeded(result)
                return
        self.isRepeating = False
        end = rospy.Time.now()
        dur = end - start
        rospy.logwarn("Rosbag runtime: %f" % (dur.to_sec()))

    def parse_rosbag(self):
        rospy.logwarn("Starting to parse the actions")
        self.action_dists = []
        for topic, msg, t in self.bag.read_messages(topics=[self.savedOdomTopic]):
            self.action_dists.append(float(msg.distance))
            self.actions.append(msg.twist)
        self.action_dists = np.array(self.action_dists)
        rospy.logwarn("Actions and distances successfully laoded!")

    def play_closest_action(self):
        # TODO: Does not support additional topics
        distance_to_pos = abs(self.curr_dist - self.action_dists)
        closest_idx = np.argmin(distance_to_pos)
        if self.isRepeating:
            self.joy_pub.publish(self.actions[closest_idx])


if __name__ == '__main__':

    rospy.init_node("replayer_server")
    server = ActionServer()
    rospy.spin()
    server.shutdown()

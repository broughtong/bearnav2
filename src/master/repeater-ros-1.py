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
from bearnav2.srv import SetDist, SetClockGain, SetClockGainResponse
from cv_bridge import CvBridge
import numpy as np


BR = CvBridge()


def load_map(mappath, images, distances, trans_hists, nn_service):
    tmp = []
    for file in list(os.listdir(mappath)):
        if file.endswith(".jpg"):
            tmp.append(file[:-4])
    rospy.logwarn(str(len(tmp)) + " images found in the map")
    tmp.sort(key=float)
    for idx, dist in enumerate(tmp):
        distances.append(float(dist))
        images.append(BR.cv2_to_imgmsg(cv2.imread(os.path.join(mappath, dist + ".jpg"))))
        """
        # this crazy part is for estimating the transitions
        if len(images) >= 2 and nn_service is not None:
            msg1 = ImageList([images[-2]])
            msg2 = ImageList([images[-1]])
            try:
                resp1 = nn_service(msg1, msg2)
                trans_hists.append(numpy_softmax(resp1.histograms[0].data))
                rospy.logwarn("transition between images is " + str(np.argmax(trans_hists[-1]) - np.size(trans_hists[-1])//2))
            except rospy.ServiceException as e:
                rospy.logwarn("Service call failed: %s" % e)
        """
        rospy.loginfo("Loaded map image: " + str(dist) + str(".jpg"))
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
        self.map_publish_span = 0

        rospy.logdebug("Waiting for services to become available...")
        rospy.wait_for_service("set_dist")
        rospy.wait_for_service("set_align")
        rospy.Service('set_clock_gain', SetClockGain, self.setClockGain)

        rospy.logdebug("Resetting distance node")
        self.distance_reset_srv = rospy.ServiceProxy("set_dist", SetDist)
        self.align_reset_srv = rospy.ServiceProxy("set_align", SetDist)
        self.distance_reset_srv(0.0)
        self.distance_sub = rospy.Subscriber("correction_cmd", SensorsOutput, self.distanceCB, queue_size=1)

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
        rospy.loginfo("Server started, awaiting goal")

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
            # transitions = self.map_transitions[lower_bound:upper_bound - 1]
            live_imgs = ImageList([img_msg])
            # rospy.logwarn(distances)
            sns_in = SensorsInput()
            sns_in.header = img_msg.header
            sns_in.map_images = map_imgs
            sns_in.live_images = live_imgs
            sns_in.distances = distances
            # sns_in.transitions = transitions
            self.sensors_pub.publish(sns_in)

            # DEBUGGING
            # self.debug_map_img.publish(self.map_images[nearest_map_idx])

    def distanceCB(self, msg):
        if self.isRepeating == False:
            return
        
        if self.img is None:
            rospy.logwarn("Warning: no image received")

        self.curr_dist = msg.distance

        if self.endPosition != 0 and self.curr_dist >= self.endPosition:
            self.isRepeating = False

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
        map_loader = threading.Thread(target=load_map, args=(goal.mapName, self.map_images, self.map_distances, None, None))
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
        for topic, message, ts in self.bag.read_messages():
            if topic is not self.savedOdomTopic:
                topicType = self.bag.get_type_and_topic_info()[1][topic][0]
                topicType = roslib.message.get_message_class(topicType)
                additionalPublishers[topic] = rospy.Publisher(topic, topicType, queue_size=1) 


        time.sleep(2)
        #replay bag
        rospy.logwarn("Starting")
        previousMessageTime = None
        expectedMessageTime = None
        self.isRepeating = True
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
                self.joy_pub.publish(message)
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

        rospy.loginfo("Goal Complete!")
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


if __name__ == '__main__':

    rospy.init_node("replayer_server")
    server = ActionServer()
    rospy.spin()
    server.shutdown()

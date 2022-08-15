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
from bearnav2.msg import MapRepeaterAction, MapRepeaterResult, Alignment
from bearnav2.srv import SetDist, SetClockGain, SetClockGainResponse
from cv_bridge import CvBridge
import numpy as np

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

        rospy.logdebug("Waiting for services to become available...")
        rospy.wait_for_service("set_dist")
        rospy.Service('set_clock_gain', SetClockGain, self.setClockGain)

        rospy.logdebug("Resetting distance node")
        self.distance_reset_srv = rospy.ServiceProxy("set_dist", SetDist)
        self.distance_reset_srv(0)
        self.distance_sub = rospy.Subscriber("distance", Float64, self.distanceCB, queue_size=1)

        rospy.logdebug("Subscibing to cameras")
        self.camera_topic = rospy.get_param("~camera_topic")
        self.cam_sub = rospy.Subscriber(self.camera_topic, Image, self.imageCB, queue_size=1, buff_size=20000000)

        rospy.logdebug("Connecting to alignment module")
        self.al_sub = rospy.Subscriber("alignment/output", Alignment, self.alignCB)
        self.al_1_pub = rospy.Publisher("alignment/inputCurrent", Image, queue_size=1)
        self.al_2_pub = rospy.Publisher("alignment/inputMap", Image, queue_size=1)
        self.al_pub = rospy.Publisher("correction_cmd", Alignment, queue_size=1)

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

    def imageCB(self, msg):

        if self.isRepeating:
            self.al_1_pub.publish(msg)
            self.img = True
            self.checkShutdown()

    def getClosestImg(self, dist):
        
        if len(self.fileList) < 1:
            rospy.logwarn("Not many map files")

        closestFilename = None
        closestDistance = 999999
        dist = float(dist)
        for filename in self.fileList:
            ffilename = float(filename)
            diff = abs(ffilename - dist)
            #rospy.logwarn("diff %f" % (diff))
            if diff < closestDistance:
                #rospy.logwarn("Better fit")
                closestDistance = diff
                closestFilename = filename

        fn = os.path.join(self.mapName, closestFilename + ".jpg")
        #rospy.logwarn("Opening file: %s dist %f" % (fn, dist))
        img = cv2.imread(fn)
        msg = self.br.cv2_to_imgmsg(img)
        self.al_2_pub.publish(msg)

    def distanceCB(self, msg):
        
        if self.isRepeating == False:
            return
        
        if self.img is None:
            rospy.logwarn("Warning: no image received")

        dist = msg.data
        self.getClosestImg(dist)

        #if dist >= self.nextStep:
        #    rospy.logdebug("Triggered wp")
        #    self.nextStep += self.mapStep

        if self.endPosition != 0 and dist >= self.endPosition:
            self.isRepeating = False

        self.checkShutdown()

    def alignCB(self, msg):
        self.al_pub.publish(msg)

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
        
        #get file list
        self.fileList = []
        allFiles = []
        for files in os.walk(goal.mapName):
            allFiles = files[2]
            break
        for filename in allFiles:
            if ".jpg" in filename:
                filename = ".".join(filename.split(".")[:-1])
                self.fileList.append(filename)
        rospy.logwarn("Found %i map files" % (len(self.fileList)))

        #set distance to zero
        rospy.logdebug("Resetting distnace")
        self.distance_reset_srv(goal.startPos)
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

class ImageFetcherThread(threading.Thread):
    def __init__(self, location, idQueue, imgLock):
        threading.Thread.__init__(self)
        self.location = location
        self.idQueue = idQueue
        self.imgLock = imgLock
    def run(self):
        threadLock.acquire()
        threadLock.release()

if __name__ == '__main__':

    rospy.init_node("replayer_server")
    server = ActionServer()
    rospy.spin()
    server.shutdown()

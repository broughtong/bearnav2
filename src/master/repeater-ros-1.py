#!/usr/bin/env python
import time
import rospy
import os
import actionlib
import cv2
import rosbag
import threading
import queue
from sensor_msgs.msg import Image, Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Header, Float32
from bearnav2.msg import MapRepeaterAction, MapRepeaterResult, Alignment, PFInput, ImageList
from bearnav2.srv import SetDist
from cv_bridge import CvBridge
import numpy as np
import threading


BR = CvBridge()


def load_map(mappath, images, distances):
    tmp = []
    for file in list(os.listdir(mappath)):
        if file.endswith(".jpg"):
            tmp.append(file[:-4])
    rospy.logwarn(str(len(tmp)) + " images found in the map")
    tmp.sort(key=float)
    for idx, dist in enumerate(tmp):
        distances.append(float(dist))
        images.append(BR.cv2_to_imgmsg(cv2.imread(os.path.join(mappath, dist + ".jpg"))))
        rospy.loginfo("Loaded map image: " + str(dist) + str(".jpg"))
    rospy.logwarn("Whole map sucessfully loaded")


class ActionServer():

    def __init__(self):

        #some vars
        self.pf_delay = 6
        self.pf_counter = 0
        self.pf_span = 2

        self.img = None
        self.mapName = ""
        self.mapStep = None
        self.nextStep = 0
        self.bag = None
        self.isRepeating = False
        self.map_images = []
        self.map_distances = []
        self.endPosition = None

        self.curr_dist = 0

        rospy.logdebug("Waiting for services to become available...")
        rospy.wait_for_service("set_dist")

        rospy.logdebug("Resetting distance node")
        self.distance_reset_srv = rospy.ServiceProxy("set_dist", SetDist)
        self.distance_reset_srv(0)
        self.distance_sub = rospy.Subscriber("distance", Float32, self.distanceCB, queue_size=1)

        rospy.logdebug("Subscibing to cameras")
        self.camera_topic = rospy.get_param("~camera_topic")
        self.cam_sub = rospy.Subscriber(self.camera_topic, Image, self.imageCB, queue_size=1)

        rospy.logdebug("Connecting to alignment module")
        self.al_sub = rospy.Subscriber("alignment/output", Alignment, self.alignCB)
        self.al_1_pub = rospy.Publisher("alignment/inputA", Image, queue_size=1)
        self.al_2_pub = rospy.Publisher("alignment/inputB", Image, queue_size=1)
        self.al_pub = rospy.Publisher("correction_cmd", Alignment, queue_size=1)
        self.pf_pub = rospy.Publisher("pf_img_input", PFInput, queue_size=1)

        rospy.logdebug("Setting up published for commands")
        self.joy_topic = "map_vel"
        self.joy_pub = rospy.Publisher(self.joy_topic, Twist, queue_size=0)

        rospy.logdebug("Starting repeater server")
        self.server = actionlib.SimpleActionServer("repeater", MapRepeaterAction, execute_cb=self.actionCB, auto_start=False)
        self.server.start()
        rospy.loginfo("Server started, awaiting goal")

    def imageCB(self, msg):
        if self.isRepeating:
            self.pf_counter += 1
            self.al_1_pub.publish(msg)
            self.img = True
            self.checkShutdown()
            if not self.pf_counter % self.pf_delay:
                self.pubClosestImgList(msg)

    def pubClosestImg(self):
        if len(self.map_images) < 1:
            rospy.logwarn("Repeater - No map files")
            return

        nearest_map_idx = np.argmin(abs(self.curr_dist - np.array(self.map_distances)))

        if self.isRepeating:
            msg = self.map_images[nearest_map_idx]
            self.al_2_pub.publish(msg)

    def pubClosestImgList(self, img_msg):
        if len(self.map_images) > 0:
            # rospy.logwarn(self.map_distances)
            nearest_map_idx = np.argmin(abs(self.curr_dist - np.array(self.map_distances)))
            lower_bound = max(0, nearest_map_idx - self.pf_span)
            upper_bound = min(nearest_map_idx + self.pf_span + 1, len(self.map_distances))
            map_imgs = ImageList(self.map_images[lower_bound:upper_bound])
            distances = self.map_distances[lower_bound:upper_bound]
            live_imgs = ImageList([img_msg])
            # rospy.logwarn(distances)
            pf_msg = PFInput()
            pf_msg.map_images = map_imgs
            pf_msg.live_images = live_imgs
            pf_msg.distances = distances
            self.pf_pub.publish(pf_msg)

    def distanceCB(self, msg):
        dist = msg.data
        if self.isRepeating == False:
            return
        if self.img is None:
            rospy.logwarn("Warning: no image received")
        rospy.logdebug("Triggered wp")
        # if dist > self.curr_dist: maybe this condition is not bad!
        self.curr_dist = dist
        self.pubClosestImg()

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

        rospy.logwarn("New goal received")
        
        if self.goalValid(goal) == False:
            rospy.logwarn("Ignoring invalid goal")
            result = MapRepeaterResult()
            result.success = False
            self.server.set_succeeded(result)
            return

        self.parseParams(os.path.join(goal.mapName, "params"))

        map_loader = threading.Thread(target=load_map, args=(goal.mapName, self.map_images, self.map_distances))
        map_loader.start()

        #set distance to zero
        rospy.logdebug("Resetting distnace")
        self.distance_reset_srv(goal.startPos)
        self.curr_dist = goal.startPos
        self.endPosition = goal.endPos
        self.nextStep = 0

        rospy.logwarn("Starting repeat")
        self.bag = rosbag.Bag(os.path.join(goal.mapName, goal.mapName + ".bag"), "r")
        self.mapName = goal.mapName

        #replay bag
        start = None
        sim_start = None
        self.isRepeating = True
        rospy.logwarn("Starting")
        for topic, message, ts in self.bag.read_messages():
            now = rospy.Time.now()
            if sim_start is None:
                start = now
                sim_start = ts
            else:
                real_time = now - start
                sim_time = ts - sim_start
                if sim_time > real_time:
                    rospy.sleep(sim_time - real_time)
            self.joy_pub.publish(message)
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

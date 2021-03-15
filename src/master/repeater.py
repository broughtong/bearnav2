#!/usr/bin/env python
import rospy
import os
import actionlib
import cv2
import rosbag
import threading
import queue
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
        self.mapStep = None
        self.nextStep = 0
        self.bag = None
        self.isRepeating = False

        print("Waiting for services to become available...")
        rospy.wait_for_service("set_dist")

        print("Resetting distance node")
        self.distance_reset_srv = rospy.ServiceProxy("set_dist", SetDist)
        self.distance_reset_srv(0)
        self.distance_sub = rospy.Subscriber("/distance", Float64, self.distanceCB)

        print("Subscibing to cameras")
        self.cam_sub = rospy.Subscriber("/camera_2/image_rect_color", Image, self.imageCB)

        print("Connecting to alignment module")
        self.al_sub = rospy.Subscriber("/alignment/output", Image, self.alignCB)
        self.al_1_pub = rospy.Publisher("/alignment/inputA", Image, queue_size=0)
        self.al_2_pub = rospy.Publisher("/alignment/inputB", Image, queue_size=0)

        print("Setting up published for commands")
        self.joy_topic = "cmd_vel"
        self.joy_pub = rospy.Publisher(self.joy_topic, Twist, queue_size=0)

        print("Starting repeater server")
        self.server = actionlib.SimpleActionServer("repeater", MapRepeaterAction, execute_cb=self.actionCB, auto_start=False)
        self.server.start()
        print("Server started, awaiting goal")

    def imageCB(self, msg):

        #self.img = self.br.imgmsg_to_cv2(msg)
        self.al_1_pub.publish(msg)
        self.checkShutdown()

    def getImg(self, dist):
        
        fn = os.path.join(self.mapName, str(int(dist)) + ".jpg")
        print("Opening : " + fn)
        img = cv2.imread(fn)
        msg = self.br.cv2_to_imgmsg(img)
        self.al_2_pub.publish(msg)

    def distanceCB(self, msg):

        dist = msg.data
        if dist >= self.nextStep:
            print("Triggered wp")
            self.getImg(self.nextStep)
            filename = os.path.join(self.mapName, str(int(self.nextStep)) + ".jpg")
            self.nextStep += self.mapStep
            cv2.imwrite(filename, self.img)

        self.checkShutdown()

    def alignCB(self, msg):

        print("Master says:")
        print(msg)

    def actionCB(self, goal):

        print(goal)

        if goal.mapName == "":
            print("Missing mapname")

        if not os.path.isdir(goal.mapName):
            print("Can't find map directory")
        if not os.path.isfile(os.path.join(goal.mapName, goal.mapName + ".bag")):
            print("Can't find commands")
        if not os.path.isfile(os.path.join(goal.mapName, "params")):
            print("Can't find params")

        self.parseParams(os.path.join(goal.mapName, "params"))

        print("Starting repeat")
        self.bag = rosbag.Bag(os.path.join(goal.mapName, goal.mapName + ".bag"), "r")
        self.mapName = goal.mapName

        #set distance to zero
        self.distance_reset_srv(0)

        #replay bag
        start = rospy.Time.now()
        sim_start = None
        print("Starting repeat")
        self.isRepeating = True
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
        self.isRepeating = False
        print("Complete")
         
    def parseParams(self, filename):

        with open(filename, "r") as f:
            data = f.read()
        data = data.split("\n")
        data = filter(None, data)
        for line in data:
            line = line.split(" ")
            if "stepSize" in line[0]:
                print("Setting step size to: %s" % (line[1]))
                self.mapStep = int(line[1])

    def checkShutdown(self):
        if self.server.is_preempt_requested():
            self.shutdown()

    def shutdown(self):
        self.isRepeating = False
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

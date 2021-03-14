#!/usr/bin/env python
import rospy
import actionlib
from std_msgs.msg import Float64
from bearnav2.msg import MapMakerAction, MapMakerFeedback
from bearnav2.srv import SetDist

class ActionServer():
    #_feedback = bearnav2.msg.MapMakerFeedback()
    #_result = bearnav2.msg.MapMakerResult()

    def __init__(self):

        print("Waiting for services to become available...")
        rospy.wait_for_service("set_dist")


        print("Starting mapmaker server")
        self.server = actionlib.SimpleActionServer("mapmaker", MapMakerAction, execute_cb=self.actionCB, auto_start=False)
        self.server.start()

        print("Resetting distance node")
        self.distance_reset_srv = rospy.ServiceProxy("set_dist", SetDist)
        self.distance_reset_srv(0)
        self.distance_sub = rospy.Subscriber("/distance", Float64, self.distanceCB)

    def distanceCB(self, msg):

        

    def actionCB(self, goal):

        print(goal)

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

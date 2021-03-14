#!/usr/bin/env python
import rospy
import actionlib
import bearnav 
import bearnav.msg 
from bearnav.srv import SetDist

class BearnavAction(object):
    _feedback = bearnav.msg.BearnavFeedback()
    _result = bearnav.msg.BearnavResult()

    def __init__(self, name):
        rospy.wait_for_service("set_dist")

        self.name = name
        self.server = actionlib.SimpleActionServer(name, bearnav.msg.BearnavAction, execute_cb=self.callback, auto_start=True)
        self.distance_sub = rospy.

        #distance server
        self.distance_reset_srv = rospy.ServiceProxy("set_dist", SetDist)


    def callback(self):

         

        
        
    


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
        
if __name__ == '__main__':
    rospy.init_node("mapmaker_server")
    server = BearnavAction(rospy.get_name())
    rospy.spin()

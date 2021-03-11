#!/usr/bin/env python3
import rospy
class Distance:
    def __init__(self,use_twist):
        self.driven_dist = 0
        self.use_twist = use_twist #TODO  make parameter
        print(use_twist)
        self.last_odom = None
        self.last_time = None
    def set(self,dst):
        self.driven_dist = dst.dist
        self.last_odom = None
        self.last_time = None
        return self.driven_dist

    def drive(self, dx,dy,dz):
        self.driven_dist = self.driven_dist + (dx**2+dy**2+dz**2)**(1/2)
       
    def processT(self, msg):
        now = rospy.get_time()
        if not self.use_twist:
            return None, False
        if self.last_time is None:
            self.last_time = now
            return None, False
        dt = now - self.last_time
        self.last_time = now
        dx = dt * msg.linear.x
        dy = dt * msg.linear.y
        dz = dt * msg.linear.z
        self.drive(dx,dy,dz)
        return self.driven_dist, True

    def processO(self,msg):
        if self.use_twist:
            return None, False
        if self.last_odom is None:
            self.last_odom = msg
            return None, False
        dx = self.last_odom.pose.pose.position.x - msg.pose.pose.position.x
        dy = self.last_odom.pose.pose.position.y - msg.pose.pose.position.y
        dz = self.last_odom.pose.pose.position.z - msg.pose.pose.position.z
        self.last_odom = msg
        self.drive(dx,dy,dz)
        return self.driven_dist, True
        


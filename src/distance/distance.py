#!/usr/bin/env python

class Distance:
    def __init__(self,use_twist):
        self.driven_dist = 0
        self.have_odom = use_twist #TODO  make parameter
        self.last_odom = None
        self.last_time = None
    def set(self,dst):
        self.driven_dist = dst.dist

    def drive(dx,dy,dz):
        self.driven_dist = self.driven_dist + (dx**2+dy**2+dz**2)**(1/2)
       
    def processT(self, msg):
        now = rospy.get_time()
        if self.have_odom:
            return None, False
        if self.last_time is None:
            self.last_time = now
            return None, False
        dt = now - self.last_time
        dx = dt * msg.linear.x
        dy = dt * msg.linear.y
        dz = dt * msg.linear.z 
        drive(dx,dy,dz)
        return self.driven_dist, True

    def processO(self,msg):
        if not self.have_odom:
            return None, False
        if self.last_odom is None:
            self.last_odom = msg
            return None, False
        dx = self.last_odom.pose.pose.position.x - msg.pose.pose.position.x
        dy = self.last_odom.pose.pose.position.y - msg.pose.pose.position.y
        dz = self.last_odom.pose.pose.position.z - msg.pose.pose.position.z
        drive(dx,dy,dz)
        return self.driven_dist, True
        


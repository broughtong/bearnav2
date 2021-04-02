#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from bearnav2.msg import IntList
import matplotlib.pyplot as plt
import numpy as np

pub = None
br = CvBridge()

def callback(msg):

    if pub.get_num_connections() == 0:
        return
    
    plt.clf()
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(msg.data)
    fig.canvas.draw()

    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    msg = br.cv2_to_imgmsg(img, encoding="rgb8")
    pub.publish(msg)

if __name__ == "__main__":

    rospy.init_node("histogram_viz")
    pub = rospy.Publisher("/histogram_viz", Image, queue_size=0)
    rospy.Subscriber("/histogram", IntList, callback)
    print("Histogram viz ready...")
    rospy.spin()

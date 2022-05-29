#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from bearnav2.msg import FloatList
import matplotlib.pyplot as plt
import numpy as np

pub = None
br = CvBridge()

def callback(msg):
    if pub.get_num_connections() == 0:
        return

    print("particles received")

    plt.clf()
    fig = plt.figure()
    ax = plt.axes()
    ax.title.set_text("Particle filter")
    msg_size = len(msg.data)
    distances = msg.data[:msg_size//2]
    displacements = msg.data[msg_size//2:]
    ax.plot(displacements, distances, "o")
    ax.set_xlim([-255, 255])
    curr_window = np.mean(distances)//3
    ax.set_ylim([3*curr_window, 3*(curr_window+1)])
    ax.grid()
    fig.canvas.draw()

    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    msg = br.cv2_to_imgmsg(img, encoding="rgb8")
    pub.publish(msg)

if __name__ == "__main__":

    rospy.init_node("pf_viz")
    pub = rospy.Publisher("pf_viz", Image, queue_size=0)
    rospy.Subscriber("/bearnav2/particles", FloatList, callback)
    print("PF viz ready...")
    rospy.spin()

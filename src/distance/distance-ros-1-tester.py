import rospy
import distance_pf
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from bearnav2.srv import SetDist, SetDistResponse
from bearnav2.msg import ImageList, PFInput
import os
import numpy as np
import cv2
from cv_bridge import CvBridge

br = CvBridge()

def get_map(path):
    distance_list = []
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            distance_list.append((file[:-4]))
    distance_list.sort(key=float)
    return np.array(distance_list).astype(float), distance_list   # floats and string sorted

def get_closest_five(curr_dist, map_path):
    closest_idx = np.argmin(abs(curr_dist - recorded_map))
    lower_bound = max(0, closest_idx - 2)
    upper_bound = min(closest_idx + 3, len(recorded_map) - 1)  # IS -1 here???
    imgs = []
    for my_idx in np.arange(lower_bound, upper_bound):
        img = cv2.imread(os.path.join(map_path, string_map[my_idx] + ".jpg"))
        imgs.append(br.cv2_to_imgmsg(img))
    ret = ImageList(imgs)
    return ret, recorded_map[lower_bound:upper_bound]

def callbackOdom(msg):
    global distance
    distance, use = d.processO(msg)
    if use:
        print("Using odometry")
        pub.publish(distance)

def callbackCamera(msg):
    global distance
    imgs, distances = get_closest_five(distance, map_path)
    msg_to_pass = PFInput(imgs, ImageList([msg]), distances)
    distance, use = d.processS(msg_to_pass)
    if use:
        print("Using camera")
        pub.publish(distance)

def handle_set_dist(dst):
    driven = d.set(dst)
    print("Distance set to " + str(driven))
    pub.publish(driven)
    return SetDistResponse()

if __name__ == "__main__":
    distance = 0
    print("Node started")
    rospy.init_node("distance")
    s = rospy.Service('set_dist', SetDist, handle_set_dist)
    map_path = "/home/zdeeno/Documents/Datasets/kn/main5"
    recorded_map, string_map = get_map(map_path)
    print("Map:", recorded_map)
    pub = rospy.Publisher("distance", Float64, queue_size=1)
    rospy.Subscriber("/imu_and_wheel_odom", Odometry, callbackOdom, queue_size=1)
    rospy.Subscriber("/camera_front/image_color", Image, callbackCamera, queue_size=1)
    d = distance_pf.DistancePF(use_twist=False)
    print("Particle filter initialized")
    rospy.spin()

    # rospy.Subscriber(cmd_vel_topic, Twist, callbackTwist)

    # rospy.init_node("distance")
    # use_twist = rospy.get_param("use_twist",'False')
    # cmd_vel_topic = rospy.get_param("~cmd_vel_topic")
    # odom_topic = rospy.get_param("~odom_topic")
    # pub = rospy.Publisher("distance", Float64, queue_size=0)

    # # rospy.Subscriber("map_images", ImageList, callbackCamera)  # TODO: make name of topic as argument
    # s = rospy.Service('set_dist', SetDist, handle_set_dist)
    # rospy.spin()

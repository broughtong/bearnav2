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
IMG_SPAN = 2
pub_counter = 0
last_closest_idx = 0
last_photos = None

def get_map(path):
    distance_list = []
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            distance_list.append((file[:-4]))
    distance_list.sort(key=float)
    return np.array(distance_list).astype(float), distance_list   # floats and string sorted

def get_closest_photos(curr_dist, map_path):
    global last_closest_idx, last_photos
    closest_idx = np.argmin(abs(curr_dist - recorded_map))
    last_diff = closest_idx - last_closest_idx
    if abs(last_diff) > 1 or last_photos is None:   # the images are significantly shifted filling buffer
        lower_bound = max(0, closest_idx - IMG_SPAN)
        upper_bound = min(closest_idx + IMG_SPAN + 1, len(recorded_map))
        imgs = []
        for my_idx in np.arange(lower_bound, upper_bound):
            # print("Loading img:", string_map[my_idx] + ".jpg", "with idx", my_idx)
            img = cv2.imread(os.path.join(map_path, string_map[my_idx] + ".jpg"))
            imgs.append(br.cv2_to_imgmsg(img))
        last_photos = imgs
        ret = ImageList(imgs)
        last_closest_idx = closest_idx
        return ret, recorded_map[lower_bound:upper_bound]
    else:   # use already buffered images
        lower_bound = max(0, closest_idx - IMG_SPAN)
        upper_bound = min(closest_idx + IMG_SPAN, len(recorded_map))
        if last_diff == 0:
            return ImageList(last_photos), recorded_map[lower_bound:upper_bound + 1]
        if last_diff == 1 and upper_bound < len(recorded_map):
            img = cv2.imread(os.path.join(map_path, string_map[upper_bound] + ".jpg"))
            last_photos.append(br.cv2_to_imgmsg(img))
            if len(last_photos) > 2*IMG_SPAN + 1:
                last_photos.pop(0)
        if last_diff == -1 and lower_bound >= 0:
            img = cv2.imread(os.path.join(map_path, string_map[lower_bound] + ".jpg"))
            last_photos.insert(0, br.cv2_to_imgmsg(img))
            if len(last_photos) > 2*IMG_SPAN + 1:
                last_photos.pop()
        last_closest_idx = closest_idx
        return ImageList(last_photos), recorded_map[lower_bound:upper_bound + 1]


def callbackOdom(msg):
    global distance
    distance, use = d.processO(msg)
    if use:
        # print("Trying to use odometry")
        pub.publish(distance)

def callbackCamera(msg):
    global distance, pub_counter
    pub_counter += 1
    if not pub_counter % 15:
        imgs, distances = get_closest_photos(distance, map_path)
        msg_to_pass = PFInput(imgs, ImageList([msg]), distances)
        distance, use = d.processS(msg_to_pass)
        if use:
            # print("Trying to use camera")
            pub.publish(distance)
    # else:
    #     print("Throwing away obtained img.")

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
    rospy.Subscriber("/husky_velocity_controller/odom", Odometry, callbackOdom, queue_size=1)
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

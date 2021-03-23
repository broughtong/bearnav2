#!/usr/bin/env python
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class PreprocessNode(Node):
	
	def __init__(self):
		super().__init__("preprocessor")
		self.subscriber = self.create_subscription(Image, "/preprocess/input", self.callback, 0)
		self.subscriber
		self.br = CvBridge()
		self.p = preprocess.Preprocessor("./config.yaml")

	def callback(self, msg):

		img = self.br.imgmsg_to_cv2(msg)
		img = self.p.process(img)
		msg = self.br.cv2_to_imgmsg(img)
		self.publisher.publish(msg)

if __name__ == "__main__":

	rclpy.init()

	node = PreprocessNode()
	rclpy.spin(node)

	node.destroy_node()
	rclpy.shutdown()

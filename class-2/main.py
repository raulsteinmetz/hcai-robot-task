#!/usr/bin/env python3

"""
    Get bottle from camera node,
    go towards it using only camera information.
"""

import math
import rospy
import tf
import tf.transformations
from sensor_msgs.msg import Joy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose2D
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from yolo_ros_msgs.msg import BoundingBoxes


class CrashBottlesNode(object):
    def __init__(self):
        rospy.init_node('crash_bottles')

        self.odom = Pose2D()
        self.max_linear_vel = 0.15
        self.max_angular_vel = 0.3

        self.image_width = 640

        # reads bounding boxes from yolo node
        self.yolo_sub = rospy.Subscriber(
            'yolov8/detection/bounding_boxes', BoundingBoxes, self.yoloCallback, queue_size=1)

        # publishes velocity commands
        self.cmd_vel_pub = rospy.Publisher(
            'ypspur_ros/cmd_vel', Twist, queue_size=1)

        self.timer = rospy.Timer(
            rospy.Duration(0.05), self.mainLoop)  # call control loop every 0.05s (20Hz)

        # control variables
        self.bottle_detected = False
        self.bottle_percentage_x = None  # image position of bottle in percentage
        self.bottle_detected_since = 1000

        rospy.loginfo("Ready to start. -------------")
        rospy.spin()

    def yoloCallback(self, in_msg):
        # try to detect bottle
        for i in range(len(in_msg.bounding_boxes)):
            if in_msg.bounding_boxes[i].class_name == "bottle":
                self.bottle_detected = True
                self.bottle_detected_since = 0
                # bounding box information
                u_min = in_msg.bounding_boxes[i].u_min
                v_min = in_msg.bounding_boxes[i].v_min
                u_max = in_msg.bounding_boxes[i].u_max
                v_max = in_msg.bounding_boxes[i].v_max

                # calculate central pixel of the bb
                center_x = (u_min + u_max) / 2.0
                center_y = (v_min + v_max) / 2.0

                # calculate percentage position in the image
                self.bottle_percentage_x = (center_x / self.image_width) * 100
                print(f"Detected a bottle!")
                print(f"Center of the bottle (in percentage): X = {self.bottle_percentage_x}%, Y = {center_y}")
                return

        self.bottle_detected_since += 1

        if self.bottle_detected_since > 5:
            self.bottle_detected = False
            

    def mainLoop(self, in_event):
        cmd_vel = Twist()

        if self.bottle_detected:
            if 45 <= self.bottle_percentage_x <= 55:
                # bottle is near the center of the image (around 50%), go forward
                cmd_vel.linear.x = self.max_linear_vel
                cmd_vel.angular.z = 0.0
            elif self.bottle_percentage_x < 45:
                # bottle is to the left, turn left
                cmd_vel.linear.x = self.max_linear_vel
                cmd_vel.angular.z = self.max_angular_vel
            elif self.bottle_percentage_x > 55:
                # bottle is to the right, turn right
                cmd_vel.linear.x = self.max_linear_vel
                cmd_vel.angular.z = -self.max_angular_vel
        else:
            # no bottle is detected, spin in place (searching for bottle)
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = self.max_angular_vel  # Spin in place

        # publish velocity on cmd_vel
        self.cmd_vel_pub.publish(cmd_vel)


if __name__ == "__main__":
    nav = CrashBottlesNode()

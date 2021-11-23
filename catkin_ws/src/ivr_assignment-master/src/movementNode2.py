#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge, CvBridgeError


class movementNode2:
    def __init__(self):
        rospy.init_node("movementNode2", anonymous=True)
        self.joint1Pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
        self.joint3Pub = rospy.Publisher("/robot/joint3_position_controller/command" , Float64, queue_size=10)
        self.joint4Pub = rospy.Publisher("/robot/joint4_position_controller/command" , Float64, queue_size=10)
        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

        self.joint1SinMove = Float64()
        self.joint3SinMove = Float64()
        self.joint4SinMove = Float64()

        while True:
            self.sinusoidal_move()

    def sinusoidal_move(self):
        self.time = rospy.get_time()

        self.joint1SinMove.data = np.pi * np.sin(np.pi * self.time / 28)
        self.joint3SinMove.data = np.pi * np.sin(np.pi * self.time / 20) / 2
        self.joint4SinMove.data = np.pi * np.sin(np.pi * self.time / 18) / 2

        # Publish the results
        try:
            self.joint1Pub.publish(self.joint1SinMove)
            self.joint3Pub.publish(self.joint3SinMove)
            self.joint4Pub.publish(self.joint4SinMove)
        except CvBridgeError as e:
            print(e)
    # Recieve data, process it, and publish

# call the class
def main(args):
    mn2 = movementNode2()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)


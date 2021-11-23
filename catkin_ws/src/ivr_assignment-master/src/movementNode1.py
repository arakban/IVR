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


class movementNode1:
    def __init__(self):
        rospy.init_node("movementNode1", anonymous=True)
        self.joint2Pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
        self.joint3Pub = rospy.Publisher("/robot/joint3_position_controller/command" , Float64, queue_size=10)
        self.joint4Pub = rospy.Publisher("/robot/joint4_position_controller/command" , Float64, queue_size=10)
        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

        self.joint2Move = Float64()
        self.joint3Move = Float64()
        self.joint4Move = Float64()

        while True:
            self.movement()

    def movement(self):
        self.time = rospy.get_time()

        self.joint2Move.data = np.pi * np.sin(np.pi * self.time / 15) / 2
        self.joint3Move.data = np.pi * np.sin(np.pi * self.time / 20) / 2
        self.joint4Move.data = np.pi * np.sin(np.pi * self.time / 18) / 2

        # Publish the results
        try:
            self.joint2Pub.publish(self.joint2Move)
            self.joint3Pub.publish(self.joint3Move)
            self.joint4Pub.publish(self.joint4Move)
        except CvBridgeError as e:
            print(e)


def main(args):
    sc1 = movementNode1()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
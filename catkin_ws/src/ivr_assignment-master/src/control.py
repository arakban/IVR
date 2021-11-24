#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
import logging 
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge, CvBridgeError


class image_converter:

  # Defines publisher and subscriber
  def __init__(self):
    # initialize the node named controller
    rospy.init_node('controller', anonymous=True)
    
    # initialize subscribers to get joints' angular position to the robot
    self.joint_1_pub = rospy.Subscriber("joint_angle_1", Float64, self.callback_joint1)
    self.joint_3_pub = rospy.Subscriber("joint_angle_3", Float64, self.callback_joint3)
    self.joint_4_pub = rospy.Subscriber("joint_angle_4", Float64, self.callback_joint4)

    # initialize publishers to send joints' angular position to the robot - joint 2 is frozen
    self.joint_1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
    self.joint_3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
    self.joint_4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)

    # array to store joint data
    self.joint_angles = np.array([0.0,0.0,0.0,0.0])
    self.joint1 = Float64()
    self.joint3 = Float64()
    self.joint4 = Float64()

    # target position
    self.target = np.array([0.0,0.0,0.0,0.0])
    self.target_sub = rospy.Subscriber("/target_pos", Float64MultiArray, self.callback_update_target)

    # end effector - we don't care about orientation
    self.end_effector_pos = np.array([0.0, 0.0, 10.0], dtype='float64')
    # forward kinematics calculation publisher
    self.forward_kin_pib = rospy.Publisher("fk_end_effector", Float64MultiArray, queue_size=10)
    
    # error data and pub
    self.error = np.array([0.0, 0.0, 0.0], dtype='float64')
    self.error_d = np.array([0.0,0.0], dtype='float64')
    self.error_pub = rospy.Publisher("error", Float64MultiArray, queue_size=10)

    self.time_previous_step = np.array([rospy.get_time()], dtype='float64')

        
  def trajectory(self):
    # get current time
    cur_time = np.array([rospy.get_time() - self.time_trajectory])
    x_d = float(6 * np.cos(cur_time * np.pi / 100))
    y_d = float(6 + np.absolute(1.5 * np.sin(cur_time * np.pi / 100)))
    return np.array([x_d, y_d])

  def forward_kinematics(self,image):
    # get joint angles 
    j1, j2, j3, j4 = self.joint_angles

    # make calculations easier to read
    s1, c1, s2, c2, c3, c4 = np.sin(j1), np.cos(j1), s2, np.cos(j2), np.sin(j3), np.cos(j3), np.sin(j4), np.cos(j4)
    
    x = 7.2 * c1 + 2.8 * (c1 * c3 - s1 * s2 * s3)
    y = 2.8 * (c * np.sin(j2) * s1 - s1 * c3) - 7.2 * s1
    z = -2.8 * c1 * s1 
    end_effector = np.array([x,y,z])
    return end_effector

  # Recieve data from joint1 
  def callback_joint1(self,data):
    self.joints[0] = data.data

  # Recieve data from joint2 
  def callback_joint2(self,data):
    self.joints[1] = data.data
  
    # Recieve data from joint3
  def callback_joint3(self,data):
    self.joints[2] = data.data

    # Recieve data from joint4 
  def callback_joint4(self,data):
    self.joints[3] = data.data
    

# call the class
def main(args):
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)

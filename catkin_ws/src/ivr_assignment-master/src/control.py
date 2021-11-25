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
    #initialize the node named controller
    rospy.init_node('controller', anonymous=True)
    
    #initialize subscribers to get joints' angular position to the robot
    self.joint1_sub = rospy.Subscriber("joint_angle_1", Float64, self.callback_joint1)
    self.joint1_sub = rospy.Subscriber("joint_angle_3", Float64, self.callback_joint3)
    self.joint1_sub = rospy.Subscriber("joint_angle_4", Float64, self.callback_joint4)

    #initialize publishers to send joints' angular position to the robot - joint 2 is frozen
    self.joint_1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
    self.joint_3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
    self.joint_4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)

    #array to store joint data
    self.joint_angles = np.array([0.0,0.0,0.0,0.0])

    self.joint1 = Float64()
    self.joint3 = Float64()
    self.joint4 = Float64()

    #target position
    self.target = np.array([0.0,0.0,0.0,0.0])
    self.target_sub = rospy.Subscriber("/target_pos", Float64MultiArray, self.update_target)

    #end effector - we don't care about orientation
    self.end_effector_pos = np.array([0.0, 0.0, 10.0], dtype='float64')
    # forward kinematics calculation publisher
    self.forward_kin_pub = rospy.Publisher("fk_end_effector", Float64MultiArray, queue_size=10)
    
    #error data and pub
    self.error = np.array([0.0, 0.0, 0.0], dtype='float64')
    self.error_d = np.array([0.0,0.0], dtype='float64')
    self.error_pub = rospy.Publisher("error", Float64MultiArray, queue_size=10)

    self.time_previous_step = np.array([rospy.get_time()], dtype='float64')

  
  # forward kinematics formuala, to get the 
  def forward_kinematics(self):
    # get joint angles 
    j1, j2, j3, j4 = self.joint_angles

    #make calculations easier to read
    s1, c1, s2, c2, c3, c4 = np.sin(j1), np.cos(j1), np.sin(j2), np.cos(j2), np.sin(j3), np.cos(j3), np.sin(j4), np.cos(j4)
    
    #calculate the effect of rotation on each componenet 
    x = 2.8 * c1 + 2.8 * (c1 * c3 - s1 * s2 * s3)
    y = 2.8 * (c * np.sin(j2) * s1 - s1 * c3) - 7.2 * s1
    z = -2.8 * c1 * s1 + 4
    end_effector = np.array([x,y,z])
    
    return end_effector

  # calculate the Jacobian matrix to do inverse kinematics - get the relation between joint velocities & end-effector velocities of a robot manipulator
  # i.e. how much each joint needs to move to get to the target position
  def calc_jacobian(self):
    j1, j2, j3, j4 = self.joint_angles
    s1, c1, s2, c2, c3, c4 = np.sin(j1), np.cos(j1), np.sin(j2), np.cos(j2), np.sin(j3), np.cos(j3), np.sin(j4), np.cos(j4)
    
    #initialise a empty matrix with dimensions 
    jacob_matrix = np.zeroes(shape=(3,3))
    
    jacob_matrix[0,0] = -7.8 * s1 + 2.8*(s1 * c3) - (c1 * s2 * s3)      #R00
    jacob_matrix[0,1] = 2.8 * (-s1 * c1 * s3)                           #R01
    jacob_matrix[0,2] = 2.8 * ((-c1 * s3) - (s1 * s2 * c3))             #R02

    jacob_matrix[1,0] = 2.8 * ((-c1 * s3) - (s1 * s2 * s3))             #R10
    jacob_matrix[1,1] = 2.8 * (c1 * c2 * s3)                            #R11
    jacob_matrix[1,2] = 2.8 * ((c1 * s2 * c3) + (s1 * s3))              #R12

    jacob_matrix[2,0] = 0                                               #R20
    jacob_matrix[2,1] = 2.8 * (s2 * s3)                                 #R21
    jacob_matrix[2,2] = -2.8 * (c2 * c3)                                #R22

    return jacob_matrix

  def control_open(self):
    #dt 
    curr_time = np.array([rospy.get_time()])
    dt = cur_time - self.time_previous_step
    #make new time current time
    self.time_previous_step = curr_time
  
    #(psuedo)inverse of Jacobian 
    J_inv = np.linalg.pinv(self.calculate_jacobian())

    #get end-effector positon
    pos = self.end_effector_pos

    #get target positon
    pos_d = self.target

    #calculate error 
    self.error = (pos_d - pos)

    #calculate delta of error
    self.error_d = self.error/dt

    #calculate the estimated change in joint angles needed for desired movement  
    joints_delta = np.dot(J_inv,self.error.T) * dt

    #new joint angles
    self.joint_angles  = self.joint_angles + joints_delta

    #publish new joint angle


  # Recieve data from joint1 
  def callback_joint1(self,joints):
    self.joint_angles[0] = joints.data[0]

  def callback_joint3(self,joints):
    self.joint_angles[2] = joints.data[2]

  def callback_joint4(self,joints):
    self.joint_angles[3] = joints.data[3]

  def update_target(self):
    self.target = np.array(target.data)
    

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

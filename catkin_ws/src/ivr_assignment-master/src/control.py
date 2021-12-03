#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
import logging 
import message_filters
from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64, Float64MultiArray
from cv_bridge import CvBridge, CvBridgeError


class control:
  # Defines publisher and subscriber
  def __init__(self):
    #initialize the node named controller
    rospy.init_node('control', anonymous=True)
    global rate 
    rate = rospy.Rate(50)  # 50hz
    
    #bridge between openCV and ROS
    self.bridge = CvBridge()

    #initialize subscribers to get joints' angular position to the robot
    self.joint1_sub = message_filters.Subscriber("joint_angle_1", Float64, queue_size=1)
    self.joint3_sub = message_filters.Subscriber("joint_angle_3", Float64, queue_size=1)
    self.joint4_sub = message_filters.Subscriber("joint_angle_4", Float64, queue_size=1)

    #initialize publishers to send joints' angular position to the robot - joint 2 is frozen, this will needed for FK
    self.joint_1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
    self.joint_3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
    self.joint_4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)
    
    #end effector - we don't care about orientation
    self.red_center_sub = message_filters.Subscriber("red_center", Float64MultiArray, queue_size=1)
    self.end_effector_sub = message_filters.Subscriber("red_center", Float64MultiArray, queue_size=10)

    # #synchronise topics for FK, listen to updates every 3 seconds as this how long 
    # self.joint_sync = message_filters.ApproximateTimeSynchronizer([self.joint1_sub, self.joint3_sub, self.joint4_sub, self.red_center_sub],
    #                                                         queue_size=1, slop= 3, allow_headerless=True)
    # self.joint_sync.registerCallback(self.callback_fk)

    #array to store joint data
    self.joint_angles = np.array([0.0,0.0,0.0])

    self.joint1 = Float64()
    self.joint3 = Float64()
    self.joint4 = Float64()

    #target position
    self.target = Float64MultiArray()
    self.target_sub = message_filters.Subscriber("/target_pos", Float64MultiArray, queue_size=1)

    #forward kinematics for end-effector calculation publisher
    self.fk_calc = Float64MultiArray()
    self.fk_error = Float64MultiArray()
    self.fk_pub = rospy.Publisher("fk_end_effector", Float64MultiArray, queue_size=1)
    self.fk_error_pub = rospy.Publisher("fk_error", Float64MultiArray, queue_size=1)
    
    #error data and pub for IK controller
    self.error = np.array([0.0,0.0,0.0,0.0], dtype='float64')
    self.error_d = np.array([0.0,0.0,0.0,0.0], dtype='float64')

    self.time_previous_step = rospy.get_time()

    #synchronise topics, at 50 HZ, for IK contoller
    self.target_sync = message_filters.ApproximateTimeSynchronizer([self.target_sub, self.end_effector_sub],
                                                            queue_size=10, slop=0.02, allow_headerless=True)
    self.target_sync.registerCallback(self.callback_ik)

  
  # forward kinematics formuala, to get the 
  def forward_kinematics(self):
    # get joint angles 
    j1, j3, j4 = self.joint_angles

    #make calculations easier to read
    s1, c1, s3, c3, s4, c4 = np.sin(j1), np.cos(j1), np.sin(j3), np.cos(j3), np.sin(j4), np.cos(j4)
    
    #calculate the effect of rotation on each componenet 
    x = 2.8 * (s1 * s3 * c4) + 2.8 * (c1 * s4) + 3.2 * (s1 * s3)
    y = 2.8 * (-c1 * s3 * c4) - 2.8 * (s1*s4) + 3.2 * (c1 * s3) 
    z = 2.8 * (c3 * c4) + 3.2 * (c3) + 4
    end_effector = np.array([x,y,z])
    
    return end_effector
  
  # calculate the Jacobian matrix to do inverse kinematics - get the relation between joint velocities & end-effector velocities of a robot manipulator
  # i.e. how much each joint needs to move to get to the target position
  def calc_jacobian(self):
    j1, j3, j4 = self.joint_angles
    s1, c1, s3, c3, s4, c4 = np.sin(j1), np.cos(j1), np.sin(j3), np.cos(j3), np.sin(j4), np.cos(j4)
    
    #initialise a empty matrix with dimensions 
    jacob_matrix = np.zeros(shape=(3,3))
    
    jacob_matrix[0,0] = 2.8 * (c1 * s3 * c4) - 2.8 * (s1 * s4) + 3.2 * (s1 * s3)                       #RX1
    jacob_matrix[0,1] = 2.8 * (s1 * c3 * c4) + 3.2 * (s1 * s3)                                         #RX3
    jacob_matrix[0,2] = 2.8 * (c1 * c4) - 2.8 * (s1 * s3 * s4)                                         #RX4

    jacob_matrix[1,0] = 2.8 * (s1 * s3 * c4) + 2.8 * (c1 * s4) + 3.2 * (s1 * s3)                       #RY1
    jacob_matrix[1,1] = 2.8 * (s1 * c3 * c4) - 2.8 * (c1 * s4) + 3.2 * (s1 * c3)                       #RY3
    jacob_matrix[1,2] = 2.8 * (s1 * s3 * s4) - 2.8 * (c1 * c4) + 3.2 * (s1 * s3)                       #RY4

    jacob_matrix[2,0] = 2.8 * (c3 * c4) + 3.2 * c3                                                     #RZ1
    jacob_matrix[2,1] = 2.8 * (-s3 * c4) - 3.2 * s3                                                    #RZ3
    jacob_matrix[2,2] = 2.8 * (c3 * s4)                                                                #RZ4

    return jacob_matrix
  
  # Make robot move towards target using a control loop
  def control_open(self, target_data, end_effector_pos):
    #dt 
    curr_time = np.array([rospy.get_time()])
    dt = curr_time - self.time_previous_step
    #make new time current time
    self.time_previous_step = curr_time
  
    #(psuedo)inverse of Jacobian 
    J_inv = np.linalg.pinv(self.calc_jacobian())

    #get end-effector positon
    pos = end_effector_pos

    #get target positon
    pos_d = target_data

    #calculate error 
    self.error = (pos_d - pos)

    #calculate delta of error
    self.error_d = self.error/dt

    #calculate the estimated change in joint angles needed for desired movement  
    joints_delta = np.dot(J_inv,self.error_d.T) * dt

    #new joint angles
    new_joint_angles = self.joint_angles + joints_delta

    return new_joint_angles

  # Receive data from joint1 
  def callback_fk(self,joint1,joint3,joint4,red_center):
    #update 1st joint
    self.joint_angles[0] = joint1.data

    #update 3rd joint
    self.joint_angles[1] = joint3.data

    #update 4th joint 
    self.joint_angles[2] = joint4.data

    #calculate forward kinematics to get the estimation of joint states and publish
    self.fk_calc.data = self.forward_kinematics()
    self.fk_error.data = self.fk_calc.data - red_center.data

    try:
      self.fk_pub.publish(self.fk_calc)
      self.fk_error_pub.publish(self.fk_error)
      #show end-effector estimated by the images
      print(f"End effector from vison2: {red_center.data}")
      #show FK estimattion
      print(f"FK pred: {self.fk_calc.data}")
      #show error
      print(f"error in FK: {self.fk_error.data}")
      rate.sleep()
    except CvBridgeError as e:
      print(e)
    
  def callback_ik(self,target_data,end_effector):
    #calculate new joint angles
    new_joint_angles = self.control_open(np.array(target_data.data),np.array(end_effector.data))

    #publish new joint angles
    self.joint1 = new_joint_angles[0]
    self.joint3 = new_joint_angles[1]
    self.joint4 = new_joint_angles[2]
    
    try: 
      self.joint_1_pub.publish(self.joint1)
      self.joint_3_pub.publish(self.joint3)
      self.joint_4_pub.publish(self.joint4)
    except CvBridgeError as e:
      print(e)

# call the class
def main(args):
  c = control()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)

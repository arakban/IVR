#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
import logging 
import message_filters
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64, Float64MultiArray
from cv_bridge import CvBridge, CvBridgeError


class control:
  # Defines publisher and subscriber
  def __init__(self):
    #initialize the node named controller
    rospy.init_node('control', anonymous=True)
    #rate = rospy.Rate(50)  # 50hz
    #bridge between openCV and ROS
    self.bridge = CvBridge()
    
    #initialize subscribers to get joints' angular position to the robot
    self.joint1_sub = message_filters.Subscriber("joint_angle_1", Float64)
    self.joint3_sub = message_filters.Subscriber("joint_angle_3", Float64)
    self.joint4_sub = message_filters.Subscriber("joint_angle_4", Float64)

    #initialize publishers to send joints' angular position to the robot - joint 2 is frozen
    self.joint_1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
    self.joint_3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
    self.joint_4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)

    #array to store joint data
    self.joint_angles = np.array([0.0,0.0,0.0])

    self.joint1 = Float64()
    self.joint3 = Float64()
    self.joint4 = Float64()

    #target position
    self.target = Float64MultiArray()
    #self.target_sub = message_filters.Subscriber("/target_pos", Float64MultiArray)

    #end effector - we don't care about orientation
    self.end_effector_pos = Float64MultiArray()
    self.end_effector_sub = message_filters.Subscriber("red_center", Float64MultiArray)

    #forward kinematics for end-effector calculation publisher
    self.forward_kin_calc = Float64MultiArray()
    self.forward_kin_error = Float64MultiArray()
    self.forward_kin_pub = rospy.Publisher("fk_end_effector", Float64MultiArray, queue_size=10)
    self.forward_kin_error_pub = rospy.Publisher("fk_error", Float64MultiArray, queue_size=10)
    
    #error data and pub
    self.error = np.array([0.0,0.0,0.0,0.0], dtype='float64')
    self.error_d = np.array([0.0,0.0,0.0,0.0], dtype='float64')
    self.fk_error_pub = rospy.Publisher("error", Float64MultiArray, queue_size=10)

    self.time_previous_step = rospy.get_time()

    #synchronise topics, at 50 HZ
    self.ts = message_filters.ApproximateTimeSynchronizer([self.joint1_sub, self.joint3_sub, self.joint4_sub, self.end_effector_sub],
                                                            queue_size=10, slop=3, allow_headerless=True)
    self.ts.registerCallback(self.callback)

  
  # forward kinematics formuala, to get the 
  def forward_kinematics(self):
    # get joint angles 
    j1, j3, j4 = self.joint_angles

    #make calculations easier to read
    s1, c1, s3, c3, s4, c4 = np.sin(j1), np.cos(j1), np.sin(j3), np.cos(j3), np.sin(j4), np.cos(j4)
    
    #calculate the effect of rotation on each componenet 
    x = 2.8 * (c1 * s3 * c4)  + 2.8 * (c1 * s4) + 3.2 * (s1 * s3)
    y = 2.8 * (c1 * s3 * c4) - 2.8 * (s1*s4) + 3.2 * (c1 * s3) 
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
    
    jacob_matrix[0,0] = 2.8 * (s1 * s3 * c4) - 2.8 * (s1 * s4) - 3.2 * (s1 * s3)                       #RX1
    jacob_matrix[0,1] = 2.8 * (s1 * c1 * s3) + 3.2 * (s1 * s3)                                         #RX3
    jacob_matrix[0,2] = 2.8 * (c1 * c4) - 2.8 * (s1 * s3 * s4)                                         #RX4

    jacob_matrix[1,0] = 2.8 * (s1 * s3 * c4) + 2.8 * (c1 * s4) + 3.2 * (s1 * s3)                       #RY1
    jacob_matrix[1,1] = 2.8 * (s1 * c3 * c4) - 2.8 * (c1 * s4) + 3.2 * (s1 * c3)                       #RY3
    jacob_matrix[1,2] = 2.8 * (s1 * s3 * s4) - 2.8 * (c1 * c4) + 3.2 * (s1 * s3)                       #RY4

    jacob_matrix[2,0] = 2.8 * (c3 * c4) + 3.2 * c3                                                     #RZ1
    jacob_matrix[2,1] = 2.8 * (-s3 * c4) - 3.2 * s3                                                    #RZ3
    jacob_matrix[2,2] = 2.8 * (c3 * s4) + 3.2 * c3                                                     #RZ4

    return jacob_matrix

  def control_open(self):
    #dt 
    curr_time = np.array([rospy.get_time()])
    dt = curr_time - self.time_previous_step
    #make new time current time
    self.time_previous_step = curr_time
  
    #(psuedo)inverse of Jacobian 
    J_inv = np.linalg.pinv(self.calc_jacobian())

    #get end-effector positon
    pos = self.end_effector_pos

    #get target positon
    pos_d = self.target

    #calculate error 
    self.error = (pos_d - pos)

    #calculate delta of error
    self.error_d = self.error/dt

    #PID control: calculate the estimated change in joint angles needed for desired movement  
    joints_delta = np.dot(J_inv,self.error_d.T) * dt

    #new joint angles
    new_joint_angles = self.joint_angles + joints_delta

    return new_joint_angles

  # Receive data from joint1 
  def callback(self,joint1,joint3,joint4,red_centre):
    #update 1st joint
    self.joint_angles[0] = joint1.data

    #update 3rd joint
    self.joint_angles[1] = joint3.data

    #update 4th joint 
    self.joint_angles[2] = joint4.data

    #calculate forward kinematics to get the estimation of joint states and publish
    self.forward_kin_calc.data = self.forward_kinematics()
    #get end-effector estimated by the images 
    self.end_effector_pos.data = red_centre.data
    self.forward_kin_error.data = self.forward_kin_calc.data - self.end_effector_pos.data
    try:
      self.forward_kin_pub.publish(self.forward_kin_calc)
      self.forward_kin_error_pub.publish(self.forward_kin_error)
      print(f"End effector from vison2: {self.end_effector_pos.data}")
      print(f"FK pred: {self.forward_kin_calc.data}")
      print(f"error in FK: {self.forward_kin_error.data}")
    except CvBridgeError as e:
      print(e)
    
    # #make robot move towards target using a control loop
    # self.target = target_data.data
    # #now calculate new joint angles
    # new_joint_angles = self.control_open()

    # #publish new joint angles
    # self.joint1 = new_joint_angles[0]
    # self.joint3 = new_joint_angles[1]
    # self.joint4 = new_joint_angles[2]
    # try: 
    #   self.joint_1_pub.publish(self.joint1)
    #   self.joint_3_pub.publish(self.joint3)
    #   self.joint_4_pub.publish(self.joint4)
    # except CvBridgeError as e:
    #   print(e)

    # self.end_effector_pos = red_centre.data

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

#!/usr/bin/env python3

import sys

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
import message_filters
from numpy import sin,cos

class vision_2:
    def __init__(self):
        rospy.init_node("vision_2", anonymous=True)

        self.greenC1 = np.array([])
        self.redC1 = np.array([])
        self.blueC1 = np.array([])
        self.yellowC1 = np.array([])

        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

        self.image_sub1 = message_filters.Subscriber("/camera1/robot/image_raw", Image)
        self.image_sub2 = message_filters.Subscriber("/camera2/robot/image_raw", Image)
        self.sync = message_filters.ApproximateTimeSynchronizer([self.image_sub1, self.image_sub2], 10, 1)
        self.sync.registerCallback(self.callback)

        self.joint1Pub = rospy.Publisher("joint_angle_1", Float64, queue_size=10)
        self.joint3Pub = rospy.Publisher("joint_angle_3", Float64, queue_size=10)
        self.joint4Pub = rospy.Publisher("joint_angle_4", Float64, queue_size=10)

        self.redCenterPub = rospy.Publisher("red_center", Float64MultiArray, queue_size=10)
        self.greenCenterPub = rospy.Publisher("green_center", Float64MultiArray, queue_size=10)
        self.blueCenterPub = rospy.Publisher("blue_center", Float64MultiArray, queue_size=10)
        self.yellowCenterPub = rospy.Publisher("yellow_center", Float64MultiArray, queue_size=10)

        self.relativeRedPub = rospy.Publisher("relative_red", Float64MultiArray, queue_size=10)
        self.relativeBluePub = rospy.Publisher("relative_blue", Float64MultiArray, queue_size=10)
        self.relativeYellowPub = rospy.Publisher("relative_yellow", Float64MultiArray, queue_size=10)

        self.meterRedPub = rospy.Publisher("meter_red", Float64MultiArray, queue_size=10)
        self.meterBluePub = rospy.Publisher("meter_blue", Float64MultiArray, queue_size=10)
        self.meterYellowPub = rospy.Publisher("meter_yellow", Float64MultiArray, queue_size=10)

        self.vectorYBPub = rospy.Publisher("vector_yb", Float64MultiArray, queue_size=10)
        self.vectorYBtoBRPub = rospy.Publisher("vector_yb_br", Float64MultiArray, queue_size=10)
        
        self.joint1 = Float64(); self.joint3 = Float64(); self.joint4 = Float64()

        self.redMsg = Float64MultiArray(); self.greenMsg = Float64MultiArray(); self.blueMsg = Float64MultiArray(); self.yellowMsg = Float64MultiArray()

        self.relativeRedMSg = Float64MultiArray(); self.relativeBlueMSg = Float64MultiArray(); self.relativeYellowMSg = Float64MultiArray()

        self.meterRedMsg = Float64MultiArray(); self.meterBlueMsg = Float64MultiArray(); self.meterYellowMsg = Float64MultiArray()

        self.vectorYBMsg = Float64MultiArray(); self.vectorYBtoBRMSg = Float64MultiArray()
        
        self.iterationNumber = 0; self.lastJoint1 = 0.0; self.lastJoint3 = 0.0; self.lastJoint4 = 0.0

    def publishInfo(self):
        self.greenMsg.data = (self.originPoint / 500.0).tolist()
        self.yellowMsg.data = (self.finalYellowCenter / 500.0).tolist()
        self.blueMsg.data = (self.finalBlueCenter / 500.0).tolist()
        self.redMsg.data = (self.finalRedCenter / 500.0).tolist()

        self.relativeYellowMSg.data = (self.finalYellowCenter - self.originPoint).tolist()
        self.relativeBlueMSg.data = (self.finalBlueCenter - self.originPoint).tolist()
        self.relativeRedMSg.data = (self.finalRedCenter - self.originPoint).tolist()

        self.meterYellowMsg.data = ((self.finalYellowCenter - self.originPoint) * self.pixelsToMetersRatio).tolist()
        self.meterBlueMsg.data = ((self.finalBlueCenter - self.originPoint) * self.pixelsToMetersRatio).tolist()
        self.meterRedMsg.data = ((self.finalRedCenter - self.originPoint) * self.pixelsToMetersRatio).tolist()

        self.vectorYBtoBRMSg.data = ((self.vectorYB - self.vectorBR) / 500.0).tolist()
        self.vectorYBMsg.data = (self.vectorYB / 500.0).tolist()

        try:
            self.greenCenterPub.publish(self.greenMsg)
            self.yellowCenterPub.publish(self.yellowMsg)
            self.blueCenterPub.publish(self.blueMsg)
            self.redCenterPub.publish(self.redMsg)

            self.relativeYellowPub.publish(self.relativeYellowMSg)
            self.relativeBluePub.publish(self.relativeBlueMSg)
            self.relativeRedPub.publish(self.relativeRedMSg)

            self.meterYellowPub.publish(self.meterYellowMsg)
            self.meterBluePub.publish(self.meterBlueMsg)
            self.meterRedPub.publish(self.meterRedMsg)

            self.vectorYBtoBRPub.publish(self.vectorYBtoBRMSg)
            self.vectorYBPub.publish(self.vectorYBMsg)
        except CvBridgeError as e:
            print(e)

    def getCenter(self, mask):
        control = sum(sum(mask))
        if control < 10:
            return np.array([])
        M = cv2.moments(mask)
        X = int(M["m10"] / M["m00"])
        Y = int(M["m01"] / M["m00"])
        return np.array([X, -Y])
            
    def findAllPoints(self, img):
        redMask = cv2.inRange(img, np.array([0, 0, 20]), np.array([20, 20, 255]))
        greenMask = cv2.inRange(img, np.array([0, 20, 0]), np.array([20, 255, 20]))
        blueMask = cv2.inRange(img, np.array([10, 0, 0]), np.array([255, 20, 20]))
        yellowMask = cv2.inRange(img, np.array([0, 10, 10]), np.array([0, 255, 255]))

        redCenter = self.getCenter(redMask)
        greenCenter = self.getCenter(greenMask)
        blueCenter = self.getCenter(blueMask)
        yellowCenter = self.getCenter(yellowMask)
        return redCenter, greenCenter, blueCenter, yellowCenter
    
    def callback(self, data1, data2):
        try:
            self.cv_image1 = self.bridge.imgmsg_to_cv2(data1, "bgr8")
            self.cv_image2 = self.bridge.imgmsg_to_cv2(data2, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.redC1, self.greenC1, self.blueC1, self.yellowC1 = self.findAllPoints(self.cv_image1)
        self.redC2, self.greenC2, self.blueC2, self.yellowC2 = self.findAllPoints(self.cv_image2)
        
        self.originPoint = self.originhandler(self.greenC1, self.greenC2)
        self.finalRedCenter = self.camJoin(self.redC1, self.redC2)
        self.finalBlueCenter = self.camJoin(self.blueC1, self.blueC2)
        self.finalYellowCenter = self.camJoin(self.yellowC1, self.yellowC2)

        self.pixelsToMetersRatio = 4.0 / np.linalg.norm(self.finalYellowCenter - self.originPoint)

        self.calculateAngles()
        
        self.iterationNumber += 1
        self.lastJoint1 = self.joint1.data
        self.lastJoint3 = self.joint3.data
        self.lastJoint4 = self.joint4.data

        try:
            self.joint1Pub.publish(self.joint1)
            self.joint3Pub.publish(self.joint3)
            self.joint4Pub.publish(self.joint4)
        except CvBridgeError as e:
            print(e)

        self.publishInfo()

    def get_vector_angle(self, vec1, vec2):
        return np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def calculateAngles(self):
        self.vectorYB = self.finalBlueCenter - self.finalYellowCenter
        self.vectorBR = self.finalRedCenter - self.finalBlueCenter

        th3 = -self.joint3.data
        reverseM3 = np.array([[1, 0, 0],
                                  [0, np.cos(th3), -np.sin(th3)],
                                  [0, np.sin(th3), np.cos(th3)]])
        revDot3 = reverseM3.dot(self.vectorBR.T)
        
        th1 = -self.joint1.data
        reverseM1 = np.array([[np.cos(th1), -np.sin(th1), 0],
                                  [np.sin(th1), np.cos(th1), 0],
                                  [0, 0, 1]])
        revDot1 = reverseM1.dot(revDot3.T)

        vecjoint4 = revDot1[0:3:2]
        sign = np.sign(vecjoint4[0])

        joint_1 = np.arctan2(self.vectorYB[0], -self.vectorYB[1])
        joint_3 = np.absolute(self.get_vector_angle(self.vectorYB, [0, 0, 1]))
        joint_4 = self.get_vector_angle(self.vectorYB, self.vectorBR) * sign
        
        self.joint1.data = self.limAngle(joint_1, np.pi, self.lastJoint1)
        self.joint3.data = self.limAngle(joint_3, np.pi / 2.0, self.lastJoint3)
        self.joint4.data = self.limAngle(joint_4, np.pi / 2.0, self.lastJoint4)

    def limAngle(self, jointAngle, limit, lastAngle):
        jointAngle = max(min(jointAngle, limit), -limit)
        if (abs(jointAngle - lastAngle) > 0.5) and (abs(jointAngle - lastAngle) <= 2) and (self.iterationNumber > 30):
            jointAngle += np.sign(jointAngle - lastAngle) * 0.05
        if (abs(jointAngle - lastAngle) > 2) and (self.iterationNumber > 10):
            jointAngle = lastAngle
        return jointAngle

    def camJoin(self, cam1, cam2):
        if (cam1.size == 0) and (cam2.size == 0):
            return np.array([-1.0, -1.0, -1.0])
        elif cam1.size == 0:
            return np.array([cam2[0], self.originPoint[1], cam2[1]])
        elif cam2.size == 0:
            return np.array([self.originPoint[1], cam1[0], cam1[1]])
        elif (cam1.size == 2) and (cam2.size == 2):
            return np.array([cam2[0], cam1[0], (cam2[1] + cam1[1]) / 2])
        else:
            return np.array([-1.0, -1.0, -1.0])

    def originhandler(self, cam1, cam2):
        if (cam1.size == 0) and (cam2.size == 0):
            return np.array([-1.0, -1.0, -1.0])
        elif cam1.size == 0:
            return np.array([cam2[0], cam2[0], cam2[1]])
        elif cam2.size == 0:
            return np.array([cam1[0], cam1[0], cam1[1]])
        elif (cam1.size == 2) and (cam2.size == 2):
            return np.array([cam2[0], cam1[0], (cam2[1] + cam1[1]) / 2])
        else:
            return np.array([-1.0, -1.0, -1.0])

# call the class
def main(args):
    v2 = vision_2()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
    # run the code if the node is called


if __name__ == '__main__':
    main(sys.argv)
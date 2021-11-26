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

class vision_1:
    def __init__(self):
        rospy.init_node("vision_1", anonymous=True)
        
        self.greenC1 = np.array([])
        self.redC1 = np.array([])
        self.blueC1 = np.array([])
        self.yellowC1 = np.array([])

        self.lim = np.pi / 2.0

        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

        self.image_sub1 = message_filters.Subscriber("/camera1/robot/image_raw", Image)
        self.image_sub2 = message_filters.Subscriber("/camera2/robot/image_raw", Image)
        self.sync = message_filters.ApproximateTimeSynchronizer([self.image_sub1, self.image_sub2], 10, 1)
        self.sync.registerCallback(self.callback)

        self.joint2Pub = rospy.Publisher("joint_angle_2", Float64, queue_size=10)
        self.joint3Pub = rospy.Publisher("joint_angle_3", Float64, queue_size=10)
        self.joint4Pub = rospy.Publisher("joint_angle_4", Float64, queue_size=10)

        self.redCenterPub = rospy.Publisher("red_center", Float64MultiArray, queue_size=10)
        self.greenCenterPub = rospy.Publisher("green_center", Float64MultiArray, queue_size=10)
        self.blueCenterPub = rospy.Publisher("blue_center", Float64MultiArray, queue_size=10)
        self.yellowCenterPub = rospy.Publisher("yellow_center", Float64MultiArray, queue_size=10)
        
        self.joint2 = Float64(); self.joint3 = Float64(); self.joint4 = Float64()

        self.redDat = Float64MultiArray(); self.greenDat = Float64MultiArray(); self.blueDat = Float64MultiArray(); self.yellowDat = Float64MultiArray()

        self.iterationNumber = 0; self.lastJoint2 = 0.0; self.lastJoint3 = 0.0; self.lastJoint4 = 0.0

    def callback(self, data1, data2):
        try:
            self.cv_image1 = self.bridge.imgmsg_to_cv2(data1, "bgr8")
            self.cv_image2 = self.bridge.imgmsg_to_cv2(data2, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.redC1, self.greenC1, self.blueC1, self.yellowC1 = self.findCenters(self.cv_image1)
        self.redC2, self.greenC2, self.blueC2, self.yellowC2 = self.findCenters(self.cv_image2)
        
        self.originPoint = self.findOrigin(self.greenC1, self.greenC2)
        self.finalRedCenter = self.camJoin(self.redC1, self.redC2)
        self.finalBlueCenter = self.camJoin(self.blueC1, self.blueC2)
        self.finalYellowCenter = self.camJoin(self.yellowC1, self.yellowC2)

        self.pixelsToMetersRatio = 4.0 / np.linalg.norm(self.finalYellowCenter - self.originPoint)

        self.calculateAngles()
        
        self.iterationNumber += 1
        self.lastJoint2 = self.joint2.data
        self.lastJoint3 = self.joint3.data
        self.lastJoint4 = self.joint4.data
        try:
            self.joint2Pub.publish(self.joint2)
            self.joint3Pub.publish(self.joint3)
            self.joint4Pub.publish(self.joint4)
        except CvBridgeError as e:
            print(e)

        self.greenDat.data = (self.originPoint / 500.0).tolist()
        self.yellowDat.data = (self.finalYellowCenter / 500.0).tolist()
        self.blueDat.data = (self.finalBlueCenter / 500.0).tolist()
        self.redDat.data = (self.finalRedCenter / 500.0).tolist()
        try:
            self.greenCenterPub.publish(self.greenDat)
            self.yellowCenterPub.publish(self.yellowDat)
            self.blueCenterPub.publish(self.blueDat)
            self.redCenterPub.publish(self.redDat)
        except CvBridgeError as e:
            print(e)

    def getCenter(self, color):
        control = sum(sum(color))
        if control < 10:
            return np.array([])
        M = cv2.moments(color)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return np.array([X, -Y])

    def findCenters(self, img):
        red = cv2.inRange(img, np.array([0, 0, 100]), np.array([0, 0, 255]))
        green = cv2.inRange(img, np.array([0, 100, 0]), np.array([0, 255, 0]))
        blue = cv2.inRange(img, np.array([100, 0, 0]), np.array([255, 0, 0]))
        yellow = cv2.inRange(img, np.array([0, 100, 100]), np.array([0, 255, 255]))

        redCenter = self.getCenter(red)
        greenCenter = self.getCenter(green)
        blueCenter = self.getCenter(blue)
        yellowCenter = self.getCenter(yellow)

        return redCenter, greenCenter, blueCenter, yellowCenter

    def calculateAngles(self):
        self.vectorYB = self.finalBlueCenter - self.finalYellowCenter
        self.vectorBR = self.finalRedCenter - self.finalBlueCenter

        vecjoint2 = np.array([self.vectorYB[0], self.vectorYB[2]])
        vecjoint3 = np.array([self.vectorYB[1], self.vectorYB[2]])
        zUnitVector = np.array([0, 1])

        self.joint2.data = np.arctan2(np.cross(vecjoint2, zUnitVector), np.dot(zUnitVector, vecjoint2))
              

        joint3Ang = np.arctan2(np.cross(vecjoint3, zUnitVector), np.dot(zUnitVector, vecjoint3))
        self.joint3.data = -(joint3Ang - np.sign(joint3Ang) * 0.35 * abs(np.sin(self.joint2.data)))

        th3 = -self.joint3.data 
        reverseM3 = np.array([[1, 0, 0], [0, np.cos(th3), -np.sin(th3)], [0, np.sin(th3), np.cos(th3)]])
        revDot3 = reverseM3.dot(self.vectorBR.T)
        
        th2 = -self.joint2.data
        reverseM1 = np.array([[np.cos(th2), 0, np.sin(th2)], [0, 1, 0], [-np.sin(th2), 0, np.cos(th2)]])
        revDot2 = reverseM1.dot(revDot3.T)

        vecjoint4 = revDot2[0:3:2]
        sign = np.sign(vecjoint4[0])

        vecj4 = self.vectorAng(self.vectorBR, self.vectorYB)
                
        self.joint4.data = vecj4 * sign

        self.joint2.data = self.limit(self.joint2.data, np.pi / 2.0, self.lastJoint2)
        self.joint3.data = self.limit(self.joint3.data, np.pi / 2.0, self.lastJoint3)
        self.joint4.data = self.limit(self.joint4.data, np.pi / 2.0, self.lastJoint4)

    def vectorAng(self,vect1,vect2):
        dot_vect1_vect2 = np.dot(vect1,vect2)
        length_vect1 = np.sqrt(np.sum(vect1 ** 2))
        length_vect2 = np.sqrt(np.sum(vect2 ** 2))
        return np.arccos(dot_vect1_vect2/(length_vect1 * length_vect2))

    def limit(self, jointAngle, lim, lastAngle):
        jointAngle = max(min(jointAngle, lim), -lim)
        if (abs(jointAngle - lastAngle) > 1.5) and (self.iterationNumber > 10):
            jointAngle = lastAngle
        if (abs(jointAngle - lastAngle) > 0.3) and (abs(jointAngle - lastAngle) <= 1.5) and (self.iterationNumber > 10):
            jointAngle += np.sign(jointAngle - lastAngle) * 0.03
        return jointAngle

    def camJoin(self, cam1, cam2):
        if (cam1.size == 2) and (cam2.size == 2):
            return np.array([cam2[0], cam1[0], (cam2[1] + cam1[1]) / 2])
        elif (cam1.size == 0) and (cam2.size == 0):
            return np.array([-1.0, -1.0, -1.0])
        elif cam1.size == 0:
            return np.array([cam2[0], self.originPoint[1], cam2[1]])
        elif cam2.size == 0:
            return np.array([self.originPoint[1], cam1[0], cam1[1]])
        else:
            return np.array([-1.0, -1.0, -1.0])

    def findOrigin(self, cam1, cam2):
        if (cam1.size == 2) and (cam2.size == 2):
            return np.array([cam2[0], cam1[0], (cam2[1] + cam1[1]) / 2])
        elif (cam1.size == 0) and (cam2.size == 0):
            return np.array([-1.0, -1.0, -1.0])
        elif cam1.size == 0:
            return np.array([cam2[0], cam2[0], cam2[1]])
        elif cam2.size == 0:
            return np.array([cam1[0], cam1[0], cam1[1]])
        else:
            return np.array([-1.0, -1.0, -1.0])

# call the class
def main(args):
    v1 = vision_1()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
    # run the code if the node is called


if __name__ == '__main__':
    main(sys.argv)
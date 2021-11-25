#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from numpy import sin,cos
import math

class vision_2:
    def __init__(self):
        rospy.init_node("vision_2", anonymous=True)

        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()
        self.initialiseSubscribers()
        self.initialisePublishers()
        self.initaliseMessageObjects()
        self.initialiseBlobCentres()
        self.initialiseJointLookback()

    def initialiseSubscribers(self):
        self.image_sub1 = message_filters.Subscriber("/camera1/robot/image_raw", Image)
        self.image_sub2 = message_filters.Subscriber("/camera2/robot/image_raw", Image)
        self.sync = message_filters.ApproximateTimeSynchronizer([self.image_sub1, self.image_sub2], 10, 1)
        self.sync.registerCallback(self.callbackk)

    def initialisePublishers(self):
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

    # required so very first callback doesn't fail
    def initialiseBlobCentres(self):
        self.greenC1 = np.array([])
        self.redC1 = np.array([])
        self.blueC1 = np.array([])
        self.yellowC1 = np.array([])

    def initaliseMessageObjects(self):
        self.joint1 = Float64()
        self.joint3 = Float64()
        self.joint4 = Float64()

        self.redMsg = Float64MultiArray()
        self.greenMsg = Float64MultiArray()
        self.blueMsg = Float64MultiArray()
        self.yellowMsg = Float64MultiArray()

        self.relativeRedMSg = Float64MultiArray()
        self.relativeBlueMSg = Float64MultiArray()
        self.relativeYellowMSg = Float64MultiArray()

        self.meterRedMsg = Float64MultiArray()
        self.meterBlueMsg = Float64MultiArray()
        self.meterYellowMsg = Float64MultiArray()

        self.vectorYBMsg = Float64MultiArray()
        self.vectorYBtoBRMSg = Float64MultiArray()

    def initialiseJointLookback(self):
        self.iterationNumber = 0

        self.lastJoint1 = 0.0
        self.lastJoint3 = 0.0
        self.lastJoint4 = 0.0

    def publishangles(self):
        # Publish the results
        try:
            self.joint1Pub.publish(self.joint1)
            self.joint3Pub.publish(self.joint3)
            self.joint4Pub.publish(self.joint4)
        except CvBridgeError as e:
            print(e)

    def publishCentresAndVectors(self):
        self.redMsg.data = (self.finalRedCenter/500.0).tolist()
        self.blueMsg.data = (self.finalBlueCenter/500.0).tolist()
        self.yellowMsg.data = (self.finalYellowCenter/500.0).tolist()
        self.greenMsg.data = (self.originPoint/500.0).tolist()

        self.relativeRedMSg.data = (self.finalRedCenter - self.originPoint).tolist()
        self.relativeBlueMSg.data = (self.finalBlueCenter - self.originPoint).tolist()
        self.relativeYellowMSg.data = (self.finalYellowCenter - self.originPoint).tolist()

        self.meterRedMsg.data = ((self.finalRedCenter - self.originPoint) * self.pixelsToMetersRatio).tolist()
        self.meterBlueMsg.data = ((self.finalBlueCenter - self.originPoint) * self.pixelsToMetersRatio).tolist()
        self.meterYellowMsg.data = ((self.finalYellowCenter - self.originPoint) * self.pixelsToMetersRatio).tolist()

        self.vectorYBMsg.data = (self.vectorYB/500.0).tolist()
        self.vectorYBtoBRMSg.data = ((self.vectorYB - self.vectorBR)/500.0).tolist()

        try:
            self.redCenterPub.publish(self.redMsg)
            self.blueCenterPub.publish(self.blueMsg)
            self.yellowCenterPub.publish(self.yellowMsg)
            self.greenCenterPub.publish(self.greenMsg)

            self.relativeRedPub.publish(self.relativeRedMSg)
            self.relativeBluePub.publish(self.relativeBlueMSg)
            self.relativeYellowPub.publish(self.relativeYellowMSg)

            self.meterRedPub.publish(self.meterRedMsg)
            self.meterBluePub.publish(self.meterBlueMsg)
            self.meterYellowPub.publish(self.meterYellowMsg)

            self.vectorYBPub.publish(self.vectorYBMsg)
            self.vectorYBtoBRPub.publish(self.vectorYBtoBRMSg)
        except CvBridgeError as e:
            print(e)

    def getCentre(self, img, mask):
        centre = self.getCentre1(img, mask)
        if centre.size == 0:
            return self.getCentre2(mask)
        return centre

    def getCentre2(self, mask):
        control = sum(sum(mask))
        if control < 10:
            return np.array([])
        M = cv2.moments(mask)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return np.array([cX, -cY])

    def getCentre1(self, img, mask):
        cimg = cv2.bitwise_and(img, img, mask = mask)
        gray = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
        #docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
        for i in range(10, 1, -1):
            circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,
                                       param1=40, param2=i, minRadius=0, maxRadius=20)
            if (circles is not None):
                circles = np.uint16(np.around(circles))
                return np.array([circles[0][0][0], -circles[0][0][1]])
            return np.array([])
            
    def findAllPoints(self, img):
        redMask = cv2.inRange(img, np.array([0, 0, 20]), np.array([20, 20, 255]))
        greenMask = cv2.inRange(img, np.array([0, 20, 0]), np.array([20, 255, 20]))
        blueMask = cv2.inRange(img, np.array([10, 0, 0]), np.array([255, 20, 20]))
        yellowMask = cv2.inRange(img, np.array([0, 10, 10]), np.array([0, 255, 255]))

        redCentre = self.getCentre(img, redMask)
        greenCentre = self.getCentre(img, greenMask)
        blueCentre = self.getCentre(img, blueMask)
        yellowCentre = self.getCentre(img, yellowMask)
        return redCentre, greenCentre, blueCentre, yellowCentre

    def callback1(self, data):
        # Receive the image
        try:
            self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        self.redC1, self.greenC1, self.blueC1, self.yellowC1 = self.findAllPoints(self.cv_image1)

    def callback2(self, data):
        # Receive the image
        try:
            self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        self.redC2, self.greenC2, self.blueC2, self.yellowC2 = self.findAllPoints(self.cv_image2)
        self.combinecenters()
        self.setPixelsToMetersRatio()
        self.determinejointangles()
        self.updateLastJoint()
        self.publishangles()
        self.publishCentresAndVectors()
    
    def callbackk(self, data1, data2):
        try:
            self.cv_image1 = self.bridge.imgmsg_to_cv2(data1, "bgr8")
            self.cv_image2 = self.bridge.imgmsg_to_cv2(data2, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.redC1, self.greenC1, self.blueC1, self.yellowC1 = self.findAllPoints(self.cv_image1)
        self.redC2, self.greenC2, self.blueC2, self.yellowC2 = self.findAllPoints(self.cv_image2)
        self.combinecenters()
        self.setPixelsToMetersRatio()
        self.determinejointangles()
        self.updateLastJoint()
        self.publishangles()
        self.publishCentresAndVectors()
        
    def rotate_axis_z(self,theta,vec):
            rotMat = np.array([[cos(theta),-sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])
            return np.dot(rotMat,vec)

    def rotate_axis_x(self,theta,vec):
            rotMat =np.array([[1, 0, 0], [0, cos(theta), -sin(theta)], [0, sin(theta), cos(theta)]])
            return np.dot(rotMat,vec)

    def projection(self,vect1, vect2):
        dot_prod = np.dot(vect1,vect2)
        sqrt_length = self.length_of_vector(vect2) * self.length_of_vector(vect2)
        amount_proj = dot_prod/sqrt_length
        return amount_proj * vect2

    def length_of_vector(self,vect):
            squared_sum = np.sum(vect ** 2)
            return np.sqrt(squared_sum)

    def determinejointangles(self):
        self.vectorYB = self.finalBlueCenter - self.finalYellowCenter
        self.vectorBR = self.finalRedCenter - self.finalBlueCenter

        joint_1 = np.arctan2(self.vectorYB[1],self.vectorYB[0])
        if (joint_1 < 0):
            joint_1 += 3*np.pi/2
        else:
            joint_1 -= 3*np.pi/2
        #rotation around x
        joint_3 = -(np.arctan2(self.vectorYB[2], self.vectorYB[1]) - np.pi/2)  
        if joint_3 > (np.pi)/2 :
            a = np.pi - joint_3
            b = (np.pi)/2 - a
            joint_3 = (np.pi)/2 - b
        elif joint_3 < -(np.pi)/2 :
            a = np.pi + joint_3
            b = (np.pi)/2 - a
            joint_3 = -(np.pi)/2 + b
        else :
            joint_3 = joint_3

        self.vectorYB = self.rotate_axis_x(-joint_3, self.vectorYB)
        proj = self.projection(self.vectorBR, self.vectorYB)
        if (self.length_of_vector(self.vectorBR + proj) >= self.length_of_vector(self.vectorBR)):
            joint_4 = self.vector_angles(self.vectorBR, proj)
        else :
            joint_4 = np.pi /2 - self.vector_angles(self.vectorBR, proj)
        
        self.joint1.data = joint_1
        self.joint3.data = joint_3
        self.joint4.data = joint_4

    def detectSwitch(self):
        if (abs(self.joint1.data - self.lastJoint1) > 1.6) and (self.iterationNumber > 30):
            print("Switching")
            self.joint1.data = -(np.pi - self.joint1.data) % 360
            return -1
        return 1

    def angleBound(self, jointAngle, limit, lastAngle):
        jointAngle = max(min(jointAngle, limit), -limit)
        if (abs(jointAngle - lastAngle) > 0.2) and (abs(jointAngle - lastAngle) <= 1.05) and (self.iterationNumber > 30):
            print("Limiting Angle Move")
            jointAngle += np.sign(jointAngle - lastAngle) * 0.05
        if (abs(jointAngle - lastAngle) > 1.05) and (self.iterationNumber > 10):
            print(jointAngle, lastAngle, self.iterationNumber)
            jointAngle = lastAngle
        return jointAngle

    def vector_angles(self,vect1,vect2):
            dot_vect1_vect2 = np.dot(vect1,vect2)
            length_vect1 = self.length_of_vector(vect1)
            length_vect2 = self.length_of_vector(vect2)
            return np.arccos(dot_vect1_vect2/(length_vect1 * length_vect2))
            

    def combinecenters(self):
        self.originPoint = self.originhandler(self.greenC1, self.greenC2)
        self.finalRedCenter = self.centercamfusion(self.redC1, self.redC2)
        self.finalBlueCenter = self.centercamfusion(self.blueC1, self.blueC2)
        self.finalYellowCenter = self.centercamfusion(self.yellowC1, self.yellowC2)

    def centercamfusion(self, campoint1, campoint2):
        if (campoint1.size == 0) and (campoint2.size == 0):
            print("Defaulting")
            return np.array([-1.0, -1.0, -1.0])
        elif campoint1.size == 0:
            print("Limited Campoint 1 visibility")
            return np.array([campoint2[0], self.originPoint[1], campoint2[1]])
        elif campoint2.size == 0:
            print("Limited Campoint 2 visibility")
            return np.array([self.originPoint[1], campoint1[0], campoint1[1]])
        elif (campoint1.size == 2) and (campoint2.size == 2):
            return np.array([campoint2[0], campoint1[0], (campoint2[1] + campoint1[1]) / 2])
        else:
            return np.array([-1.0, -1.0, -1.0])


    def originhandler(self, campoint1, campoint2):
        if (campoint1.size == 0) and (campoint2.size == 0):
            print("Defaulting")
            return np.array([-1.0, -1.0, -1.0])
        elif campoint1.size == 0:
            print("Limited Campoint 1 visibility")
            return np.array([campoint2[0], campoint2[0], campoint2[1]])
        elif campoint2.size == 0:
            print("Limited Campoint 2 visibility")
            return np.array([campoint1[0], campoint1[0], campoint1[1]])
        elif (campoint1.size == 2) and (campoint2.size == 2):
            return np.array([campoint2[0], campoint1[0], (campoint2[1] + campoint1[1]) / 2])
        else:
            return np.array([-1.0, -1.0, -1.0])

    def setPixelsToMetersRatio(self):
        self.pixelsToMetersRatio = 4.0 / np.linalg.norm(self.finalYellowCenter - self.originPoint)

    def updateLastJoint(self):
        self.iterationNumber += 1
        self.lastJoint1 = self.joint1.data
        self.lastJoint3 = self.joint3.data
        self.lastJoint4 = self.joint4.data

    def reverseJoint1(self, vectorBR):
        th = -self.joint1.data
        reverseMatrix = np.array([[np.cos(th), -np.sin(th), 0],
                                  [np.sin(th), np.cos(th), 0],
                                  [0, 0, 1]])
        return reverseMatrix.dot(vectorBR.T)

    def reverseJoint3(self, vectorBR):
        th = -self.joint3.data
        reverseMatrix = np.array([[1, 0, 0],
                                  [0, np.cos(th), -np.sin(th)],
                                  [0, np.sin(th), np.cos(th)]])
        return reverseMatrix.dot(vectorBR.T)
        

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
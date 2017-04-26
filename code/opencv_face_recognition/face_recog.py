#!/usr/bin/env python

import cv2, os, sys, numpy
import roslib
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

size = 2
(im_width, im_height) = (112, 92)
id = 0
names = {}

class face_recognition:

  def __init__(self):
    self.bridge = CvBridge()
    # creat a publisher
    self.name_pub = rospy.Publisher("/people_name", String, queue_size = 1)
    # creat a subscriber
    self.image_sub = rospy.Subscriber("/head_camera/rgb/image_raw", Image, self.callback)
    # creat a Recognizer
    self.model = cv2.face.createFisherFaceRecognizer()
    # load the trained model
    self.model.load("trained_model.yml")
    # creat a classifier
    self.classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    global id
    # load the name of different people
    for (subdirs, dirs, files) in os.walk('att_faces'): 
    	for subdir in dirs:
    		names[id] = subdir
    		id += 1

  def callback(self,data):
  	# covert ROS_image to cv_image
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    # get gray image
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # resize the image
    mini_gray_image = cv2.resize(gray_image, (int(gray_image.shape[1] / size), int(gray_image.shape[0] / size)))

    # detect faces in the sensor's view 
    faces = self.classifier.detectMultiScale(mini_gray_image)

    for f in faces:
    	(x, y, w, h) = [v * size for v in f]
    	new_face = gray_image[y:y + h, x:x + w]
    	resize_new_face = cv2.resize(new_face, (im_width, im_height))

    	prediction = self.model.predict(resize_new_face)

    	# draw a rectangle with the predictied face's name
    	cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0,255,0), thickness = 3)
    	cv2.putText(cv_image, '%s, %.0f' % (names[prediction[0]], prediction[1]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))

        self.name_pub.publish(names[prediction[0]])
    	
    cv2.imshow("OpenCV", cv_image)
    cv2.waitKey(10)

if __name__ == '__main__':
    face_recog = face_recognition()
    rospy.init_node('face_recognition', anonymous=True)
    rospy.spin()
    cv2.destroyAllWindows()

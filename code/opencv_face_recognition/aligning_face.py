#!/usr/bin/env python

import cv2, sys, numpy, os
import roslib
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

size = 4
fn_haar = 'haarcascade_frontalface_default.xml'
fn_dir = 'att_faces'


class image_converter:

  def __init__(self):
    print("\n\033[94mThe program will save 20 samples. \
        Move your head around to increase while it runs.\033[0m\n")
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/head_camera/rgb/image_raw",Image,self.callback)
    self.count = 0
    self.count_max = 20

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    try:
        fn_name = sys.argv[1]
    except:
        print("You must provide a name")
        sys.exit(0)

    path = os.path.join(fn_dir, fn_name)
    if not os.path.isdir(path):
        os.mkdir(path)
    (im_width, im_height) = (112, 92)
    haar_cascade = cv2.CascadeClassifier(fn_haar)

    # Generate name for image file
    pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path)
        if n[0]!='.' ]+[0])[-1] + 1
    
    pause = 0

    # Get image size
    height, width, channels = cv_image.shape
    
    # Flip frame
    #cv_image = cv2.flip(cv_image, 1, 0)
    
    # Convert to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Scale down for speed
    mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

    # Detect faces
    faces = haar_cascade.detectMultiScale(mini)

    # We only consider largest face
    faces = sorted(faces, key=lambda x: x[3])
    if faces and self.count < 20:
        
        face_i = faces[0]
        (x, y, w, h) = [v * size for v in face_i]
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))

        # Draw rectangle and write name
        cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(cv_image, fn_name, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,
        1,(0, 255, 0))

        # Remove false positives
        if(w * 6 < width or h * 6 < height):
            print("Face too small")
        else:
            # To create diversity, only save every fith detected image
            if(pause == 0):
                print("Saving training sample "+str(self.count+1)+"/"+str(self.count_max))

                # Save image file
                cv2.imwrite('%s/%s.png' % (path, pin), face_resize)

                pin += 1
                self.count += 1

                pause = 1

    if(pause > 0):
        pause = (pause + 1) % 5

    cv2.imshow('OpenCV', cv_image)
    key = cv2.waitKey(10)

def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

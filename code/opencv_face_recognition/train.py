#!/usr/bin/env python

import cv2, sys, numpy, os

size = 2
data_dir = 'att_faces'
images = []
lables = [] 
names = {}
id = 0

print("Start training.......")

# Find the path to folders containing the training data
for (subdirs, dirs, files) in os.walk(data_dir):

    # Loop through each folder 
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(data_dir, subdir)

        # Loop through each photo in the folder
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            lable = id

            # Add photo to training data
            images.append(cv2.imread(path, 0))
            lables.append(int(lable))
        id += 1

(im_width, im_height) = (112, 92)

# Create a Numpy array from the two lists above
(images, lables) = [numpy.array(lis) for lis in [images, lables]]

model = cv2.face.createFisherFaceRecognizer()
model.train(images, lables)
model.save("trained_model.yml")
print("Training is finished!!!!!")



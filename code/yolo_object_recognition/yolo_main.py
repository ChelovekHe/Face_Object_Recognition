#!/usr/bin/env python

import caffe
import cv2
import sys, getopt
import numpy as np

from datetime import datetime

# set GPU
caffe.set_mode_gpu()
caffe.set_device(0)

def interpret_output(output, img_width, img_height):
	result = []
	classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
	
	threshold = 0.2
	outline_threshold = 0.5

	probs = np.zeros((7, 7, 2, 20))
	class_probs = np.reshape(output[0:980], (7, 7, 20))
	scales = np.reshape(output[980:1078],(7, 7, 2))

	boxes = creat_prob_boxes(output, img_width, img_height)

	for i in range(2):
		for j in range(20):
			probs[:,:,i,j] = np.multiply(class_probs[:,:,j], scales[:,:,i])
	
	filter_mat_probs = np.array(probs>=threshold,dtype='bool')
	filter_mat_boxes = np.nonzero(filter_mat_probs)
	
	filter_box = boxes[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
	filter_prob = probs[filter_mat_probs]
	classes_num_filtered = np.argmax(probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]

	argsort = np.array(np.argsort(filter_prob))[::-1]
	filter_box = filter_box[argsort]
	filter_prob = filter_prob[argsort]
	classes_num_filtered = classes_num_filtered[argsort]
		
	for i in range(len(filter_box)):
		if filter_prob[i] == 0 : continue
		for j in range(i+1,len(filter_box)):
			if outline(filter_box[i],filter_box[j]) > outline_threshold : 
				filter_prob[j] = 0.0
		
	filter_outline = np.array(filter_prob>0.0,dtype='bool')
	filter_box = filter_box[filter_outline]
	filter_prob = filter_prob[filter_outline]
	classes_num_filtered = classes_num_filtered[filter_outline]

	for i in range(len(filter_box)):
		result.append([classes[classes_num_filtered[i]],filter_box[i][0],filter_box[i][1],filter_box[i][2],filter_box[i][3],filter_prob[i]])

	return result

def creat_prob_boxes(output, img_width, img_height):
	
	offset = np.transpose(np.reshape(np.array([np.arange(7)] * 14),(2, 7, 7)), (1, 2, 0))
	boxes = np.reshape(output[1078:],(7, 7, 2, 4))

	boxes[:,:,:,0] += offset
	boxes[:,:,:,1] += np.transpose(offset,(1,0,2))
	boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / 7.0
	boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2],boxes[:,:,:,2])
	boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3],boxes[:,:,:,3])
		
	boxes[:,:,:,0] *= img_width
	boxes[:,:,:,1] *= img_height
	boxes[:,:,:,2] *= img_width
	boxes[:,:,:,3] *= img_height
	return boxes


def outline(box1, box2):
	h = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
	w = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
	if h < 0 or w < 0 : intersection = 0
	else : intersection =  h * w
	return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)


def show_results(name, img, results, img_width, img_height):
	box_color = {"aeroplane": (255,255,255), 
				 "bicycle":(255,255,255), 
				 "bird":(255,255,255), 
				 "boat":(255,255,255), 
				 "bottle":(0,255,0), 
				 "bus":(255,255,255), 
				 "car":(255,255,255), 
				 "cat":(255,255,255), 
				 "chair":(50,50,0), 
				 "cow":(255,255,255), 
				 "diningtable": (10,50,255), 
				 "dog":(255,255,255), 
				 "horse":(255,255,255), 
				 "motorbike":(255,255,255), 
				 "person":(0,0,255), 
				 "pottedplant":(255,255,255), 
				 "sheep":(255,255,255), 
				 "sofa":(255,255,255), 
				 "train":(255,255,255),
				 "tvmonitor":(10,0,0)}
	
	for i in range(len(results)):
		x = int(results[i][1])
		y = int(results[i][2])
		w = int(results[i][3])//2
		h = int(results[i][4])//2
		
		if x - w < 0:
			xmin = 0
		else:
			xmin = x - w
		if y - h < 0:
			ymin = 0
		else:
			ymin = y - h
		if x + w > img_width:
			xmax = img_width
		else :
			xmax = x + w
		if y + h > img_height:
			ymax = img_height
		else:
			ymax = y + h
		
		cv2.rectangle(img,(xmin,ymin),(xmax,ymax),box_color[results[i][0]],3)
		cv2.rectangle(img,(xmin,ymin-20),(xmax,ymin),box_color[results[i][0]],-1)
		cv2.putText(img,results[i][0] + ' : %.2f' % results[i][5],(xmin+5,ymin-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)	

	cv2.imshow('Reognition Result',img)
	cv2.waitKey(10000)

if __name__=='__main__':	
	path = sys.argv[1:]
	opts, file_names = getopt.getopt(path, "hm:w:i:")
	for opt, name in opts:
		if opt == "-m":
			model_filename = name
		if opt == "-w":
			weight_filename = name
		if opt == "-i":
			img_filename = name

	# load model
	model = caffe.Net(model_filename, weight_filename, caffe.TEST)

	# load input image
	image = caffe.io.load_image(img_filename) 

	# build a transformer
	transf = caffe.io.Transformer({'data': model.blobs['data'].data.shape})
	transf.set_transpose('data', (2,0,1))
	
	# start to detect and recognize
	output = model.forward_all(data=np.asarray([transf.preprocess('data', image)]))
	
	# convert RGB to BGR
	cv_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

	# obtain the recognition result 
	results = interpret_output(output['result'][0], image.shape[1], image.shape[0])
	show_results(img_filename,cv_image,results, image.shape[1], image.shape[0])


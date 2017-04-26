#/usr/bin/env python

import caffe
import numpy as np
import sys
import caffe
import matplotlib.pyplot as plt
import os

def prediction(img, labels, model, deploy):

	#load the model
	#caffe.set_mode_cpu()
	caffe.set_mode_gpu()
	caffe.set_device(0)
	net = caffe.Net(deploy, model, caffe.TEST)

	mean_file = root + '/data/imagenet_mean.npy'  
	mu = np.load(mean_file).mean(1).mean(1)
	 
	# print 'mean-subtracted values:', zip('BGR', mu)

	# create transformer for the input called 'data'
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
	transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
	transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
	transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR


	# set the size of the input (we can skip this if we're happy
	#  with the default; we can also change it later, e.g., for different batch sizes)
	#net.blobs['data'].reshape(50,        # batch size
	#                          3,         # 3-channel (BGR) images
	#                          227, 227)  # image size is 227x227


	image = caffe.io.load_image(img)
	transformed_image = transformer.preprocess('data', image)
	#show image in ipyhon
	# plt.imshow(image)

	#classification
	net.blobs['data'].data[...] = transformed_image
	out = net.forward()
	prob = out['Softmax1'][0]
	return(prob.argmax())


def writeFile(caffeModel, deploy, labelsFile, fileName):

	labels = np.loadtxt(labelsFile, str, delimiter=' ')
 	count = 1
 	res = []

 	output = open(fileName, "w")
 	output.write('prediction_result\n')
 	output.write('actualClass\tpredictClass\n')

 	for l in labels:
 		image = imageDir + l[0]
		pred = prediction(image, labelsFile, caffeModel, deploy)
		output.write(str(l[1])+"\t"+str(pred)+"\n")
		if l[1] == pred:
			count += 1
	acc = count * 1.0 / 41
	output.write("accuracy rate is:" + str(acc))
	output.close



if __name__ == "__main__":

	root='/home/xiao/Desktop/caffe_face'  
	deploy1=root + '/example/deploy_3000_0.01.prototxt'
	deploy2=root + '/example/deploy_3000_0.001.prototxt'
	deploy3=root + '/example/deploy_1500_0.0005.prototxt'  
	caffeModel1=root + '/example/model/Lenet_3000_0.01_iter_3000.caffemodel'  
	caffeModel2=root + '/example/model/Lenet_3000_0.001_iter_3000.caffemodel'  
	caffeModel3=root + '/example/model/Lenet_1500_0.0005_iter_1500.caffemodel'  
	imageDir=root+'/data/images/'
	labelsFile = root + '/data/test.txt'  
 	
	writeFile(caffeModel1, deploy1, labelsFile, 'Lenet_3000_0.01_prdict.txt')
	writeFile(caffeModel2, deploy2, labelsFile, 'Lenet_3000_0.001_prdict.txt')
	writeFile(caffeModel3, deploy3, labelsFile, 'Lenet_1500_0.0005_prdict.txt')
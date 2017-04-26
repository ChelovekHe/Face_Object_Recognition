#!/usr/bin/env python
import caffe
import numpy as np

MEAN_PROTO_PATH = 'imagenet_mean.binaryproto'  #input file path
MEAN_NPY_PATH = 'imagenet_mean.npy'                         # output file path

blob = caffe.proto.caffe_pb2.BlobProto()           # create protobuf blob
data = open(MEAN_PROTO_PATH, 'rb' ).read()         # read input file
blob.ParseFromString(data)                         # parse input file into blob
# change mean in blob into numpy format, array[(mean_number, channel, hight, width)]
array = np.array(caffe.io.blobproto_to_array(blob))
mean_npy = array[0]            
np.save(MEAN_NPY_PATH ,mean_npy)

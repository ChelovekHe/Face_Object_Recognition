#!/usr/bin/env python
import caffe
from caffe import layers as L, params as P, proto, to_proto
path='/home/xiao/Desktop/caffe_face/data/'                    #path of new data and configuration file
train_lmdb=path+'train_lmdb'                #location of train LMDB files 
val_lmdb=path+'val_lmdb'                    #location of validation LMDB files
mean_file=path+'imagenet_mean.binaryproto'         #location of mean file 
train_proto=path+'train.prototxt'         #location of training configulation file
val_proto=path+'val.prototxt'             #location of validation configulation file

#create net
def lenet(lmdb, batch_size, include_acc=False):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)
    
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'),bias_filler=dict(type='constant'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'),bias_filler=dict(type='constant'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'),bias_filler=dict(type='constant'))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.score = L.InnerProduct(n.relu1, num_output=42, weight_filler=dict(type='xavier'),bias_filler=dict(type= 'constant'))
    # n.loss =  L.SoftmaxWithLoss(n.score, n.label)
    
    if include_acc:
	n.acc = L.Accuracy(n.score, n.label)
	n.loss = L.SoftmaxWithLoss(n.score, n.label)
	return n.to_proto()
    else:
	n.loss = L.SoftmaxWithLoss(n.score, n.label)
        return n.to_proto()
    
with open('/home/xiao/Desktop/caffe_face/example/lenet_auto_train.prototxt', 'w') as f:
    f.write(str(lenet(train_lmdb, 41, include_acc=False)))
    
with open('/home/xiao/Desktop/caffe_face/example/lenet_auto_test.prototxt', 'w') as f:
    f.write(str(lenet(val_lmdb, 41, include_acc=True)))




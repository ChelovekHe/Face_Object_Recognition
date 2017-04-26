#!/usr/bin/env python
import caffe
from caffe import layers as L, params as P, proto, to_proto
path='/home/xiao/Desktop/caffe_face/data/'                    #path of new data and configuration file
train_lmdb=path+'train_lmdb'                #location of train LMDB files 
val_lmdb=path+'val_lmdb'                    #location of validation LMDB files
mean_file=path+'imagenet_mean.binaryproto'         #location of mean file 
train_proto=path+'trainAlexNet.prototxt'         #location of training configulation file
val_proto=path+'valAlexNet.prototxt'             #location of validation configulation file

#create net
def lenet(lmdb, batch_size, include_acc=False):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)
    
    n.conv1 = L.Convolution(n.data, kernel_size=11, num_output=96, stride=4, weight_filler=dict(type='gaussian', std=0.01),bias_filler=dict(type='constant', value=0))
    n.relu1 = L.ReLU(n.conv1, in_place=True)
    n.norm1 = L.LRN(n.conv1, local_size=5, alpha=1e-4, beta=0.75)
    n.pool1 = L.Pooling(n.norm1, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, group=2, num_output=256, pad=2, weight_filler=dict(type='gaussian', std=0.01),bias_filler=dict(type='constant', value=0.1))
    n.relu2 = L.ReLU(n.conv2, in_place=True)
    n.norm2 = L.LRN(n.conv2, local_size=5, alpha=1e-4, beta=0.75)
    n.pool2 = L.Pooling(n.norm2, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    n.conv3 = L.Convolution(n.pool2, kernel_size=3, num_output=384, pad=2, weight_filler=dict(type='gaussian', std=0.01),bias_filler=dict(type='constant', value=0))
    n.relu3 = L.ReLU(n.conv3, in_place=True)
    n.conv4 = L.Convolution(n.conv3, kernel_size=3, group=2, num_output=384, pad=1, weight_filler=dict(type='gaussian', std=0.01),bias_filler=dict(type='constant', value=0))
    n.relu4 = L.ReLU(n.conv4, in_place=True)
    n.conv5 = L.Convolution(n.conv4, kernel_size=3, group=2, num_output=256, pad=1, weight_filler=dict(type='gaussian', std=0.01),bias_filler=dict(type='constant', value=0.1))
    n.relu5 = L.ReLU(n.conv5, in_place=True)
    n.pool5 = L.Pooling(n.conv5, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    n.fc6   = L.InnerProduct(n.pool5, num_output=4096, weight_filler=dict(type='gaussian', std=0.005),bias_filler=dict(type='constant', value=0.1))
    n.relu6 = L.ReLU(n.fc6, in_place=True)
    n.drop6 = L.Dropout(n.fc6, in_place=True)
    n.fc7   = L.InnerProduct(n.fc6, num_output=4096, weight_filler=dict(type='gaussian', std=0.005),bias_filler=dict(type='constant', value=0.1))
    n.relu7 = L.ReLU(n.fc7, in_place=True)
    n.drop6 = L.Dropout(n.fc7, in_place=True)
    n.fc8   = L.InnerProduct(n.fc7, num_output=42, weight_filler=dict(type='gaussian', std=0.005),bias_filler=dict(type='constant', value=0))
     
    if include_acc:
	n.acc = L.Accuracy(n.fc8, n.label)
	n.loss = L.SoftmaxWithLoss(n.fc8, n.label)
	return n.to_proto()
    else:
	n.loss = L.SoftmaxWithLoss(n.fc8, n.label)
        return n.to_proto()
    
with open('/home/xiao/Desktop/caffe_face/example/lenet_auto_train.prototxt', 'w') as f:
    f.write(str(lenet(train_lmdb, 41, include_acc=False)))
    
with open('/home/xiao/Desktop/caffe_face/example/lenet_auto_test.prototxt', 'w') as f:
    f.write(str(lenet(val_lmdb, 41, include_acc=True)))




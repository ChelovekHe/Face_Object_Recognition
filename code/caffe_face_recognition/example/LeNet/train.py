#/usr/bin/env python

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()
#caffe.set_mode_cpu()
solver = caffe.SGDSolver('/home/xiao/Desktop/caffe_face/example/solver.prototxt')
solver.solve()

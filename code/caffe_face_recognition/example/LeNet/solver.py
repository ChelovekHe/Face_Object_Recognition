#/usr/bin/env python

path='/home/xiao/Desktop/caffe_face/example/'
solver_file=path+'solver.prototxt'     #location to save solver

sp={}
sp['train_net']='"' + path + 'lenet_auto_train.prototxt"'  # training config file
sp['test_net']='"' + path + 'lenet_auto_test.prototxt"'     # testing config file
sp['test_iter']='1'                  # number of iter
sp['test_interval']='300'              # number of interval
sp['base_lr']='0.0005'                  # basic learning rate
sp['display']='20'                    # interval of display
sp['max_iter']='1500'                 # maximumn of iter
sp['lr_policy']='"step"'                 # policy of learning rate
sp['gamma']='0.0001'                      # gamme
sp['momentum']='0.9'                   # momentum
sp['weight_decay']='0.0005'            # decay of weight
sp['stepsize'] = '2000'
sp['snapshot']='3000'                   # the interval of save model
sp['snapshot_prefix']='"Lenet_1500_0.0005"'       # prefix of model
sp['solver_mode']='GPU'                # whether to use gpu 
sp['solver_type']='SGD'                # optimal algorithm

def write_solver():
    #write files
    with open(solver_file, 'w') as f:
        for key, value in sorted(sp.items()):
            if not(type(value) is str):
                raise TypeError('All solver parameters must be strings')
            f.write('%s: %s\n' % (key, value))
if __name__ == '__main__':
    write_solver()

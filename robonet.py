# Create a new Network to train and test hand-raising database
# April 6th, 2015
# one package: robonet.py | robonet_label.txt | robonet_deploy.prototxt

import numpy as np
from PIL import Image
from pylab import *
import tempfile
import os

import sys
project_root = '/home/edward/py-faster-rcnn/'
caffe_root = '/home/edward/py-faster-rcnn/caffe-fast-rcnn/'
sys.path.insert(0, caffe_root + 'python')

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
caffe.set_device(0)
caffe.set_mode_gpu()

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=0, weight_filler=dict(type='gaussian', std=0.01),
              bias_filler=dict(type='constant', value=0.1)):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout, param=0, weight_filler=dict(type='gaussian', std=0.005),
            bias_filler=dict(type='constant', value=0.1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def caffenet(data, label=None, train=True, num_classes=2, classifier_name='fc8', learn_all=False):
    #define network by our own. We follow the template by caffenet though, since we will
    #transfer learning the pre-trained weights later on. But caffenet by now has no idea
    #that we will do so.
    n = caffe.NetSpec()
    n.data = data

    weight_param = dict(lr_mult=1, decay_mult=1)
    bias_param = dict(lr_mult=2, decay_mult=0)
    learned_param = [weight_param, bias_param]
    frozen_param = [dict(lr_mult=0)] * 2 #lr_mult = 0 means layers won't be trained
    param = learned_param if learn_all else frozen_param

    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, param=param)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)

    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2, param=param)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)

    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, param=param)

    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, param=param)

    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, param=param)
    n.pool5 = max_pool(n.relu5, 3, stride=2)

    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=param)
    if train:
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
    else:
        fc7input = n.relu6

    n.fc7, n.relu7 = fc_relu(fc7input, 4096, param=param)
    if train:
        n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)
    else:
        fc8input = n.relu7

    #always learn fc8 (param=learned_param, the other layers param=param
    #set to learned_param only if learn_all=True)
    fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)
    #give fc8 the name specified by argument 'classifier_name'
    n.__setattr__(classifier_name, fc8)
    if not train:
        n.probs = L.Softmax(fc8)
    if label is not None:
        n.label = label
        n.loss = L.SoftmaxWithLoss(fc8, n.label)
        n.acc = L.Accuracy(fc8, n.label)

    # write the net to a temporary file and return its filename
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(n.to_proto()))
        return f.name
    #figure out how to write this to a permanent .protottxt file to be accessed later   
    #BUT TRAINING PROTOTXT AND DEPLOY PROTOTXT ARE DIFFERENT (NO TRAINING DATA AND LABEL)

def solver(train_net_path, test_net_path=None, base_lr=0.001):
    s = caffe_pb2.SolverParameter()

    #Specify locations of train and test networks
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = 1000 # Test after very 1000 training iterations
        s.test_iter.append(100) # Test on 100 batches each time we test

    # The number of iterations over which to average the gradient
    s.iter_size = 1

    s.max_iter = 100000     #of times to update the net (training iterations)

    # Solve using SGD algorithm (other choics are Adam and RMSProp)
    s.type = 'SGD'

    # Set initial learning rate for SGD
    s.base_lr = base_lr

    # Set 'lr_policy' to define how learning rate changes during training
    # step the learning rate by multiplying with factor 'gamma'
    # every 'stepsize' iterations
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 20000

    # Set other SGD hyperparameters. 'Momentum' takes weighted average of current and
    # previous gradients -> more stable. 'Weight decay' regularizes learning, prevent overfitting
    s.momentum = 0.9
    s.weight_decay = 5e-4

    # Display the current training loss and accuracy every 1000 iterations
    s.display = 1000

    # Snapshots are files used to store networks we've trained. We'll snapshot
    # every 10k iterations - ten times during training
    s.snapshot = 10000
    s.snapshot_prefix = caffe_root + 'models/finetune_flickr_style/finetune_flickr_style'

    # Train on the GPU. Using CPU to train large networks is very slow
    s.solver_mode = caffe_pb2.SolverParameter.CPU
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    # Write the solver to a temporary file and return its filename
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(s))
        return f.name

#function to train the network with Solver.
#it's also possible to run the train command from bash
def run_solvers(niter, solvers, disp_interval=10):
    """Run solvers for niter iterations,
    returning the loss and accuracy recorded each iteration."""
    blobs = ('loss', 'acc')
    
    #initiate array of zeros with the number of niter corresponding to
    #number of columns
    loss, acc = (np.zeros(niter) for _ in blobs)

    for it in range(niter):
        solvers.step(1) # run a single SGD step in Caffe
        loss[it], acc[it] = (solvers.net.blobs[b].data.copy() for b in blobs)
        
        #print loss and accuracy every interval
        if it % disp_interval == 0 or it + 1 == niter:
		loss_disp = ''.join('loss=%.3f, acc=%2d%%' % (loss[it], np.round(100*acc[it])))
		print '%3d) %s' % (it, loss_disp)

    # Save the learned weights from both nets
    weight_dir = tempfile.mkdtemp()
    filename = os.path.join(weight_dir, 'robonet_weights.caffemodel')
    solvers.net.save(filename)
    
    # added manually to save file to permanent directory
   # permanent_dir = project_root + 'workspace/raise.jpg'
   # permanent_filename = os.path.join(weight_dir, 'robonet_weights.caffemodel')
   # solvers.net.save(permanent_filename)   
    #NOT SUCCESSFUL YET

    return loss, acc, filename


if __name__ == '__main__':
    
    #####load image training database, using ImageData layer (not lmdb or hdf5)
    #load source file name
    train_source = project_root + 'workspace/raisehand/train.txt' #make sure this data exist
    #load mean and cropping
    transform_train = dict(mirror=True, crop_size=227, mean_file = caffe_root + 'data/ilsvrc12/imagenet_mean.binaryproto')
    train_data, train_label = L.ImageData(transform_param=transform_train, source = train_source, batch_size=10, new_height=256, new_width=256, ntop=2) 

    #####create network template with prototxt, load image into data layers
    output_classes = 2
    classifier_name = 'fc_raise'
    learn_all = False
    train = True
    robonet_proto = caffenet(data=train_data, label=train_label, train=train, num_classes=output_classes, classifier_name=classifier_name, learn_all=learn_all)

    #####create solver, import pre-trained weights 
    robonet_solver_filename = solver(robonet_proto)
    robonet_solver = caffe.get_solver(robonet_solver_filename)

    pretrained_weights_name = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    assert os.path.exists(pretrained_weights_name)
    robonet_solver.net.copy_from(pretrained_weights_name)

    #####run solver, train network and save weights in caffemodel
    niter = 40
    train_loss, train_acc, robonet_weights = run_solvers(niter, robonet_solver)
    print 'Running solvers for %d iterations...' % niter

    #figure out prototxt, image preprocessing
    #####create test network using the pre-trained weights
    #####and test with new image database (use Transformer, not ImageData layer)
	
    #generate deploy prototxt manually, because the one in above code
    #is for training (has data and label). Robonet has the same structure with caffenet,
    #so we can borrow deploy txt (just change last layer)
	
    robonet_prototxt = project_root + 'workspace/robonet_deploy.prototxt' #manually made based on caffenet
    robonet_test = caffe.Net(robonet_prototxt, robonet_weights, caffe.TEST)

    # load input and configure preprocessing
    transformer = caffe.io.Transformer({'data': robonet_test.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)

    #note we can change the batch size on-the-fly
    #since we classify only one image, we change batch size from 10 to 1
    robonet_test.blobs['data'].reshape(1,3,227,227)


    #load the image in the data layer
    test_image = project_root + 'workspace/raisehand.jpg'
    im = caffe.io.load_image(test_image)
    robonet_test.blobs['data'].data[...] = transformer.preprocess('data', im)

    #compute
    out = robonet_test.forward()

    #predicted predicted class
    print out['prob'].argmax()

    #print predicted labels
    robonet_label = project_root + 'workspace/robonet_label.txt'
    labels = np.loadtxt(robonet_label, str, delimiter='\t')
    top_k = robonet_test.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
    print labels[top_k]

        #load the image in the data layer
    test_image = project_root + 'workspace/normal.jpg'
    im = caffe.io.load_image(test_image)
    robonet_test.blobs['data'].data[...] = transformer.preprocess('data', im)

    #compute
    out = robonet_test.forward()

    #predicted predicted class
    print out['prob'].argmax()

    #print predicted labels
    robonet_label = project_root + 'workspace/robonet_label.txt'
    labels = np.loadtxt(robonet_label, str, delimiter='\t')
    top_k = robonet_test.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
    print labels[top_k]

# Fine-Tuning Tutorial from https://github.com/BVLC/caffe/blob/master/examples/00-classification.ipynb
# March 14th, 2016
# Reproduced by Edward Elson

caffe_root = '/home/edward/py-faster-rcnn/caffe-fast-rcnn/'
project_root = '/home/edward/py-faster-rcnn/'

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

# caffe.set_device(0)
caffe.set_mode_gpu()
#caffe.set_mode_cpu()

import numpy as np
from pylab import *
import tempfile

########Defining and running the Nets
from caffe import layers as L
from caffe import params as P

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]
frozen_param = [dict(lr_mult=0)] * 2

#########Defining Solver to train the Nets
from caffe.proto import caffe_pb2

# helper function for deprocessing preprocessed images
def deprocess_net_image(image):
    image = image.copy() #don't modify destructively (?)
    image = image[::-1] # BGR -> RGB
    image = image.transpose(1, 2, 0) # CHW -> HWC
    image += [123, 117, 104] # (approximately) undo mean subtraction

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    #round and cast from float3 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image

#EDITED, NO CHANGE
def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param,
              weight_filler=dict(type='gaussian', std=0.01),
              bias_filler=dict(type='constant', value=0.1)):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)

#EDITED, NO CHANGE
def fc_relu(bottom, nout, param=learned_param,
            weight_filler=dict(type='gaussian', std=0.005),
            bias_filler=dict(type='constant', value=0.1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)

#EDITED, NO CHANGE
def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

#EDITED, NO CHANGE
def caffenet(data, label=None, train=True, num_classes=1000,
             classifier_name='fc8', learn_all=False):
    """Returns a NetSpec specifying CaffeNet, following the original proto text
       specification (./models/bvlc_reference_caffenet/train_val.prototxt)."""
    n = caffe.NetSpec()

    n.data = data
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

#EDITED
def style_net(train=True, learn_all=False, subset=None):
    if subset is None:
        subset = 'train' if train else 'new_test'
    source = project_root + 'workspace/raisehand/%s.txt' % subset
    transform_param = dict(mirror=train, crop_size=227,
        mean_file=caffe_root + 'data/ilsvrc12/imagenet_mean.binaryproto')
    style_data, style_label = L.ImageData(
        transform_param=transform_param, source=source,
        batch_size=25, new_height=256, new_width=256, ntop=2)
    return caffenet(data=style_data, label=style_label, train=train,
                    num_classes=NUM_STYLE_LABELS,
                    classifier_name='fc8_flickr',
                    learn_all=learn_all)

def disp_preds(net, image, labels, k=10, name='ImageNet'):
    input_blob = net.blobs['data']
    net.blobs['data'].data[0, ...] = image
    probs = net.forward(start='conv1')['probs'][0]
    top_k = (-probs).argsort()[:k]
    print 'top %d predicted %s labels =' % (k, name)
    print '\n'.join('\t(%d) %5.2f%% %s' % (i+1, 100*probs[p], labels[p])
                    for i, p in enumerate(top_k))

def disp_imagenet_preds(net, image):
    disp_preds(net, image, imagenet_labels, name='ImageNet')

def disp_style_preds(net, image):
    disp_preds(net, image, style_labels, name='style')

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
    s.base_lr = base_lr*0.1*0.1

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
    returning the loss and accuracy recorded each iteration.
    'solvers' is a list of (name, solver) tuples."""
    blobs = ('loss', 'acc')
    loss, acc = ({name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs)
    for it in range(niter):
        for name, s in solvers:
            s.step(1) # run a single SGD step in Caffe
            loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy()
                                             for b in blobs)
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = '; '.join('%s: loss=%.3f, acc=%2d%%' %
                                  (n, loss[n][it], np.round(100*acc[n][it]))
                                  for n, _ in solvers)
            print '%3d) %s' % (it, loss_disp)

    # Save the learned weights from both nets
    weight_dir = tempfile.mkdtemp()
    weights = {}
    for name, s in solvers:
        filename = 'weights.%s.caffemodel' % name
        weights[name] = os.path.join(weight_dir, filename)
        s.net.save(weights[name])
    return loss, acc, weights

def eval_style_net(weights, test_iters=10):
    test_net = caffe.Net(style_net(train=False), weights, caffe.TEST)
    accuracy = 0
    for it in xrange(test_iters):
        accuracy += test_net.forward()['acc']
    accuracy /= test_iters
    return test_net, accuracy


if __name__ == '__main__':

    # ##########Image Database Preparation

    NUM_STYLE_IMAGES = 77 #number of images for training
    NUM_STYLE_LABELS = 2

    #define path to ImageNet pretrained weights
    import os
    weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    assert os.path.exists(weights)


    #load 1000 ImageNet labels
    imagenet_label_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
    imagenet_labels = list(np.loadtxt(imagenet_label_file, str, delimiter ='\t'))
    assert len(imagenet_labels) == 1000
    print 'Loaded ImageNet labels:\n', '\n'.join(imagenet_labels[:10] + ['...'])


    #load Hand-Raising style labels to style_labels
    style_label_file = project_root + 'workspace/raisehand/style_names.txt'
    style_labels = list(np.loadtxt(style_label_file, str, delimiter='\n'))
    if NUM_STYLE_LABELS > 0:
        style_labels = style_labels[:NUM_STYLE_LABELS]
    print '\nLoaded style labels:\n', ', '.join(style_labels)


#    dummy_data = L.DummyData(shape=dict(dim=[1, 3, 227, 227]))
    # load pretrained network
  #  imagenet_net_filename = caffenet(data=dummy_data, train=False)
  #  imagenet_net = caffe.Net(imagenet_net_filename, weights, caffe.TEST)

    # create an instance of CaffeNet network with input data from Flickr
    # and weights from pretrained ImageNet model _> via style_net
    # untrained_style_net = caffe.Net(style_net(train=False, subset='train'),
    #                                 weights, caffe.TEST)
    # untrained_style_net.forward()

#     style_data_batch = untrained_style_net.blobs['data'].data.copy()
#     style_label_batch = np.array(untrained_style_net.blobs['label'].data, dtype=np.int32)

#      #try pre-trained network on our Flickr image
#     batch_index = 1
#     image = style_data_batch[batch_index]
#     plt.imshow(deprocess_net_image(image))
# #    plt.show() #in the original code this line was not written
#     print 'actual label', style_labels[style_label_batch[batch_index]]

   #  disp_imagenet_preds(imagenet_net, image)
    # disp_style_preds(untrained_style_net, image)

     # just to make sure that the original and style_net are similar (except last layer)
 #    diff = untrained_style_net.blobs['fc7'].data[0] - imagenet_net.blobs['fc7'].data[0]
#     error = (diff ** 2).sum()
#     assert error < 1e-8

#     # delete untrained network to save memory
 #   del untrained_style_net


#     ######## Create an instance of Solver and train two networks
#     ######## one from scratch, the other based on ImageNet
#     ######## Train only the last layer

#     # niter = 200 # number of training iterations
    niter = 300

    # Reset style_solver as before
    style_solver_filename = solver(style_net(train=True))
    style_solver = caffe.get_solver(style_solver_filename)
    style_solver.net.copy_from(weights)

# """
#     # For reference, we also create a solver that isn't initialized from
#     # the pretrained ImageNet weights
#     scratch_style_solver_filename = solver(style_net(train=True))
#     scratch_style_solver = caffe.get_solver(scratch_style_solver_filename)
# """

    print 'Running solvers for %d iterations...' % niter
    # solvers = [('pretrained', style_solver),
    #            ('scratch', scratch_style_solver)]
    solvers = [('pretrained', style_solver)]
    loss, acc, weights = run_solvers(niter, solvers)
    print 'Done.'

    # train_loss, scratch_train_loss = loss['pretrained'], loss['scratch']
    # train_acc, scratch_train_acc = acc['pretrained'], acc['scratch']
    # style_weights, scratch_style_weights = weights['pretrained'], weights['scratch']

    train_loss = loss['pretrained']
    train_acc = acc['pretrained']
    style_weights = weights['pretrained']

    # Evaluate performance
    test_net, accuracy = eval_style_net(style_weights)
    print 'Accuracy, trained from ImageNet initialization: %3.1f%%' % (100*accuracy, )
# """
#  	scratch_test_net, scratch_accuracy = eval_style_net(scratch_style_weights)
#     print 'Accuracy, trained from random initialization: %3.1f%%'% (100*scratch_accuracy, )
# """
#     # Delete solvers to save memory
# """    del style_solver, scratch_style_solver, solvers"""
#     del style_solver, solvers

# """
#     ######## Perform end to end training, now train all layer
#     ######## using updated weights from only last layer training
#     end_to_end_net = style_net(train=True, learn_all=True)

#     #set base_lr to 1e-3, same with last time
#     #if learning diverges (loss gets larger), decrease base_lr (to 1e-4, 1e-5 etc...)
#     #until learning no longeryl
#     base_lr = 0.001

#     style_solver_filename = solver(end_to_end_net, base_lr=base_lr)
#     style_solver = caffe.get_solver(style_solver_filename)
#     style_solver.net.copy_from(style_weights)

#     scratch_style_solver_filename = solver(end_to_end_net, base_lr=base_lr)
#     scratch_style_solver = caffe.get_solver(scratch_style_solver_filename)
#     scratch_style_solver.net.copy_from(scratch_style_weights)

#     print 'Running solvers for %d iterations...' % niter
#     solvers = [('pretrained, end-to-end', style_solver),
#                ('scratch, end-to-end', scratch_style_solver)]
#     _, _, finetuned_weights = run_solvers(niter, solvers)
#     print 'Done.'

#     style_weights_ft = finetuned_weights['pretrained, end-to-end']
#     scratch_style_weights_ft = finetuned_weights['scratch, end-to-end']

#     # Delete solvers to save memory
#     del style_solver, scratch_style_solver, solvers

#     # Evaluate Performance
#     test_net, accuracy = eval_style_net(style_weights_ft)
#     print 'Accuracy, finetuned from ImageNet initialization: %3.1f%%' % (100*accuracy, )
#     scratch_test_net, scratch_accuracy = eval_style_net(scratch_style_weights_ft)
#     print 'Accuracy, finetuned from   random initialization: %3.1f%%' % (100*scratch_accuracy, )
# """
#     # Analyze first image sample
#     plt.imshow(deprocess_net_image(image))
#     plt.show()
#     disp_style_preds(test_net, image)

    # Test with another image
 #   for batch_index in range(1,8):
#	    image = test_net.blobs['data'].data[batch_index]
#	    plt.imshow(deprocess_net_image(image))
#	    plt.show()
#	    print 'actual label =', style_labels[int(test_net.blobs['label'].data[batch_index])]
#	    disp_style_preds(test_net, image)

    # batch_index = 2
    # image = test_net.blobs['data'].data[batch_index]
    # plt.imshow(deprocess_net_image(image))
    # plt.show()
    # print 'actual label =', style_labels[int(test_net.blobs['label'].data[batch_index])]
    # disp_style_preds(test_net, image)

    # batch_index = 1
    # image = test_net.blobs['data'].data[batch_index]
    # plt.imshow(deprocess_net_image(image))
    # plt.show()
    # print 'actual label =', style_labels[int(test_net.blobs['label'].data[batch_index])]
    # disp_style_preds(test_net, image)

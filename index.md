# INTRODUCTION
These are the works that I did during my research internship project in the [RoboTutor](https://robotutor.weblog.tudelft.nl/) team of [TU Delft Robotics Institute](http://robotics.tudelft.nl/). RoboTutor is a project that aims to use a Nao Robot to teach Robotics to students of schools in the Netherlands. Below is an image of RoboTutor in action, taken last year.  
  
![RoboTutor in Action](https://dl.dropboxusercontent.com/s/r1r5uqcewqvuvn4/robotutor.jpg?dl=0)  
  
The main goal of my project is to detect hand-raising gestures in a video, and feed the coordinates of such gesture to the Nao Robot central processor. In the process, I tried out different methods as listed below:     
   
### (1) Running Faster R-CNN algorithm in Caffe ([demo_robotutor.py](https://github.com/edwardelson/RoboTutor/blob/master/demo_robotutor.py))   
During this period, Deep CNN was one of the state-of-the-art and the "hottest" method in the community. So Iinitially thought, why don't I try to adapt it for hand-raising gesture detection. So I forked [Ross Girshick's Faster R-CNN repo](https://github.com/rbgirshick/py-faster-rcnn) and modify it so that it can run on video (initially it was only for image). After spending some time installing Caffe in my Ubuntu laptop, it runs quite well. However, as I can't use the GPU in my laptop (not enough memory), we installed a new computer in the [INSYGHT lab](http://www.insyghtlab.tudelft.nl/) with GeForce GTX 680 GPU. The result can be seen in the following image.
  
### (2) Transfer Learning to train a Pre-existing Deep CNN for self-crafted hand-raising images ([robonet.py](https://github.com/edwardelson/RoboTutor/blob/master/robonet.py))    
The next step is to train Faster R-CNN for a new dataset of handraising gestures that it has not seen before. Faster R-CNN consists of several stages: Region Proposal Network, Convolutional Layers and Classification Network. In this step, I focused more on the Convolutional Layers and Classification Network. I figured out that these layers were actually a result of transfer learning from another existing deep networks. So I went to [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), and browse for some existing pretrained networks. As a preliminary test, I chose the [Flickr model](http://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html) as there's an existing tutorial from Caffe on how to dissect and retrain it.  
  
Another problem is that there's no existing dataset on hand-raising gesture images. So I decided to try crafting our own dataset from scratch. I experimented with very small size of dataset (about the size of 20 images). I then train the network for this dataset.
  
The result is of course very fast convergence, with 100% training data accuracy. This is absolutely an overfit, and when tested with a new image it has never seen, it performs almost no better than chance. We are left with the option of enlarging the dataset (which takes a lot of time) or try out another method. 

![Resulting Image demo Faster R-CNN](https://dl.dropboxusercontent.com/s/pyflmxn6fwk7d4a/robonet.jpeg?dl=0)   
  
Left image is a sample dataset, Right image is the python code that I ran.    
  
### (3) Haar Feature Approach for Face + Hand Detection ([haar_handraise_detection.py](https://github.com/edwardelson/RoboTutor/blob/master/haar_handraise_detection.py))
I was inspired by a paper which tries to tackle this problem by detecting face and hand separately. I borrowed this idea and decide to find the Euclidean distance between detected faces and hands to decide whether a hand-raising gesture exists. I experimented with Haar Feature with Skin Detection (preliminary elimination of non-skin objects). The result can be seen below.   
   
![Normal Raw Image](https://dl.dropboxusercontent.com/s/o9vsf88k3ildt3e/raw_norm.jpeg?dl=0)  
  
![Normal with Skin Detected Image](https://dl.dropboxusercontent.com/s/mqrlkfmh23mc9ac/mask_norm.jpeg?dl=0)  
    
![Hand-raising Gesture Detected in Raw Image](https://dl.dropboxusercontent.com/s/0qx0um6bow9dsnj/raw.jpeg?dl=0)  
  
![Hand-raising Gesture Detected in Skin Detected Image](https://dl.dropboxusercontent.com/s/gul0rm636znasky/mask.jpeg?dl=0)     
    
A similar result can also be seen in the following [video](https://www.youtube.com/watch?v=Lps8rkGjzvY&feature=youtu.be) that I've made.   
  
I wanted to use the same idea, but now using Faster R-CNN to train and detect face and hand, hence combining step 1-3. However, I did not have enough time as my Student Exchange Period is ending soon. So I stopped at this stage and proceed to implement the Java wrapper for the code. This is to allow the Nao robot to use this module. The current result is far from perfect (especially in noisy environment), but it can roughly point out the location of hand raising gestures in a classroom setting. I believe that the usage of CNN will yield a better result (more accurate detection of hand and face).   

### (4) Implementing Java Wrapper for Python code to fit into the Nao system (to be updated)    
This work is under progress.  
     
>[Back to List of Projects](https://edwardelson.github.io)  

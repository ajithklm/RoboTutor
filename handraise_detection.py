#author Edward Elson, May 3rd 2016
#references:
#http://nmarkou.blogspot.nl/2012/02/haar-xml-file.html for hand cascade .xml
#opencv for face cascade .xml
#https://github.com/seereality/opencvDemos/blob/master/skinDetect.py for skin detection
# http://www.icmc.usp.br/pessoas/moacir/papers/NazarePonti_CIARP2013.pdf for overall method inspiration (no skin though)

#1. perform skin detection to image captured from video
#2. detect face and hand on skinned image with .xml files
#3. find euclidean distance between hand and face, must be less than 2*width and 2*height of face
#4. if fulfilled, draw another bounding box

# Required moduls
import cv2
import numpy
import math

# Constants for finding range of skin color in YCrCb
min_YCrCb = numpy.array([0,133,77],numpy.uint8)
max_YCrCb = numpy.array([255,173,127],numpy.uint8)

# Create a window to display the camera feed
cv2.namedWindow('Camera Output')

# Get pointer to video frames from primary device
videoFrame = cv2.VideoCapture(0)

# Process the video frames
keyPressed = -1 # -1 indicates no key pressed

#detect hand
hand_cascade = cv2.CascadeClassifier("/home/edwardelson/Downloads/Hand.Cascade.1.xml")
# taken from http://nmarkou.blogspot.nl/2012/02/haar-xml-file.html
#hand_cascade = cv2.CascadeClassifier("home/edwardelson/Downloads/GstHanddetect-master/src/xml/palm.xml")
#hand_cascade = cv2.CascadeClassifier("/home/edwardelson/Downloads/Opencv-master/haarcascade/fist.xml")

#detect face
face_cascade = cv2.CascadeClassifier("/home/edwardelson/anaconda3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")

while(keyPressed < 0): # any key pressed has a value >= 0

    # Grab video frame, decode it and return next video frame
    readSucsess, sourceImage = videoFrame.read()

    sourceImage = cv2.flip(sourceImage,1)
    #sourceImage = cv2.imread("/home/edwardelson/Downloads/classroom.jpg")

    # Convert image to YCrCb
    imageYCrCb = cv2.cvtColor(sourceImage,cv2.COLOR_BGR2YCR_CB)
    # Find region with skin tone in YCrCb image
    skinRegion = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
    # Convert image to only with skin
    masked_img = cv2.bitwise_and(sourceImage, sourceImage, mask = skinRegion)

    #detect faces and draw bounding box
    face_gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(face_gray, 1.3, 5)
    for (x_f,y_f,w_f,h_f) in faces:
        cv2.rectangle(masked_img,(x_f,y_f),(x_f+w_f,y_f+h_f),(0,255,0),2)
        cv2.rectangle(sourceImage,(x_f,y_f),(x_f+w_f,y_f+h_f),(0,255,0),2)

    #detect hands and draw bounding box
    hands = hand_cascade.detectMultiScale(masked_img, 1.3, 5)
    for (x_h,y_h,w_h,h_h) in hands:
        cv2.rectangle(masked_img,(x_h,y_h),(x_h+w_h,y_h+h_h),(0,0,255),2)
        cv2.rectangle(sourceImage,(x_h,y_h),(x_h+w_h,y_h+h_h),(0,0,255),2)

    #detect hand-raising
    #euclidean distance between center of face and center of hand must be less than 2 times the width of face
    #if fulfilled draw another bounding box    
    for (x_f,y_f,w_f,h_f) in faces:
        for (x_h,y_h,w_h,h_h) in hands:
            dx = math.fabs((x_f+0.5*w_f)-(x_h+0.5*w_h))
            dy = math.fabs((y_f+0.5*h_f)-(y_h+0.5*h_h))

            if (dx <= 2*w_f):
                if dy <= 2*h_f:
                    cv2.rectangle(masked_img,(x_f-w_f,y_f-w_f),(x_f+2*w_f,y_f+2*h_f),(255,0,0,),2)
                    cv2.rectangle(sourceImage,(x_f-w_f,y_f-w_f),(x_f+2*w_f,y_f+2*h_f),(255,0,0,),2)

    # Display the source image
    cv2.imshow('Masked Image',masked_img)
    cv2.imshow('Normal Image',sourceImage)

    # Check for user input to close program
    keyPressed = cv2.waitKey(2) # wait 2 millisecond in each iteration of while loop

# Close window and camera after exiting the while loop
cv2.destroyWindow('Camera Output')
videoFrame.release()
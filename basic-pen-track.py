## Code Inspired from 
## https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/

# import the necessary packages
from collections import deque # a super fast type data str
## more info https://docs.python.org/2/library/collections.html#collections.deque
from imutils.video import VideoStream
# used to capture videostream from your camera
import numpy as np
import argparse
import cv2
import imutils
import time
import matplotlib.pyplot as plt
## ending the importing units ##


ap = argparse.ArgumentParser()
'''ap.add_argument("-v", "--video",
	help="path to the (optional) video file")'''
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

## Add color of object and length of it ##
#greenLower = (29, 86, 6)
#greenUpper = (64, 255, 255)
blue_lower = (138, 150,20)
blue_higher = (150,250,255)
pts = deque(maxlen=args["buffer"])

## allow it to warm up/sleep
# allow the camera or video file to warm up
time.sleep(2.0)

##Start the video capture 
#cv2.namedWindow("preview")
cap = cv2.VideoCapture(0)

while True:
    # read the frame
    ret, frame = cap.read()
    # read the frame, resize and convert to hsv
    frame = imutils.resize(frame, width=700)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # make the mask of the color, and apply smoothing
    mask = cv2.inRange(hsv, blue_lower, blue_higher)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=4)

    # find contours in the mask and initialize the current
	# (x, y) center of the ball
    
    # Used to segment white from black
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only detect contours if they are present 
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # # it to compute the minimum enclosing circle and
        # # centroid

        # find the x,y of the biggest contour
        c = max(cnts, key=cv2.contourArea)
        # find the minimum enclosed circle 
        # for the contour
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        ## this section is similar to center of mass
        # this is used to find the image moment 
        M = cv2.moments(c)
        # which is used to calculate the image centroid
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only go ahead if the radius is greater than n
        if radius >12:
            #draw the circle on top of it 
            cv2.circle(frame,(int(x),int(y)),int(radius),
                (0,255,255),2)
            #draw the centroid
            cv2.circle(frame,center,5,(191, 70, 232),-1)
        #use the last center and apend in our deque
            pts.appendleft(center)

    #loop over the dequeue
    for i in range(1,len(pts)):
        # check if line is there or not 
        if pts[i-1] is None or pts[i] is None:
            continue
        #otherwise calculate thickness and draw line
        thickness = int(np.sqrt(args['buffer']/float(i+1))*2.5)
        cv2.line(frame, pts[i - 1], pts[i], (232, 131, 252), thickness)

    #cv2.imshow('frame',mask)
    cv2.imshow('origin',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

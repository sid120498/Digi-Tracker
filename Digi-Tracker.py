# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 10:26:58 2018

@author: Siddharth
"""
#imports
import cv2
import numpy as np
from collections import deque
import time
from math import ceil
from keras.models import model_from_json

#loading model
json_file = open('models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("models/model.h5")
print("Loaded model from disk")


#GLOBAL VARIABLES
#create video capture object
cap = cv2.VideoCapture(0)
f_queue = deque(maxlen= 300)
const = 0
counter = 0
start_time = time.time()
prev_digit = -1


#function to return upper and lower hsv value for a RGB valye
def colorPicker(color):
    color = np.uint8([[color]])
    hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
    lower = np.array([hsv[0][0][0] - 10, 50, 50])
    upper = np.array([hsv[0][0][0] + 10, 255, 255])
    return upper, lower
#print(lower_col, upper_col)


#to create Empty(black) image 
def createEmFrame(shape):
    blank_image = np.zeros((shape[0],shape[1],3), np.uint8)
    return blank_image


def newDigFrame(sleep, counter, img):
    global start_time, f_queue
    if(int((time.time() - start_time))%sleep == 0 and (time.time()-start_time)>1):
        start_time = time.time()
        f_queue.clear()
        if(not isinstance(img, int)):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find contours in the image
            _, cntr, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #return if no contour is found
            if(not len(cntr)>0):
                return 0
            max_cnt = max(cntr, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(max_cnt)
            roi = img[ceil(y):ceil(y+h), ceil(x):ceil(x+w)]
            hor_padding = int(w/14)
            ver_padding = int(h/14)
            padded_roi = cv2.copyMakeBorder(roi, hor_padding, hor_padding,
                                      ver_padding, ver_padding, 
                                      cv2.BORDER_CONSTANT,value = [0, 0 ,0])
            resized_roi = cv2.resize(padded_roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = resized_roi.reshape(1,28,28, 1)
            #converting to model compatible form
            roi = roi.astype("float32")/255
            digit_prob = model.predict(roi)
            digit = np.argmax(digit_prob, axis=1)
            return 0, digit
            
        return 0, -1
    return counter, 10


upper_col, lower_col = colorPicker([0, 0, 255])


while True:
    counter, digit = newDigFrame(10, counter, const)
    
    counter = counter + 1
    #frame read bool and frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if(counter == 1):
        const = createEmFrame(frame.shape)
           
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    kernel = np.ones((5,5), np.uint8)
    mask=cv2.inRange(hsv,lower_col,upper_col)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)   
    mask = cv2.dilate(mask, kernel, iterations=1)
    res=cv2.bitwise_and(frame, frame,mask=mask)
    
    _, cnts, heir=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    center = None
    if len(cnts)>0:
        #selecting contour of maximum area
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        if(radius>5):
            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            #cv2.circle(const, center,15,(255, 255,255), -1)
            
    f_queue.appendleft(center)
    for i in range (1,len(f_queue)):
        if f_queue[i-1] is None or f_queue[i] is None:
            continue
        cv2.line(const, f_queue[i-1], f_queue[i], (255,255,225), 5)
        cv2.line(frame, f_queue[i-1], f_queue[i], (0, 0 ,255), 5)
    
    if(digit>=0):
        if(digit<10):
            prev_digit = digit
        cv2.putText(frame, str(prev_digit), (50, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
    cv2.imshow("Camera", frame)
    
    k=cv2.waitKey(30) & 0xFF
    
    if k==32:
        break

# When everything done, release the capture
cap.release()
#cv2.imshow("Test", const)
cv2.destroyAllWindows()
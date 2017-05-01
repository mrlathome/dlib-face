#!/usr/bin/python

from scipy.spatial import distance as dist
import cv2, os
import sys
import numpy as np
import dlib
import time
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")

alpha = 0.35

def draw_face(image,dlib_rect):
    scale_x = (d.right() - d.left())/4
    scale_y = (d.bottom() - d.top())/4
    cv2.line(blank_image,(d.left(),d.top()),(d.left()+scale_x,d.top()),(100,200,100))
    cv2.line(blank_image,(d.left(),d.top()),(d.left(),d.top()+scale_y),(100,200,100))
    cv2.line(blank_image,(d.right(),d.top()),(d.right()-scale_x,d.top()),(100,200,100))
    cv2.line(blank_image,(d.right(),d.top()),(d.right(),d.top()+scale_y),(100,200,100))
    cv2.line(blank_image,(d.left(),d.bottom()),(d.left()+scale_x,d.bottom()),(100,200,100))
    cv2.line(blank_image,(d.left(),d.bottom()),(d.left(),d.bottom()-scale_y),(100,200,100))
    cv2.line(blank_image,(d.right(),d.bottom()),(d.right()-scale_x,d.bottom()),(100,200,100))
    cv2.line(blank_image,(d.right(),d.bottom()),(d.right(),d.bottom()-scale_y),(100,200,100))

def draw_landmark(shape,name,output):    
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS[name]   
    cv2.drawContours(output, [cv2.convexHull(shape[rStart:rEnd])], -1, (100, 100, 100), 1)

try:
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('webcam',cv2.WINDOW_NORMAL)
    while True:
        ret_val, image = cam.read()        
        image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
        blank_image = image.copy()                    
        dets = detector(image, 1)
        for i, d in enumerate(dets):
            shape = predictor(image, d)
            shape = face_utils.shape_to_np(shape)          
            
            draw_landmark(shape,"mouth",blank_image)
            draw_landmark(shape,"nose",blank_image)
            draw_landmark(shape,"right_eye",blank_image)
            draw_landmark(shape,"left_eye",blank_image)

            draw_face(blank_image,d)
            
            for point in shape[48:60]:
                cv2.circle(blank_image,(point[0],point[1]),1,(200,0,0))            
            
        cv2.addWeighted(blank_image, alpha, image, 1 - alpha, 0, image)

        cv2.imshow('webcam', image)
        key = cv2.waitKey(1) & 0XFF                
        if key == 27: #esc
            break
        
    cam.release()
    cv2.destroyAllWindows()    

except KeyboardInterrupt:
    cv2.destroyAllWindows()
    sys.exit(-1)

except Exception:
    print "Error Occurred"
    raise
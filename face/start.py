#!/usr/bin/python

import cv2, os
import sys
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")

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
            landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
            for idx, point in enumerate(landmarks):
                pos = (point[0, 0], point[0, 1])                
                cv2.circle(blank_image, pos, 1, color=(0, 255, 255))  

            cv2.rectangle(blank_image,(d.left(),d.top()),(d.right(),d.bottom()),(100,200,100),1)
            
        cv2.addWeighted(blank_image, 0.5, image, 1 - 0.5, 0, image)

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
    print "error occurred"
    sys.exit(-2)
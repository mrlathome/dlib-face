#!/usr/bin/python

import math
import cv2, os
import sys
import numpy as np
import dlib
import time
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")

alpha = 0.67

def draw_face(d,output,color):
    scale_x = (d.right() - d.left())/4
    scale_y = (d.bottom() - d.top())/4
    cv2.line(output,(d.left(),d.top()),(d.left()+scale_x,d.top()),color)
    cv2.line(output,(d.left(),d.top()),(d.left(),d.top()+scale_y),color)
    cv2.line(output,(d.right(),d.top()),(d.right()-scale_x,d.top()),color)
    cv2.line(output,(d.right(),d.top()),(d.right(),d.top()+scale_y),color)
    cv2.line(output,(d.left(),d.bottom()),(d.left()+scale_x,d.bottom()),color)
    cv2.line(output,(d.left(),d.bottom()),(d.left(),d.bottom()-scale_y),color)
    cv2.line(output,(d.right(),d.bottom()),(d.right()-scale_x,d.bottom()),color)
    cv2.line(output,(d.right(),d.bottom()),(d.right(),d.bottom()-scale_y),color)

def draw_landmark(shape,name,output):        
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS[name]    
    cv2.drawContours(output, [cv2.convexHull(shape[rStart:rEnd])], -1, (100, 100, 100), 1)

def get_angle(pointa,pointb):
    return math.atan2(pointa[1] - pointb[1],pointa[0] - pointb[0])

def radian_to_degree(a):
    return (a * 180) / math.pi

def get_distance(pointa,pointb):
    return math.sqrt(math.pow(pointa[0] - pointb[0],2) + math.pow(pointa[1] - pointb[1],2))

def process():
    try:
        cam = cv2.VideoCapture(0)
        cv2.namedWindow('webcam',cv2.WINDOW_NORMAL)
        while True:
            ret_val, image = cam.read() 
            image = cv2.flip(image,1)
            original_image = image.copy()       
            image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
            width,height,channel = image.shape
            blank_image = image.copy()                    
            dets = detector(image, 1)
            for i, d in enumerate(dets):
                shape = predictor(image, d)
                shape = face_utils.shape_to_np(shape)
                
                face_scale_x = float(d.right() - d.left())/width                
                
                draw_landmark(shape,"mouth",blank_image)
                draw_landmark(shape,"nose",blank_image)
                draw_landmark(shape,"right_eye",blank_image)
                draw_landmark(shape,"left_eye",blank_image)

                face_angle = (int(radian_to_degree(get_angle(shape[45],shape[36]))))

                right_distance = int(get_distance(shape[33],shape[36]))
                left_distance = int(get_distance(shape[33],shape[45]))
                face_align = right_distance - left_distance                

                face_align_text = "right"
                if face_align > -5 and face_align < 5:
                    face_align_text = "center"
                elif face_align < -5:
                    face_align_text = "left"                

                cv2.putText(blank_image,face_align_text,(d.left(),d.bottom() + 15),cv2.FONT_HERSHEY_DUPLEX,face_scale_x,(100,100,200))

                if (face_angle > -10 and face_angle < 10):
                    draw_face(d,blank_image,(100,255,100))
                else:
                    draw_face(d,blank_image,(100,100,255))

                cv2.putText(blank_image,"tetha {}".format(face_angle),(d.left(),d.top() - 5),cv2.FONT_HERSHEY_DUPLEX,face_scale_x,(100,100,200))     
                
                
            blank_image = cv2.resize(blank_image,(0,0),fx=2,fy=2)
            cv2.addWeighted(blank_image, alpha, original_image, 1 - alpha, 0, original_image)

            cv2.imshow('webcam', original_image)
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

if __name__ == '__main__':
    process()
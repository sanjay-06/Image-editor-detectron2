import cv2
import numpy as np

class Feature:

    def __init__(self):
        self.img = None
    
    def harris_corner(self, img, gray):
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,blockSize=2, ksize=3, k=0.04)
        dst = cv2.dilate(dst,None)
        img[dst>0.01*dst.max()]=[0,0,255]
        return img

    def sift(self,img, gray):
        sift = cv2.SIFT_create()
        kp = sift.detect(gray,None)
        img=cv2.drawKeypoints(gray,kp,img)
        return img
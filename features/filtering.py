import numpy as np
import cv2
from numpy import asarray

class Filter:
    def __init__(self, sepia_kernel = [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]]) -> None:
        self.img = None
        self.sepia_kernel = np.array(sepia_kernel)
    
    def perform_serpia_filter(self, img):
        sepia_kernel = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
        sepia = cv2.transform(img, sepia_kernel)
        return sepia
    
    def perform_vintage(self, img):
        vintage_kernel = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])
        vintage = cv2.transform(img, vintage_kernel)
        return vintage
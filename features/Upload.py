import cv2
from features.object_detection import ObjectDetection
class UploadObj:
    def __init__(self):
        self.img = None
    
    def set_image(self, img):
        self.img = img

    def set_object(self):
        self.obj = ObjectDetection(cv2.imread('html/'+self.img))
        return self.obj
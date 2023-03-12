# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
import cv2
import matplotlib.pyplot as plt
from fastapi.responses import HTMLResponse

object_detect=APIRouter()
templates=Jinja2Templates(directory="html")

class ObjectDetection:
    def __init__(self) -> None:
        # Inference with a panoptic segmentation model
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        self.predictor = None
    
    def predict(self, image):
        self.predictor = DefaultPredictor(self.cfg)
        panoptic_seg, segments_info = self.predictor(image)["panoptic_seg"]
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
        visual = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
        return visual
# cv2_imshow(out.get_image()[:, :, ::-1])

obj = ObjectDetection()

@object_detect.get('/', response_class=HTMLResponse)
def load(request: Request):
    return templates.TemplateResponse("index.html",{"request":request})

@object_detect.post('/{string}')
def detect(request : Request, string):
    print(string)
    im = cv2.imread(string)
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 
    visual = obj.predict(img)
    plt.imshow(visual)
    plt.show()
    return templates.TemplateResponse("index.html",{"request":request})


# import some common libraries
import os
import cv2
import json
import random
import numpy as np

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
import matplotlib.pyplot as plt
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from fastapi import File, UploadFile
import shutil
from PIL import Image

object_detect=APIRouter()
templates=Jinja2Templates(directory="html")

class ObjectDetection:
    def __init__(self) -> None:
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        self.predictor = None
        self.image_name = None
    
    def predict(self, image):
        self.predictor = DefaultPredictor(self.cfg)
        panoptic_seg, segments_info = self.predictor(image)["panoptic_seg"]
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
        visual = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
        return visual

obj = ObjectDetection()

@object_detect.get('/')
def detect(request : Request):
    return templates.TemplateResponse("index.html",{"request":request})

@object_detect.get('/detect', response_class=HTMLResponse)
def load(request: Request):
    im = cv2.imread('html/'+obj.image_name)
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 
    visual = obj.predict(img)
    file_static_location = f"static/detect.jpg"
    file_location = f"html/static/detect.jpg"
    cv2.imwrite(file_location, visual)
    return templates.TemplateResponse("show.html",{"request":request, "image": obj.image_name, "detect": file_static_location})

@object_detect.post('/upload_file')
async def handle_form(upload_file:UploadFile = File(...)):
    filename=upload_file.filename
    file_static_location = f"static/{filename}"
    file_location = f"html/{file_static_location}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    obj.image_name = file_static_location

    return {"message":"success","statuscode":200}

@object_detect.get('/show')
async def show_image(request: Request):
    print(obj.image_name)
    return templates.TemplateResponse("show.html",{"request":request, "image": obj.image_name})


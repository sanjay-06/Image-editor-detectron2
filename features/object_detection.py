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
from PIL import Image

class ObjectDetection:
    def __init__(self) -> None:
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        self.predictor = None
        self.image_name = None
        self.pred = None
    
    def predict_image(self, image):
        self.predictor = DefaultPredictor(self.cfg)
        self.pred = self.predictor(image)
    
    def predict(self, image):
        self.predict_image(image)
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
        visual = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
        return visual

    def get_masks(self, item_mask_index, pred):
        masks = np.asarray(pred['instances'].pred_masks.to("cpu"))
        item_mask = masks[item_mask_index]
        segmentation = np.where(item_mask == True)
        x_min = int(np.min(segmentation[1]))
        x_max = int(np.max(segmentation[1]))
        y_min = int(np.min(segmentation[0]))
        y_max = int(np.max(segmentation[0]))
        return x_min, x_max, y_min, y_max, item_mask

    def crop(self, im, x_min, x_max, y_min, y_max):
        cropped = Image.fromarray(im[y_min:y_max, x_min:x_max, :], mode='RGB')
        return cropped

    def alpha_mask(self, cropped, background, cropped_mask, paste_position = (0, 300)):
        new_fg_image = Image.new('RGB', background.size)
        new_fg_image.paste(cropped, paste_position)
        new_alpha_mask = Image.new('L', background.size, color=0)
        new_alpha_mask.paste(cropped_mask, paste_position)
        return new_fg_image, new_alpha_mask

    def clone(self, im, item_mask_index=6, paste_position = (0, 300)):
        self.predict_image(im)
        x_min, x_max, y_min, y_max, item_mask = self.get_masks(item_mask_index, self.pred)
        cropped = self.crop(im, x_min, x_max, y_min, y_max)
        mask = Image.fromarray((item_mask * 255).astype('uint8'))

        cropped_mask = mask.crop((x_min, y_min, x_max, y_max))
        background = Image.fromarray(im, mode='RGB')
        new_fg_image, new_alpha_mask = self.alpha_mask(cropped, background, cropped_mask)

        composite = Image.composite(new_fg_image, background, new_alpha_mask)
        return np.array(composite)
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
        panoptic_seg, segments_info = self.pred["panoptic_seg"]
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
    
    def blur_box(self, im):
        self.predict_image(im)
        x_min, x_max, y_min, y_max, item_mask = self.get_masks(6, self.pred)      
        roi = im[y_min:y_max, x_min:x_max]
        mask = np.zeros(im.shape[:2], dtype=np.uint8)
        mask[y_min:y_max, x_min:x_max] = 255
        mask_inv = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(im, im, mask=mask_inv)
        blurred_bg = cv2.GaussianBlur(bg, (51,51), 0)
        blurred_bg[y_min:y_max, x_min:x_max] = roi
        return blurred_bg
    
    def get_idx(self, im, mask_idx=0):
        self.predict_image(im)
        x_min, x_max, y_min, y_max, item_mask = self.get_masks(int(mask_idx), self.pred)
        return im[y_min:y_max, x_min:x_max]
    
    def blur_bg(self, im, mask_idx=0):
        self.predict_image(im)
        x_min, x_max, y_min, y_max, item_mask = self.get_masks(int(mask_idx), self.pred)
        mask = Image.fromarray((item_mask * 255).astype('uint8'))
        cropped_mask = mask.crop((x_min, y_min, x_max, y_max))
        blurred_image = cv2.GaussianBlur(im, (21,21), 0)
        _, binary_mask = cv2.threshold(np.array(mask), 128, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(binary_mask)
        masked_image = cv2.bitwise_and(blurred_image, blurred_image, mask=mask_inv)
        result = cv2.bitwise_or(im, im, mask=binary_mask)
        blended_image = cv2.bitwise_or(np.array(result), masked_image)
        return blended_image
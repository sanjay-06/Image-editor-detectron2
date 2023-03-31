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
    def __init__(self, im) -> None:
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        self.predictor = None
        self.image_name = im
        self.pred = None
        self.predict_image()
    
    def predict_image(self):
        self.predictor = DefaultPredictor(self.cfg)
        self.pred = self.predictor(self.image_name)
    
    def predict(self):
        panoptic_seg, segments_info = self.pred["panoptic_seg"]
        v = Visualizer(self.image_name[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
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

    def crop(self, img, x_min, x_max, y_min, y_max):
        cropped = Image.fromarray(img[y_min:y_max, x_min:x_max, :], mode='RGB')
        return cropped

    def alpha_mask(self, cropped, background, cropped_mask, paste_position = (0, 300)):
        new_fg_image = Image.new('RGB', background.size)
        new_fg_image.paste(cropped, paste_position)
        new_alpha_mask = Image.new('L', background.size, color=0)
        new_alpha_mask.paste(cropped_mask, paste_position)
        return new_fg_image, new_alpha_mask

    def clone(self, item_mask_index=6, paste_position = (0, 300)):
        x_min, x_max, y_min, y_max, item_mask = self.get_masks(int(item_mask_index), self.pred)
        cropped = self.crop(self.image_name, x_min, x_max, y_min, y_max)
        mask = Image.fromarray((item_mask * 255).astype('uint8'))

        cropped_mask = mask.crop((x_min, y_min, x_max, y_max))
        background = Image.fromarray(self.image_name, mode='RGB')
        new_fg_image, new_alpha_mask = self.alpha_mask(cropped, background, cropped_mask)

        composite = Image.composite(new_fg_image, background, new_alpha_mask)
        return np.array(composite)
    
    def blur_box(self, mask_idx=0):
        x_min, x_max, y_min, y_max, item_mask = self.get_masks(int(mask_idx), self.pred)      
        roi = self.image_name[y_min:y_max, x_min:x_max]
        mask = np.zeros(self.image_name.shape[:2], dtype=np.uint8)
        mask[y_min:y_max, x_min:x_max] = 255
        mask_inv = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(self.image_name, self.image_name, mask=mask_inv)
        blurred_bg = cv2.GaussianBlur(bg, (51,51), 0)
        blurred_bg[y_min:y_max, x_min:x_max] = roi
        return blurred_bg
    
    def change_bg_image(self, image_file, item_mask_index=0):
        x_min, x_max, y_min, y_max, item_mask = self.get_masks(int(item_mask_index), self.pred)
        mask = Image.fromarray((item_mask * 255).astype('uint8'))
        cropped_mask = mask.crop((x_min, y_min, x_max, y_max))
        _, binary_mask = cv2.threshold(np.array(mask), 128, 255, cv2.THRESH_BINARY)
        inverted_mask = binary_mask
        mask_inv = cv2.bitwise_not(binary_mask)
        image_file = cv2.resize(image_file, (mask_inv.shape[1], mask_inv.shape[0]))
        masked_image = cv2.bitwise_and(image_file, image_file, mask=mask_inv)
        result = cv2.bitwise_or(self.image_name, self.image_name, mask=inverted_mask)
        image_file = cv2.resize(image_file, (result.shape[1], result.shape[0]))
        blended_image = cv2.bitwise_or(result, masked_image)
        return blended_image
    
    def get_idx(self, mask_idx=0):
        x_min, x_max, y_min, y_max, item_mask = self.get_masks(int(mask_idx), self.pred)
        return self.image_name[y_min:y_max, x_min:x_max]
    
    def blur_bg(self, mask_idx=0):
        x_min, x_max, y_min, y_max, item_mask = self.get_masks(int(mask_idx), self.pred)
        mask = Image.fromarray((item_mask * 255).astype('uint8'))
        cropped_mask = mask.crop((x_min, y_min, x_max, y_max))
        blurred_image = cv2.GaussianBlur(self.image_name, (21,21), 0)
        _, binary_mask = cv2.threshold(np.array(mask), 128, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(binary_mask)
        masked_image = cv2.bitwise_and(blurred_image, blurred_image, mask=mask_inv)
        result = cv2.bitwise_or(self.image_name, self.image_name, mask=binary_mask)
        blended_image = cv2.bitwise_or(np.array(result), masked_image)
        return blended_image
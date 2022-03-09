import torch
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
setup_logger()
import numpy as np
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import time


configPath = model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml")
weights = "./fs/model_gun9b.pth"
confidence = 0.5
nms = 0.3
logname = "./fs/logs.txt"


cfg = get_cfg()
cfg.merge_from_file(configPath)

cfg.OUTPUT_DIR = ''
cfg.MODEL.RETINANET.NUM_CLASSES = 80
cfg.MODEL.RETINANET.NMS_THRESH_TEST = nms
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128, 256]]

cfg.INPUT.MIN_SIZE_TEST = 760
cfg.INPUT.MAX_SIZE_TEST = 1296

cfg.MODEL.WEIGHTS = weights
predictor = DefaultPredictor(cfg)
print('detector init !')

fr = 0
inputvid = "./fs/test_1080_30_3.mp4"
vid = cv2.VideoCapture(inputvid)

t1 = time.time()
while True:

  grabbed, frame = vid.read()
  if not grabbed: break

  fr+=1
  if fr < 600:
    continue
  framTxt = str(fr) + ', model9b, 8x' 
  cv2.putText(frame, framTxt, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,250,255), 2) 

  outputs = predictor(frame)

  t2 = time.time()
  ttext = 'Frame - ' + str(fr) + ', time = ' + str(t2-t1) + "\n"
  if fr%10 == 0:
    print(ttext)
  with open(logname, "a") as logtxt:
    logtxt.write(ttext)
  
  t1 = time.time()
  
  




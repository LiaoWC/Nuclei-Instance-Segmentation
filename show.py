# This script is to plot the prediction on the 6 test images.
from detectron2.data.datasets import register_coco_instances
import os
import json
import cv2
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

#############################
#         Setting           #
#############################
# Must set the same as what you set for training
DEFAULT_CONFIG = "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
# Model weights path
WEIGHTS = 'pretrained.pth'

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(DEFAULT_CONFIG))
cfg.DATASETS.TRAIN = ("nuclei_train",)
cfg.DATASETS.TEST = ("nuclei_test")
# cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(DEFAULT_CONFIG)
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 3000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.MODEL.WEIGHTS = WEIGHTS
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001  # You can modify this
cfg.TEST.DETECTIONS_PER_IMAGE = 3000  # You can modify this
cfg.DATASETS.TEST = ("nuclei_test",)
predictor = DefaultPredictor(cfg)

# dataset_dicts = get_microcontroller_dicts(
# 'Microcontroller Segmentation/test')
dd = 'test'
with open(f'dataset/nuclei_{dd}_dataset.json', 'r') as f:
    ddd = json.load(f)['images']

for d in ddd:
    print(os.path.join(f'dataset/{dd}_images', d["file_name"]))
    im = cv2.imread(os.path.join(f'dataset/{dd}_images', d["file_name"]))
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata={},
                   scale=0.8,
                   # remove the colors of unsegmented pixels
                   instance_mode=ColorMode.IMAGE_BW
                   )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize=(14, 10))
    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    plt.show()

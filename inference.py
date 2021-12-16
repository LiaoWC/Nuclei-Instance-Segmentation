# Make prediction and output as a file.
# Prediction file will be saved in "pred/"
import os
import json
import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from pycocotools import mask
from datetime import datetime

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
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(DEFAULT_CONFIG)
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 3000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

cfg.MODEL.WEIGHTS = WEIGHTS
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001  # You can modify this
cfg.DATASETS.TEST = ("nuclei_test",)
cfg.TEST.DETECTIONS_PER_IMAGE = 3000  # You can modify this
predictor = DefaultPredictor(cfg)

with open("dataset/test_img_ids.json", "r") as f:
    test_img_ids = json.load(f)

results = []
for test_img_dict in test_img_ids:
    print(os.path.join(f'dataset/test_images', test_img_dict["file_name"]))
    im = cv2.imread(os.path.join(f'dataset/test_images',
                                 test_img_dict["file_name"]))
    outputs = predictor(im)
    instances = outputs['instances']

    pred_boxes = instances.pred_boxes
    scores = instances.scores
    pred_masks = instances.pred_masks
    for i in range(len(instances)):
        result = {
            'image_id': test_img_dict['id'],
            'bbox': pred_boxes[i].tensor.cpu().tolist()[0],
            'score': scores[i].cpu().item(),
            'category_id': 1,
            'segmentation': {
                'size': [1000, 1000],
                'counts': mask.encode(
                    np.asfortranarray(
                        pred_masks[i].cpu()))['counts'].decode('utf-8')
            }
        }
        results.append(result)

os.makedirs("./pred", exist_ok=True)
filename = f'./pred/pred_{datetime.now().strftime("%Y-%m%d-%H%M%S")}.json'
with open(filename, 'w') as f:
    json.dump(results, f)

print("------------- Done ---------------")
print("Prediction file is saved:", filename)

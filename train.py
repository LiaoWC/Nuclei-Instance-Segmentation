from detectron2.data.datasets import register_coco_instances
import os
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

#############################
# Register datasets
#############################
# Augmented train dataset
register_coco_instances(f'nuclei_aug',
                        json_file=os.path.join('aug', f'annotation.json'),
                        image_root='aug', metadata={})
# Non-augmented train & test dataset (the default number of valid img is 0 so not building a valid dataset.)
for phase in ["train", "test"]:
    register_coco_instances(f'nuclei_{phase}',
                            json_file=os.path.join('dataset', f'nuclei_{phase}_dataset.json'),
                            image_root=os.path.join('dataset', f'{phase}_images'), metadata={})

#############################
#         Setting           #
#############################
# Use the default configuration. You can modify it by yourself.
cfg = get_cfg()

# Modify this to use other model architecture
# Refer to this: https://github.com/facebookresearch/detectron2/tree/main/configs
cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"))

# Set datasets
cfg.DATASETS.TRAIN = ("nuclei_aug",)
cfg.DATASETS.TEST = "nuclei_test"

# Train from a detectron2 pretrained model weights
# Uncomment the following line to use the detectron2 pretrained model (based on your architecture configurations)
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml")

# Train from your pretrained model weights
# Uncomment the following line to use your own pretrained model by assigning a path
# cfg.MODEL.WEIGHTS = 'pretrained.pth'

# Other settings (You can refer to detectron2's configuration files. If not set, default config will be used.)
# cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.01
# cfg.SOLVER.MAX_ITER = 60000
cfg.SOLVER.CHECKPOINT_PERIOD = 500  # The period of storing a model
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Number of classes
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)  # Default output directory: output/

#############################
#          Train            #
#############################
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)  # Important! Set True if you want resume your training.
trainer.train()

import torch, detectron2
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from data_function2 import *
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultPredictor

for d in ["train","val"]:
    DatasetCatalog.register(d, lambda d=d: get_sports_dicts("/data/soyeonhong/test_dataset/"+d+"_json") )
    MetadataCatalog.get(d).set(thing_classes=["Stadium","Three_Point_Line","Paint_zone","Player","Goal_post","Game_tool"])
basketball_metadata = MetadataCatalog.get("train")

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg=get_cfg()
cfg.merge_from_file("/data/soyeon/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
# cfg.DATASETS.TRAIN = ('train',)
# cfg.DATASETS.TEST = ()
# cfg.DATALOADER.NUM_WORKERS = 2
# cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
# cfg.SOLVER.BASE_LR = 0.02  # pick a good LR
# cfg.SOLVER.MAX_ITER = 100    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
# cfg.SOLVER.STEPS = []        # do not decay learning rate
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.WEIGHTS = os.path.join("/data/soyeon/detectron2/custom_datasets/output/", "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
predictor = DefaultPredictor(cfg)



dataset_dicts = get_sports_dicts("/data/soyeon/test_dataset/val_json")
i=1
for d in random.sample(dataset_dicts, 12):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    print(outputs)
    v = Visualizer(im[:, :, ::-1],
                   metadata=basketball_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(os.path.join("/data/soyeonhong/",str(i)+"sample.jpg"),(out.get_image()[:, :, ::-1]))
    i=i+1

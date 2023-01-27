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
import os, json, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from data_function import *
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultPredictor

import gc
gc.collect()
torch.cuda.empty_cache()

for d in ["train","val"]:
    DatasetCatalog.register(d, lambda d=d: get_sports_dicts("/local_datasets/detectron2/basketball/annotations/"+d+"_json") )
    MetadataCatalog.get(d).set(thing_classes=["Stadium","Three_Point_Line","Paint_zone","Player","Goal_post","Game_tool"])
basketball_metadata = MetadataCatalog.get("train")

os.makedirs("./output", exist_ok=True)
def main():
    cfg=get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
    cfg.DATASETS.TRAIN = ('train',)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.SOLVER.IMS_PER_BATCH = 64  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.02  # pick a good LR
    cfg.SOLVER.MAX_ITER = 80000
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
    cfg.OUTPUT_DIR="./output"
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("val", output_dir="./output")
    val_loader = build_detection_test_loader(cfg,"val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
    return predictor

# _C.DATALOADER.NUM_WORKERS = 4 -> number of data loading threads
# _C.SOLVER.IMS_PER_BATCH = 16
# _C.SOLVER.BASE_LR = 0.001
# _C.SOLVER.MAX_ITER = 40000
# _C.SOLVER.STEPS = (30000,)
# _C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
# _C.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05






    
if __name__=="__main__":
    args = default_argument_parser().parse_args()
    print("start")
    print(f"Device Count{torch.cuda.device_count()}")
    launch(
        main,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
    )
import sys
import torch, detectron2

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, yaml

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode

DATA_DIR = sys.argv[1]
model_name = sys.argv[2]
config_path = sys.argv[3]
processor = sys.argv[4]

with open(config_path, 'r') as config_yaml:
            model_yaml = yaml.load(config_yaml, Loader=yaml.FullLoader)
            model_config = model_yaml[model_name]
            model_info = model_config['info']
            model_hyperparams = model_config['hyperparameters']

            model_device = model_yaml['device']

PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODEL_TRAIN_DATA_DIR = os.path.join(PROCESSED_DATA_DIR,"train")
MODEL_VAL_DATA_DIR = os.path.join(PROCESSED_DATA_DIR,"val")
MODEL_TEST_DATA_DIR = os.path.join(PROCESSED_DATA_DIR,"test")

def get_csd_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        record["file_name"] = os.path.join(img_dir, v["filename"])
        record["image_id"] = idx

        annos = v["regions"]

        objs = []

        if type(annos) == list: # custom JSONs are in list format need to make them the same
             annos = dict(enumerate(annos))

        for _, anno in annos.items():

            regions = anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]

            # if len(px) <= 10 or len(py) <= 10:
            #         # print("Ignoring polygon from ", v["filename"], "because a polygon was too small for detectron2.")
            #         continue
            
            poly = [(x, y) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            try: 
                category_dict = {"ND": 0, "LD": 1, "CD": 2, "RD": 3, "DD": 4}
                category_id = category_dict[regions["label"]]

            except KeyError:
                # diff model
                category_dict = {0: 0, 5: 1, 10: 2, 3: 3, 8: 4, 13: 5, 6:6, 11:7, 16:8} # converts the unique charge state number to an ID.
                class_dict = {"(0,0)": 0, "(0,1)": 5, "(0,2)": 10, "(1,0)": 3, "(1,1)": 8, "(1,2)": 13, "(2,0)": 6, "(2,1)": 11, "(2,2)": 16}
                try: 
                    category_id = category_dict[regions["label"]]
                except KeyError:
                    category_id = category_dict[class_dict[regions['label']]]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": category_id,
            }

            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

DatasetCatalog.clear()
MetadataCatalog.clear()

for d in ["train", "val"]:
    DatasetCatalog.register("csd_" + d, lambda d=d: get_csd_dicts(os.path.join(PROCESSED_DATA_DIR,d)))
    MetadataCatalog.get("csd_" + d).set(thing_classes=list(model_info['class_dict'].keys()))

csd_metadata = MetadataCatalog.get("csd_train")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("csd_train",)
cfg.DATASETS.TEST = ()

cfg.MODEL.DEVICE = processor 
cfg.DATALOADER.NUM_WORKERS = model_device[processor]['num_workers']

cfg.SOLVER.IMS_PER_BATCH = model_hyperparams['batch_num']
cfg.SOLVER.BASE_LR = model_hyperparams['learning_rate']
cfg.SOLVER.MAX_ITER = model_hyperparams['num_epochs']
cfg.SOLVER.STEPS = []        
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = model_hyperparams['batch_size_per_img']
cfg.MODEL.ROI_HEADS.NUM_CLASSES = model_info['num_of_classes'] 

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
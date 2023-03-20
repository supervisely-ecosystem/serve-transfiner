import os, sys
import cv2
import torch
import gdown

sys.path.append('transfiner')
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import MetadataCatalog
from transfiner.swinb import add_swinb_config
from transfiner.swint import add_swint_config
from src.model_zoo import model_zoo


def build_model(model_name, weights_path, device):
    model_info = model_zoo[model_name]
    cfg = get_cfg()
    if model_name == "Swin-B":
        add_swinb_config(cfg)
    elif model_name == "Swin-T":
        add_swint_config(cfg)
    cfg.merge_from_file(model_info["config"])
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = None
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = None
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = None
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = device
    predictor = DefaultPredictor(cfg)
    set_conf_thres(predictor.model, conf_thres=0.5)
    return predictor, cfg

def dump_cfg(cfg, output="model_config.yaml"):
    with open(output, "w") as f:
        f.write(cfg.dump())

def set_conf_thres(model, conf_thres):
    model.roi_heads.box_predictor.test_score_thresh = conf_thres

def set_nms_thres(model, nms_thres):
    # nms_thres is also called iou_thres
    model.roi_heads.box_predictor.test_nms_thresh = nms_thres

def download_weights(url, output):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    if not os.path.exists(output):
        gdown.download(url, output=output)

def get_coco_classes():
    meta = MetadataCatalog.get("coco_2017_val")
    return meta.thing_classes, meta.thing_colors


# TESTING
if __name__ == "__main__":
    for model_name, model_info in model_zoo.items():
        print(model_name, "start")

        weights_path = f"pretrained/{model_name}.pth"
        download_weights(model_info["weights_url"], weights_path)
        predictor, cfg = build_model(model_name, weights_path, "cuda")

        img = cv2.imread("demo_data/image_03.jpg")
        outputs = predictor(img)
        pred_classes = outputs["instances"].pred_classes.cpu().numpy()
        pred_scores = outputs["instances"].scores.cpu().numpy().tolist()
        pred_masks = outputs["instances"].pred_masks.cpu().numpy()

        import numpy as np
        print(len(pred_masks))
        for i, mask in enumerate(pred_masks):
            mask = (mask*255).astype(np.uint8)
            cv2.imwrite(f"pred_{model_name}_{i}.png", mask)
        print(model_name, "done.")

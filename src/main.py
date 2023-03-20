import os, sys
import numpy as np
import torch
from pathlib import Path
import cv2

import supervisely as sly
from dotenv import load_dotenv
try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal
from typing import List, Any, Dict

from src import transfiner_api
from src.model_zoo import model_zoo


load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
root_source_path = str(Path(__file__).parents[1])

class TransfinerModel(sly.nn.inference.InstanceSegmentation):
    def load_on_device(
        self,
        model_dir: str = None,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        if self.gui:
            self.model_name = self.gui.get_checkpoint_info()["Model"]
        else:
            self.model_name = "Swin-B"
            sly.logger.warn(f"GUI can't be used, default model is {self.model_name}.")

        model_info = model_zoo[self.model_name]
        self.device = device

        sly.logger.info(f"Downloading weights for {self.model_name}...")
        self.weights_path = f"{model_dir}/{self.model_name}.pth"
        transfiner_api.download_weights(model_info["weights_url"], self.weights_path)
        
        sly.logger.info(f"Building the model {self.model_name}...")
        self.predictor, self.cfg = transfiner_api.build_model(self.model_name, self.weights_path, self.device)

        default_conf_thres = self.custom_inference_settings_dict["conf_thres"]
        transfiner_api.set_conf_thres(self.predictor.model, default_conf_thres)

        self.class_names, colors = transfiner_api.get_coco_classes()  # 80 COCO classes
        obj_classes = [sly.ObjClass(name, sly.Bitmap, color) for name, color in zip(self.class_names, colors)]
        self._model_meta = sly.ProjectMeta(
            obj_classes=sly.ObjClassCollection(obj_classes),
            tag_metas=sly.TagMetaCollection([self._get_confidence_tag_meta()])
        )
        print(f"✅ Model has been successfully loaded on {device.upper()} device")

    def predict(self, image_path: str, settings: Dict[str, Any]) -> List[sly.nn.PredictionMask]:
        conf_thres = settings.get("conf_thres")
        if conf_thres is not None:
            transfiner_api.set_conf_thres(self.predictor.model, conf_thres)
        
        img = cv2.imread(image_path)
        outputs = self.predictor(img)
        pred_classes = outputs["instances"].pred_classes.cpu().numpy()
        pred_class_names = [self.class_names[pred_class] for pred_class in pred_classes]
        pred_scores = outputs["instances"].scores.cpu().numpy().tolist()
        pred_masks = outputs["instances"].pred_masks.cpu().numpy()

        results = []
        for score, class_name, mask in zip(pred_scores, pred_class_names, pred_masks):
            if np.any(mask):
                results.append(sly.nn.PredictionMask(class_name, mask, score))
        return results
    
    def get_models(self):
        models = []
        for name, info in model_zoo.items():
            cfg = os.path.basename(info["config"])
            info = info.copy()
            info.pop("config")
            models.append({"Model": name, "config": cfg, **info})
        return models

    def get_info(self):
        info = super().get_info()
        info["pretrained_on_dataset"] = "COCO"
        info["model_name"] = self.model_name
        info["device"] = self.device
        return info

    def get_classes(self) -> List[str]:
        return self.class_names
    
    def support_custom_models(self):
        return False


m = TransfinerModel(
    use_gui=True,
    custom_inference_settings=os.path.join(root_source_path, "custom_settings.yaml"),
)

if sly.is_production():
    m.serve()
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    m.load_on_device(m.model_dir, device)
    image_path = "./demo_data/image_01.jpg"
    results = m.predict(image_path, settings={})
    vis_path = "./demo_data/image_01_prediction.jpg"
    m.visualize(results, image_path, vis_path, thickness=0)
    print(f"predictions and visualization have been saved: {vis_path}")

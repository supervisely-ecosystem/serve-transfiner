import os, sys
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

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

class TransfinerModel(sly.nn.inference.SalientObjectSegmentation):
    def load_on_device(
        self,
        model_dir: str = None,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        self.model_name = "Swin-B"
        model_info = model_zoo[self.model_name]
        self.device = device

        sly.logger.info("Downloading weights...")
        self.weights_path = f"{model_dir}/{self.model_name}.pth"
        transfiner_api.download_weights(model_info["weights_url"], self.weights_path)
        
        sly.logger.info("Building the model...")
        self.predictor, self.cfg = transfiner_api.build_model(self.model_name, self.weights_path, self.device)

        default_conf_thres = self.custom_inference_settings_dict["conf_thres"]
        transfiner_api.set_conf_thres(self.predictor.model, default_conf_thres)

        self.coco_classes = transfiner_api.get_classes()  # the model seems had been trained on COCO 80 classes
        self.class_names = ["object_mask"]  # but we will test it as SalientObjectSegmentation
        print(f"✅ Model has been successfully loaded on {device.upper()} device")

    def predict(self, image_path: str, settings: Dict[str, Any]) -> List[sly.nn.PredictionMask]:
        conf_thres = settings.get("conf_thres")
        if conf_thres is not None:
            transfiner_api.set_conf_thres(self.predictor.model, conf_thres)
        
        img = cv2.imread(image_path)
        outputs = self.predictor(img)
        pred_classes = outputs["instances"].pred_classes.cpu().numpy()
        pred_scores = outputs["instances"].scores.cpu().numpy().tolist()
        pred_masks = outputs["instances"].pred_masks.cpu().numpy()

        res = [sly.nn.PredictionMask(class_name=self.class_names[0], mask=mask) for mask in pred_masks]
        return res
    
    def get_models(self):
        models = [{"Model": k, **v} for k, v in model_zoo.items()]
        return models

    def binarize_mask(self, mask, threshold):
        mask[mask < threshold] = 0
        mask[mask >= threshold] = 1
        return mask

    @property
    def model_meta(self):
        if self._model_meta is None:
            self._model_meta = sly.ProjectMeta(
                [sly.ObjClass(self.class_names[0], sly.Bitmap, [255, 0, 0])]
            )
            self._get_confidence_tag_meta()
        return self._model_meta

    def get_info(self):
        info = super().get_info()
        info["videos_support"] = False
        info["async_video_inference_support"] = False
        return info

    def get_classes(self) -> List[str]:
        return self.class_names


m = TransfinerModel(
    use_gui=True,
    custom_inference_settings=os.path.join(root_source_path, "custom_settings.yaml"),
)

if sly.is_production():
    m.serve()
else:
    m.load_on_device(m.model_dir, device)
    image_path = "./demo_data/image_03.jpg"
    # rect = sly.Rectangle(360, 542, 474, 700).to_json()
    # ann = m._inference_image_path(image_path=image_path, settings={"rectangle": rect, "bbox_padding":"66%"}, data_to_return={})
    # ann.draw_pretty(sly.image.read(image_path), [255,0,0], 7, output_path="out.png")
    results = m.predict(image_path, settings={})
    vis_path = "./demo_data/image_03_prediction.jpg"
    m.visualize(results, image_path, vis_path, thickness=0)
    print(f"predictions and visualization have been saved: {vis_path}")

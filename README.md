<div align="center" markdown>
<img src="https://user-images.githubusercontent.com/115161827/228323903-8661cf1b-e0f7-4793-810f-87a5545c2323.jpg"/>  

# Serve Transfiner for Instance Segmentation

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#Pretrained-Models">Pretrained Models</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Related-apps">Related Apps</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/serve-transfiner)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/serve-transfiner)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/serve-transfiner.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/serve-transfiner.png)](https://supervise.ly)

</div>

## Overview

This app deploys a pretrained **Transfiner** model as a Supervisely Application for **Instance Segmentation**.

The app allows you to apply the model to an image inside the **Supervisely platform** or beyond it (using the [Inference Session API](https://developer.supervise.ly/app-development/neural-network-integration/inference-api-tutorial)).


## Pretrained Models


| Backbone(configs)  | Method | mAP(mask) | 
|----------|--------|-----------|
[R50-FPN](configs/transfiner/mask_rcnn_R_50_FPN_3x.yaml) | Transfiner (3x, CVPR'22)  | 39.2,  [Pretrained Model](https://drive.google.com/file/d/1EA9pMdUK6Ad9QsjaZz0g5jqbo_JkqtME/view?usp=sharing)|
[R50-FPN-DCN](configs/transfiner/mask_rcnn_R_50_FPN_3x_deform.yaml) | Transfiner (3x, CVPR'22)  | 40.5, [Pretrained Model](https://drive.google.com/file/d/1N0C_ZhES7iu8qEPG2mrdxf8rWteemxQD/view?usp=sharing) |
[R101-FPN](configs/transfiner/mask_rcnn_R_101_FPN_3x.yaml) | Transfiner (3x, CVPR'22) | 40.5, [Pretrained Model](https://drive.google.com/file/d/1Jn27jTpFFWjuX22xvR1upP99nOXfZ1nk/view?usp=sharing) | 
[R101-FPN-DCN](configs/transfiner/mask_rcnn_R_101_FPN_3x_deform.yaml) | Transfiner (3x, CVPR'22) | 42.2, [Pretrained Model](https://drive.google.com/file/d/1TpVQksuaXlhioD3WqWppX84MB-l_Eb7-/view?usp=sharing) | 
[Swin-T](configs/transfiner/mask_rcnn_swint_FPN_3x.yaml) | Transfiner | 43.5, [Pretrained Model](https://drive.google.com/file/d/1ezIxmwdMl_cC7gCPEqtLL6zlSYd3R9wA/view?usp=sharing) |
[Swin-B](configs/transfiner/mask_rcnn_swinb_FPN_3x.yaml) | Transfiner | **45.5**, [Pretrained Model](https://drive.google.com/file/d/1XkEwTMiyADYfvniIrBIDX7RPTSLI4fys/view?usp=sharing) |



### Prediction preview (Swin-B):
![prediction preview (Swin-B)](https://raw.githubusercontent.com/supervisely-ecosystem/serve-transfiner/master/demo_data/image_01_prediction.jpg)


## How To Run

1. Start the application from Ecosystem
2. Open the app in your browser

<img src="https://user-images.githubusercontent.com/31512713/228284304-0f498122-80f9-4cb4-8734-21927612e542.png" width="80%"/>

3. Select the model and click **"SERVE"** button

{% hint style="success" %}
That's it, now you can use other apps with your deployed model!
{% endhint %}


## Related Apps

You can use a served model in the following Supervisely Applications ⬇️ :

- [Apply NN to Images Project](https://ecosystem.supervise.ly/apps/nn-image-labeling/project-dataset) - app allows to play with different inference options and visualize predictions in real time.  Once you choose inference settings you can apply model to all images in your project to visually analyse predictions and perform automatic data pre-labeling.   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/project-dataset" src="https://i.imgur.com/M2Tp8lE.png" height="70px" margin-bottom="20px"/>  

- [Apply NN to Videos Project](https://ecosystem.supervise.ly/apps/apply-nn-to-videos-project) - app allows to label your videos using served Supervisely models.  
  <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/apply-nn-to-videos-project" src="https://imgur.com/LDo8K1A.png" height="70px" margin-bottom="20px" />

- [NN Image Labeling](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fannotation-tool) - integrate any deployed NN to Supervisely Image Labeling UI. Configure inference settings and model output classes. Press `Apply` button (or use hotkey) and detections with their confidences will immediately appear on the image.   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/annotation-tool" src="https://i.imgur.com/hYEucNt.png" height="70px" margin-bottom="20px"/>


---

## Acknowledgment

- Based on: [https://github.com/SysCV/transfiner](https://github.com/SysCV/transfiner) ![GitHub Org's stars](https://img.shields.io/github/stars/SysCV/transfiner?style=social)
- Paper: [https://arxiv.org/abs/2111.13673](https://arxiv.org/abs/2111.13673)

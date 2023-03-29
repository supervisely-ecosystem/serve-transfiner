<div align="center" markdown>
<img src="https://user-images.githubusercontent.com/115161827/228323903-8661cf1b-e0f7-4793-810f-87a5545c2323.jpg"/>  

# Serve Transfiner for Instance Segmentation

<p align="center">
  <a href="#Overview">Overview</a> •
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

# Overview

///

# How to Run

**Step 1.** Run the app from the Ecosystem

**Step 2.** Select pretrained model, click **Serve**

<img src="https://user-images.githubusercontent.com/115161827/228323630-dda54876-8ca5-411a-9418-7834a62e2e8e.png" width="80%">  

**Step 3.** Wait for the model to deploy

# Related Apps

You can use served model in next Supervisely Applications ⬇️ 
  

- [Apply NN to Images Project](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fproject-dataset) - app allows to play with different inference options and visualize predictions in real time.  Once you choose inference settings you can apply model to all images in your project to visually analyse predictions and perform automatic data pre-labeling.   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/project-dataset" src="https://i.imgur.com/M2Tp8lE.png" height="70px" margin-bottom="20px"/>  

- [Apply NN to Videos Project](https://ecosystem.supervise.ly/apps/apply-nn-to-videos-project) - app allows to label your videos using served Supervisely models.  
  <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/apply-nn-to-videos-project" src="https://imgur.com/LDo8K1A.png" height="70px" margin-bottom="20px" />

- [NN Image Labeling](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fannotation-tool) - integrate any deployd NN to Supervisely Image Labeling UI. Configure inference settings and model output classes. Press `Apply` button (or use hotkey) and detections with their confidences will immediately appear on the image.   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/annotation-tool" src="https://i.imgur.com/hYEucNt.png" height="70px" margin-bottom="20px"/>


# Acknowledgment

This app is based on the great work `Mask Transfiner` ([github](https://github.com/SysCV/transfiner)). ![GitHub Org's stars](https://img.shields.io/github/stars/SysCV/transfiner?style=social)
FROM supervisely/base-py-sdk:6.73.418

RUN pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

RUN pip3 install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

RUN git clone https://github.com/supervisely-ecosystem/detectron2

RUN pip3 install timm gdown
RUN pip3 install supervisely==6.73.486

LABEL python_sdk_version="6.73.486"
# T3-Net

# Introduce
  
This is the official code for the paper Visible and Clear: Finding Tiny Objects in Difference Map.


**NOTE: Our paper has been accepted by ECCV 2024.**

# Environment

To use the AI-TOD evaluation metrics, you need to download aitodpycocotools. You can install it using the following command:

```shell
pip install "git+https://github.com/jwwangchn/cocoapi-aitod.git#subdirectory=aitodpycocotools"
```

# Training and Test
The training and test commands can also be referenced from mmdetection.

1 gpu:

```shell
python tools/train.py ./srtod_project/srtod_cascade_rcnn/config/srtod-cascade-rcnn_r50_fpn_1x_coco.py
```
```shell
python tools/test.py ./srtod_project/srtod_cascade_rcnn/config/srtod-cascade-rcnn_r50_fpn_1x_coco.py your_model.pth
```

If you need to use more GPUs, you should use ./tools/dist_train.sh instead of tools/train.py.

# DroneSwarms
If you want to access the DroneSwarms dataset, please visit the following link：[DroneSwarms](https://hiyuur.github.io)



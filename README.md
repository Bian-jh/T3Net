# A Unified Token Reliability Modeling Framework for Enhanced Tiny Object Detection

## Preparation

Install aitodpycocotools to use the AI-TOD evaluation metrics:

```shell
pip install "git+https://github.com/jwwangchn/cocoapi-aitod.git#subdirectory=aitodpycocotools"
```

## Training process
Before start training, modify parameters in directory /configs.
We recommend 2 gpus for VisDrone, 3 or 4 gpus for AI-TODV2.

1. Stage 1
```
CUDA_VISIBLE_DEVICES=<gpu ids> accelerate launch main_enhance_aitod_pre.py
```
2. Stage 2
```
CUDA_VISIBLE_DEVICES=<gpu ids> accelerate launch main_enhance_aitod.py
```

## Acknowledgement
Our code is built upon [Salience-DETR](https://github.com/xiuqhou/Salience-DETR), thanks for their inspiring work!



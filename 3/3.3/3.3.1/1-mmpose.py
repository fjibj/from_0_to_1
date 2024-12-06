#1．基于热力图的方法（修改）
#以下是使用mmpose库加载HRNet模型并进行2D姿态估计的python代码示例。
#!pip install mmpose mmcv-full opencv-python
import cv2
import numpy as np
import torch
from mmpose.apis import (init_pose_model, inference_top_down_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo
from mmpose.datasets.pipelines import Compose

# 配置文件路径
config_file = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py'
# 预训练模型权重路径
checkpoint_file = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-6e6b7ec6_20200708.pth'

# 初始化HRNet模型
model = init_pose_model(config_file, checkpoint_file, device='cuda:0')

# 读取输入图像
image_path = 'input_image.jpg'
image = cv2.imread(image_path)

# 生成检测结果
person_results = [{'bbox': [50, 50, 200, 200]}]  # 示例中假设已知人体边界框（bbox），实际应用中需要用检测器获得

# 推理2D姿态
pose_results, _ = inference_top_down_pose_model(
    model,
    image,
    person_results,
    bbox_thr=0.3,
    format='xyxy',
    dataset='TopDownCocoDataset',
    dataset_info=None,
    return_heatmap=False,
    outputs=None
)

# 可视化结果
vis_result = vis_pose_result(
    model,
    image,
    pose_results,
    dataset='TopDownCocoDataset',
    kpt_score_thr=0.3,
    show=False
)

# 显示带有姿态估计结果的图像
cv2.imshow('Pose Estimation', vis_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
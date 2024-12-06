#3．基于转换器的方法
#以下是使用TokenPose进行2D姿态估计的简化代码示例
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from timm import create_model

# 定义TokenPose模型类
class TokenPoseModel(torch.nn.Module):
    def __init__(self, model_name='tokenpose_s', num_keypoints=17):
        super(TokenPoseModel, self).__init__()
        self.model = create_model(model_name, pretrained=False, num_classes=num_keypoints*2)

    def forward(self, x):
        output = self.model(x)
        output = output.reshape(-1, 17, 2)  # 假设输出每个关键点的 (x, y) 坐标
        return output

# 初始化TokenPose模型
model = TokenPoseModel(model_name='tokenpose_s')
# 加载预训练权重（需要本地文件路径）
model.load_state_dict(torch.load('tokenpose_s_coco.pth'))
model.eval()

# 读取输入图像
image_path = 'input_image.jpg'
image = cv2.imread(image_path)
orig_height, orig_width = image.shape[:2]

# 预处理图像
input_size = (256, 192)  # TokenPose模型的输入大小
image_resized = cv2.resize(image, input_size)
image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

# 转换为Tensor并归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image_tensor = transform(image_rgb).unsqueeze(0)  # 添加批次维度

# 前向传播，获得关键点坐标序列
with torch.no_grad():
    keypoints = model(image_tensor)

# 将关键点映射回原始图像尺寸
keypoints = keypoints.squeeze(0).cpu().numpy()
keypoints[:, 0] *= (orig_width / input_size[1])
keypoints[:, 1] *= (orig_height / input_size[0])

# 输出姿态估计结果
print("Estimated keypoints:", keypoints)

# 可视化关键点
for coord in keypoints:
    cv2.circle(image, (int(coord[0]), int(coord[1])), 5, (0, 255, 0), -1)

# 显示带有关键点的图像
cv2.imshow("Pose Estimation", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#更多TokenPose相关的训练和推理请参考https://github.com/leeyegy/TokenPose
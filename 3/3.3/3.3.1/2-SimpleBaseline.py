#2．基于关键点回归的方法（修改）
#以下是使用SimpleBaseline进行2D姿态估计的代码示例。
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import resnet50

# 定义SimpleBaseline模型类
class SimpleBaseline(torch.nn.Module):
    def __init__(self, backbone, num_keypoints=17):
        super(SimpleBaseline, self).__init__()
        self.backbone = backbone
        self.deconv_layers = self._make_deconv_layers()
        self.final_layer = torch.nn.Conv2d(
            in_channels=256,
            out_channels=num_keypoints,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def _make_deconv_layers(self):
        layers = []
        for _ in range(3):
            layers.append(torch.nn.ConvTranspose2d(2048 if _ == 0 else 256, 256, kernel_size=4, stride=2, padding=1))
            layers.append(torch.nn.BatchNorm2d(256))
            layers.append(torch.nn.ReLU(inplace=True))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        retur: x

# 加载ResNet-50作为骨干网络
backbone = resnet50(pretrained=True)
# 去掉最后的全连接层
backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

# 初始化SimpleBaseline模型
model = SimpleBaseline(backbone)
# 加载预训练权重（需要本地文件路径）
model.load_state_dict(torch.load('simplebaseline_res50_coco.pth'))
model.eval()

# 定义输入图像路径
image_path = 'input_image.jpg'
image = cv2.imread(image_path)

# 预处理图像
input_size = (256, 192)
image_resized = cv2.resize(image, input_size)
image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

# 转换为Tensor并归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image_tensor = transform(image_rgb).unsqueeze(0)

# 前向传播，获得关键点坐标序列
with torch.no_grad():
    heatmaps = model(image_tensor)

# 后处理:将热图转换为关键点坐标
heatmaps = heatmaps.squeeze(0).cpu().numpy()
num_keypoints = heatmaps.shape[0]

keypoint_coords = []
for i in range(num_keypoints):
    heatmap = heatmaps[i]
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    keypoint_coords.append((x, y))

# 将关键点坐标映射回原始图像尺寸
keypoint_coords = np.array(keypoint_coords)
scale_x = image.shape[1] / input_size[1]
scale_y = image.shape[0] / input_size[0]
keypoint_coords[:, 0] *= scale_x
keypoint_coords[:, 1] *= scale_y

# 输出姿态估计结果
print("Estimated keypoints:", keypoint_coords)

# 可视化关键点
for coord in keypoint_coords:
    cv2.circle(image, (int(coord[0]), int(coord[1])), 5, (0, 255, 0), -1)

# 显示带有关键点的图像
cv2.imshow("Pose Estimation", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#更多SimpleBaseline相关内容请参考https://github.com/microsoft/human-pose-estimation.pytorch
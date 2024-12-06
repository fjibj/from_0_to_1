#3．基于风格迁移的表情融合（部分修改）
#下面是一个简化的代码示例，演示了如何使用PyTorch库进行神经风格迁移，将一个图像的风格应用到另一个图像上。
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image

# 加载内容图像和风格图像
content_image = Image.open("content.jpg")
style_image = Image.open("style.jpg")

# 转换图像大小并对其进行规范化
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
content_tensor = preprocess(content_image).unsqueeze(0) # 添加批次维度
style_tensor = preprocess(style_image).unsqueeze(0)

# 使用GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将图像移动到GPU
content_tensor = content_tensor.to(device)
style_tensor = style_tensor.to(device)

# 加载预训练的VGG模型，用于特征提取
vgg = models.vgg19(pretrained=True).features.to(device).eval()

# 定义损失函数，包括内容损失和风格损失
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    def forward(self, x):
        loss = nn.functional.mse_loss(x, self.target)
        return loss

class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target).detach()
    def forward(self, x):
        G = self.gram_matrix(x)
        loss = nn.functional.mse_loss(G, self.target)
        return loss
    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

# 定义内容损失和风格损失计算模块
content_criterion = ContentLoss(content_tensor)
style_criterion = StyleLoss(style_tensor)

# 定义生成图像，初始为内容图像的副本
generated_image = content_tensor.clone().requires_grad_(True)

# 定义优化器
optimizer = optim.LBFGS([generated_image])

# 定义损失函数权重
content_weight = 1  # 调整权重以控制内容与风格之间的平衡
style_weight = 1000

# 迭代优化过程
num_steps = 300
for step in range(num_steps):
    def closure():
        optimizer.zero_grad()
        # 获取模型的特征图
        content_features = vgg(content_tensor)
        style_features = vgg(style_tensor)
        generated_features = vgg(generated_image)

        # 计算内容损失
        content_loss = content_weight * content_criterion(generated_features, content_features)
        # 计算风格损失
        style_loss = style_weight * style_criterion(generated_features, style_features)
        # 总损失
        total_loss = content_loss + style_loss
        total_loss.backward()
        return total_loss

    optimizer.step(closure)

# 将生成的图像从张量（矩阵格式）转换回图像格式，为显示图像做准备
output_image = generated_image.squeeze(0).cpu().clone()
output_image = output_image.clamp(0, 1)
output_image = transforms.ToPILImage()(output_image)

# 显示融合后的图像
output_image.show()
#更多请参考https://github.com/gordicaleksa/pytorch-neural-style-transfer
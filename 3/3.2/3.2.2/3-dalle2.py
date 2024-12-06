#3．基于迁移学习的表情生成（修改）
#以下是一个简化的Python代码示例，使用图像生成模型laion/DALLE2-PyTorch来生成数字人表情。
!pip install dalle2-pytorch

import torch
from torchvision.transforms import ToPILImage
from dalle2_pytorch import DiffusionPrior, DiffusionPriorNetwork, OpenAIClipAdapter, Decoder, DALLE2
from dalle2_pytorch.train_configs import TrainDiffusionPriorConfig, TrainDecoderConfig

prior_config = TrainDiffusionPriorConfig.from_json_path("weights/prior_config.json").prior
prior = prior_config.create().cuda()

prior_model_state = torch.load("weights/prior_latest.pth")
prior.load_state_dict(prior_model_state, strict=True)

decoder_config = TrainDecoderConfig.from_json_path("weights/decoder_config.json").decoder
decoder = decoder_config.create().cuda()

decoder_model_state = torch.load("weights/decoder_latest.pth")["model"]

for k in decoder.clip.state_dict().keys():
    decoder_model_state["clip." + k] = decoder.clip.state_dict()[k]

decoder.load_state_dict(decoder_model_state, strict=True)

dalle2 = DALLE2(prior=prior, decoder=decoder).cuda()

images = dalle2(
    ['一个带笑脸的数字人'],
    cond_scale = 2.
).cpu()

print(images.shape)

for img in images:
    img = ToPILImage()(img)
    img.show()
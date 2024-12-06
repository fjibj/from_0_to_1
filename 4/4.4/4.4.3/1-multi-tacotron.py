#2．多说话人模型
#这里给出一个使用说话人条件来实现多说话人语音合成的 Python 代码示例。
import torch
import torch.nn as nn
# 定义多说话人Tacotron模型
class MultiSpeakerTacotron(nn.Module) ：
 def __init__(self) ：
  super().__init__()

  # 文本编码器
  self.text_encoder = TextEncoder()

  # 声学特征解码器
  self.decoder = AcousticDecoder()

  # 嵌入层，获得说话人条件嵌入向量
  self.spk_emb = nn.Embedding(num_speakers, spk_emb_dim)

 def forward(self, text, spk_id) ：
  # 对文本进行编码
  text_fea = self.text_encoder(text)

  # 获取说话人嵌入向量
  spk_emb = self.spk_emb(spk_id)

  # 将文本特征和说话人嵌入向量拼接
  cond_input = torch.cat([text_fea, spk_emb], dim=-1)

  # 解码得到声学特征
  mel_spect = self.decoder(cond_input)

  return mel_spect

# 实例化多说话人模型
model = MultiSpeakerTacotron()
# 输入文本序列
text = "This is an example."
# 输入说话人ID，比如0、1、2等
spk_id = torch.LongTensor([1])
# 预测对应的语音特征
mel = model(text, spk_id)
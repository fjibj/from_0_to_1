#2．模型训练过程
#（1）文本特征处理
import numpy as np
chars = "this is some text"
char_indices = dict((c, i) for i, c in enumerate(set(chars)))
indices = [char_indices[c] for c in chars]
embedding_dim = 20
embedding_matrix = np.random.randn(len(char_indices), embedding_dim)
char_embeds = embedding_matrix[indices]

#（2）音频特征提取
#以下是音频特征提取代码，这里主要使用第三方库 librosa 来提取 MFCC 特征。
import librosa
def get_spectrograms(sound_file):
 # 加载声音文件
 y, sr = librosa.load(sound_file, sr=hp.sr) # or set sr to hp.sr.
 # 短时傅里叶变换
 D = librosa.stft(y=y,
 n_fft=hp.n_fft,
 hop_length=hp.hop_length,
 win_length=hp.win_length)
 # 幅度谱图
 magnitude = np.abs(D)
 # 功率谱图
 power = magnitude**2
 # 梅尔谱图
 S = librosa.feature.melspectrogram(S=power, n_mels=hp.n_mels)
 return np.transpose(S.astype(np.float32)), np.transpose(magnitude.astype(np.float32))

#（3）端到端监督训练
from keras.layers import Attention, Dense, LSTM
from keras.models import Model
encoder = LSTM(...) # 编码器
decoder = LSTM(...) # 解码器
attn = Attention(...) # 注意力层
model = Model([encoder, decoder, attn], Dense(n_linear))
model.compile(loss='mse', ...)
model.fit([char_embeds, linear_spect], linear_spect, ...) # 端到端训练

linear_pred = model.predict(char_embeds)
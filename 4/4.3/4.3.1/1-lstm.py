#2．LSTM 带来的优势
#（3）代码实现
#基于 LSTM 的神经网络语音合成系统的 Python 实现代码如下。
import numpy as np
import librosa
from keras.layers import LSTM, Dense
from keras.models import Sequential
# 载入语音样本并提取MFCC特征
audio, sr = librosa.load("speech.wav", sr=16000)
mfcc = librosa.feature.mfcc(audio, sr=sr)
# 构建LSTM编码器-解码器模型
model = Sequential()
model.add(LSTM(128, input_shape=(None, 20), return_sequences=True)) # 编码器
model.add(LSTM(128, return_sequences=True)) # 解码器
model.add(Dense(20, activation='sigmoid')) # 输出层
# 训练模型参数
model.compile(loss='mse', optimizer='adam')
model.fit(mfccs, mfccs, epochs=10)
# 预测语音参数
mfcc_pred = model.predict(mfccs)
# 通过GL算法合成语音波形
audio_pred = librosa.griffinlim(mfcc_pred, n_iter=30)
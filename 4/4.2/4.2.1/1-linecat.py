#2．拼接方法
#（1）线性拼接
import librosa
# 加载两个音频片段
audio1, sr = librosa.load('audio1.wav')
audio2, sr = librosa.load('audio2.wav')
# 简单线性拼接
concat = np.concatenate((audio1, audio2))
# 保存拼接结果
librosa.output.write_wav('linear_concat.wav', concat, sr)

#（2）叠加拼接
import numpy as np
# 加载音频片段
audio1, sr = librosa.load('audio1.wav')
audio2, sr = librosa.load('audio2.wav')
# 计算拼接处的重叠长度
overlap = int(sr * 0.01)
# 汉明窗加权叠加
window = np.hamming(overlap)
concat = np.concatenate((audio1[ :-overlap],
 audio1[-overlap :]*window + audio2[ :overlap]*window,
 audio2[overlap :]))
# 保存拼接语音
librosa.output.write_wav('overlap_concat.wav', concat, sr)

#（3）多音素拼接
import librosa
import numpy as np
# 加载3段音频
audio1, sr = librosa.load('audio1.wav')
audio2, sr = librosa.load('audio2.wav')
audio3, sr = librosa.load('audio3.wav')
# 重叠区线性混合
concat = np.concatenate((audio1[ :-2],
 0.5*audio1[-2 :] + 0.5*audio2[ :2],
 audio2[2 :-2],
 0.5*audio2[-2 :] + 0.5*audio3[ :2],
 audio3[2 :]))
# 保存拼接语音
librosa.output.write_wav('multi_concat.wav', concat, sr)
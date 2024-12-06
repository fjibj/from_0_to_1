#3．正弦编码
#（1）基频检测
import parselmouth
from parselmouth.praat import call
sound = parselmouth.Sound("speech.wav")
pitch = call(sound, "To Pitch", 0.0, 75, 600)
# 提取基频曲线
pitch_values = pitch.selected_array['frequency']

#（2）幅度谱建模
import librosa
import numpy as np
# 提取语音的幅度谱
amp_spect = np.abs(librosa.stft(speech))
# LPC预测全波段幅度谱
lpc_model = librosa.core.lpc(amp_spect, order=10)

#（3）基频编码
from scipy.signal import quantize
# 量化基频参数
quant_pitch = quantize(pitch_values, 64, 'log')
# 基频参数的矢量量化
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=32)
kmeans.fit(pitch_values[ ：, np.newaxis])

#（4）幅度谱编码
import librosa
from sklearn.cluster import KMeans
# 幅度谱矢量量化
kmeans = KMeans(n_clusters=16)
kmeans.fit(amp_spect)
# 幅度谱LPC编码
lpc_coeffs = librosa.core.lpc(amp_spect, order=12)
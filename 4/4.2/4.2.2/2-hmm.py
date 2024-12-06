#2．隐马尔可夫模型
#（2）实现代码
import numpy as np
import librosa
from hmmlearn import hmm
def smooth_with_hmm(audio, n_states, transition_prob) ：
 model = hmm.GaussianHMM(n_components=n_states, covariance_type='diag')

 # 训练HMM模型
 model.fit(audio.reshape(-1, 1))

 # 预测状态序列
 _, states = model.decode(audio.reshape(-1, 1))

 # 进行状态转移平滑
 smoothed_audio = np.copy(audio)
 for i in range(1, len(audio)) ：
 smoothed_audio[i] = smoothed_audio[i-1] * transition_prob[states[i]]

 return smoothed_audio
# 示例音频
filename = 'your_audio_file.wav'
audio, sr = librosa.load(filename, sr=None)
# 设置HMM模型的状态数和状态转移概率，示例中简化为常数
n_states = 5
transition_prob = np.array([0.95, 0.9, 0.85, 0.9, 0.95])
# 进行状态转移平滑
smoothed_audio = smooth_with_hmm(audio, n_states, transition_prob)
# 保存合成的声音
output_filename = 'smoothed_audio_hmm.wav'
librosa.output.write_wav(output_filename, smoothed_audio, sr)
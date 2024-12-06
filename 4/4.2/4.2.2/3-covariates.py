#3．协变量回归
#（2）实现代码
import numpy as np
import librosa
def smooth_with_covariates(audio, covariates) ：
 smoothed_audio = np.copy(audio)

 # 根据协变量信息进行声音合成调整
 for i in range(1, len(audio)) ：
 smoothed_audio[i] = smoothed_audio[i-1] * covariates[i]

 return smoothed_audio
# 示例音频
filename = 'your_audio_file.wav'
audio, sr = librosa.load(filename, sr=None)
# 示例协变量，示例中协变量简化为线性变化
covariates = np.linspace(0.8, 1.2, len(audio))
# 进行声音合成调整
smoothed_audio = smooth_with_covariates(audio, covariates)
# 保存合成的声音
output_filename = 'smoothed_audio_covariates.wav'
librosa.output.write_wav(output_filename, smoothed_audio, sr)
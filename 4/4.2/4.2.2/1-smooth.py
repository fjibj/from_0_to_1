#1．最大似然连续化方法
#（2）实现代码
import numpy as np
import librosa
def smooth_transitions(audio, transition_prob) ：
 smoothed_audio = np.copy(audio)

 # 进行状态转移平滑
 for i in range(1, len(audio)) ：
 smoothed_audio[i] = smoothed_audio[i-1] * transition_prob

 return smoothed_audio
# 示例音频
filename = 'your_audio_file.wav'
audio, sr = librosa.load(filename, sr=None)
# 设置状态转移概率，示例中简化为一个常数
transition_prob = 0.95
# 进行状态转移平滑
smoothed_audio = smooth_transitions(audio, transition_prob)
# 保存合成的声音
output_filename = 'smoothed_audio.wav'
librosa.output.write_wav(output_filename, smoothed_audio, sr)
#1．采样频率选择
#对语音合成任务而言，16kHz 采样率已经很好地平衡了语音质量与存储效率。我们可以用librosa 库来加载 16kHz 采样的语音。
import librosa
audio_path = 'english.wav'
Samples, sample_rate = librosa.load(audio_path， sr=16000)

#也可以用 pydub 库实现音频重采样。
from pydub import AudioSegment
# 加载音频文件
audio = AudioSegment.from_file("input.wav", format="wav")
# 将采样率设置为16000Hz
audio = audio.set_frame_rate(16000)
# 导出重采样后的音频文件
audio.export("output.wav", format="wav")
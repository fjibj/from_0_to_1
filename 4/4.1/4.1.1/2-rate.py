#2．量化位数选择
#使用Python的librosa库可以方便地读取16bit@16kHz语音。
import librosa
Samples, sample_rate = librosa.load("speech.wav", sr=16000)
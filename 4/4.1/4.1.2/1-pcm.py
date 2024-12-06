#1．PCM 编码
#（3）µ法则与A法则
#在 Python 中 scipy 库提供了这两种算法的编码实现。
import librosa
from scipy.io import wavfile

# μ法则编码
encoded = librosa.core.codec.mu_encode(samples, mu=255)

# A法则编码
encoded = librosa.core.codec.a_encode(samples, a=87)
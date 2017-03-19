import librosa
import numpy as np

time_series, sample_rate = librosa.load("./7a.wav")
spectrogram = librosa.feature.melspectrogram(time_series, sr=sample_rate, n_mels=23)
# Convert to log scale (dB). We'll use the peak power as reference.
log_S = librosa.logamplitude(spectrogram, ref_power=np.max)
np.savetxt('7a.txt', log_S.T)

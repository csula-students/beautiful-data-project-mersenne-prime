from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import wave
import numpy as np
from numpy.lib.stride_tricks import as_strided
from collector import collector
# Generate plots of waveform

wav = wave.open("datasets/TUT-acoustic-scenes-2016-development/audio/a001_90_120.wav")
sample_rate = wav.getframerate()
chan = wav.getnchannels()
nframes = wav.getnframes()
width = wav.getsampwidth()
print(width)
frames = wav.readframes(nframes)
dt=np.dtype(np.int16)
dt=dt.newbyteorder('<')
audio = np.frombuffer(frames,count=1323001,dtype=dt)

# Normalize data
audio = audio / (2.**15)
timeArray = np.arange(0,audio.shape[0],1)
timeArray = timeArray / sample_rate
print(timeArray)
timeArray = timeArray * 1000

co = collector()

result = np.asarray(co.get_feature_vector_array(sound_scene='residential_area'))

plt.figure()
plt.plot(timeArray[:len(timeArray)/300],audio[0:audio.shape[0]/300])
## label the axes
plt.ylabel("Amplitude")
plt.xlabel("Time (ms)")
## set the title
plt.title("Residential Area")
plt.savefig("res_area_waveform.png")

print(result[0].T.shape)
plt.figure()
plt.imshow(result[0].T,aspect="auto",interpolation="none")
plt.title("Residential Area MFCC Features")
plt.xlabel("Frame")
plt.ylabel("Dimension")
plt.savefig("res_area_mfcc.png")



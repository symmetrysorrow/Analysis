import general as gn
import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy

def NoiseCheck():
    path=gn.InputPath()
    json=gn.LoadJson(f"{path}/PulseConfig.json")

    noise_pathes=glob.glob(f"{path}/CH*_noise/rawdata/CH*.dat")
    noise_sample=gn.LoadBin(noise_pathes[3])
    noise_sample=gn.Bessel(noise_sample,json["Readout"]["Rate"],json["Analysis"]["CutoffFrequency"])
    noise_sample = scipy.signal.medfilt(noise_sample, kernel_size=15)
    noise_sample-=np.mean(noise_sample)

    noise_rfft = np.fft.rfft(noise_sample)
    noise_amp=np.abs(noise_rfft)
    noise_time=gn.GN(noise_rfft)
    plt.plot(noise_sample,label="Original Noise")
    plt.plot(noise_time,label="Generated Noise")
    plt.legend()
    plt.show()

    noise_sample_fft=np.abs(np.fft.rfft(noise_sample))
    plt.plot(noise_sample_fft,label="Original Noise FFT")
    plt.plot(noise_amp,label="Model Noise FFT")
    plt.legend()
    plt.loglog()
    plt.show()

NoiseCheck()




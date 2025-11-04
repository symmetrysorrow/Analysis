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

    print(f"sample len:{len(noise_sample)}")
    print(f"fft len:{len(noise_rfft)}")
    print(f"time len:{len(noise_sample)}")
    

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

def PulseNoiseRFFT():
    path=gn.InputPath()
    noise_folder=glob.glob(f"{path}/CH*_noise")[0]
    noise_model=np.loadtxt(f"{noise_folder}/noise_fft_Amplitude.txt")
    noise_model=noise_model[:len(noise_model)//2+1]

    pulse_folder=glob.glob(f"{path}/CH*_pulse")[0]
    pulse=gn.LoadBin(f"{pulse_folder}/rawdata/CH0_26.dat")
    pulse_rfft=np.fft.rfft(pulse)
    pulse_amp=np.abs(pulse_rfft)

    plt.plot(pulse_amp,label="Pulse FFT")
    plt.plot(noise_model,label="Noise Model FFT")
    plt.legend()
    plt.loglog()
    plt.show()

def NoiseModelCompare():
    eta=108
    #path=gn.InputPath()
    path="G:/tagawa/20241206/r1ch12_215mK_1400uA1400uA_difftrig5e-5_rate500k_samples100k_gain5_day2"
    json=gn.LoadJson(f"{path}/PulseConfig.json")
    rate=json["Readout"]["Rate"]
    cutoff=json["Analysis"]["CutoffFrequency"]
    sample=json["Readout"]["Sample"]
    noise_folder=glob.glob(f"{path}/CH*_noise")[0]
    noise_model=np.loadtxt(f"{noise_folder}/modelnoise.txt")

    noise_pathes=glob.glob(f"{path}/CH*_noise/rawdata/CH*.dat")

    fq=gn.GetFreq(rate/2,sample/2+1)

    for i in [13,19]:
        noise=gn.LoadBin(noise_pathes[i])
        if len(noise)!=sample:
            continue
        noise = gn.Bessel(noise, rate, cutoff)
        noise_fft = np.fft.fft(noise)
        noise_amp = np.abs(noise_fft)

        df=rate/sample
        power=noise_amp**2 / df
        amp_dens=np.sqrt(power)
        amp_dens = amp_dens[: int(sample / 2) + 1] * eta * 1e+6
        
        plt.plot(fq,amp_dens,label=f"sample - {i}")

    fq= np.linspace(0, rate / 2, 50001)

    print(len(fq),len(noise_model))

    plt.plot(fq,noise_model,label="Average Noise",linestyle="--")
    plt.xlabel("Frequency[Hz]")
    plt.ylabel("Intensity[pA/Hz$^{1/2}$]")
    plt.legend()
    plt.loglog()
    plt.show()

    time=gn.GetTime(rate,sample)

    for i in [13,19]:
        noise=gn.LoadBin(noise_pathes[i])
        if len(noise)!=sample:
            continue
        noise = gn.Bessel(noise, rate, cutoff)
        noise-=np.mean(noise)

        plt.plot(time,noise,label=f"sample - {i}",color="orange")
        plt.ylim(-0.03,0.03)
        plt.xlabel("Time[s]")
        plt.ylabel("Intensity[V]")
        plt.show()

    #13ue
    #17sita
    #22good sita


NoiseModelCompare()
#PulseNoiseRFFT()
#NoiseCheck()
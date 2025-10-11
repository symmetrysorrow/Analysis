import general
import os
import shutil
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import questionary
import tqdm
import scipy
import re
import natsort

def PulseAnalysis(config: dict, path: str):
    for folder in glob.glob(f"{path}/CH*_pulse"):
        pulse_pathes = glob.glob(f"{folder}/rawdata/CH*.dat")
        pulse_pathes = natsort.natsorted(pulse_pathes)
        results = []

        for pulse_path in tqdm.tqdm(pulse_pathes):
            # ファイル名からキーを抽出
            filename = os.path.basename(pulse_path)
            key = os.path.splitext(filename)[0].split("_")[-1]
            
            pulse = general.LoadBin(pulse_path)
            result = general.AnalyzePulse(pulse, config,key)
            if result is not None:
                results.append(result)

        df = pd.DataFrame(results)
        if "key" in df.columns:
            df = df.sort_values("key").reset_index(drop=True)

        # 保存
        output_path = f"{folder}/output.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved results to {output_path}")

def NoiseAnalysis(config:dict, path:str):
    eta=input("eta:")
    eta=float(eta)
    for folder in tqdm.tqdm(glob.glob(f"{path}/CH*_noise")):
        noise_pathes=glob.glob(f"{folder}/rawdata/CH*.dat")
        model=np.array(0)*config["Readout"]["Sample"]
        for noise_path in noise_pathes:
            noise=general.LoadBin(noise_path)
            pulse=general.BesselCoeff(pulse,config["Readout"]["Rate"],config["Analysis"]["CutoffFrequency"])
            noise_fft = np.fft.fft(noise)
            noise_fft_amp=np.abs(noise_fft)
            model+=noise_fft_amp
        model/=len(noise_pathes)
        df=config["Readout"]["Rate"]/config["Readout"]["Sample"]
        fq=general.GetFreq(config["Readout"]["Rate"],config["Readout"]["Sample"])
        power=model**2/df
        amp_dens=np.sqrt(power)
        amp_dens = amp_dens[: int(config["Readout"]["Sample"] / 2) + 1] * eta * 1e+6
        np.savetxt(f"{folder}/modelnoise.txt", amp_dens)
        plt.plot(fq[: int(config["Readout"]["Sample"] / 2) + 1], amp_dens, linestyle="-", linewidth=0.7)
        plt.loglog()
        plt.xlabel("Frequency[Hz]")
        plt.ylabel("Intensity[pA/Hz$^{1/2}$]")
        plt.grid()
        plt.savefig(f"{folder}/modelnoise.png")
        plt.show()

def TempCalib(path:str,SelectedKeys,SavePath="output_tempcalib.csv",LoadPath="output_optimalfilter.csv"):
    for folder in tqdm.tqdm(glob.glob(f"{path}/CH*_pulse")):
        if os.path.exists(f"{folder}/{LoadPath}"):
            df=pd.read_csv(f"{folder}/{LoadPath}")
            df=df[df["key"].isin(SelectedKeys)].reset_index(drop=True)
            data_calib=general.TempCalib(df)
            data_calib.to_csv(f"{folder}/{SavePath}",index=False)
        else:
            print(f"output.csv not found in {folder}, skipping TempCalib.")

def OptimalFilter(config:dict,path:str,NoiseSPE, Channel:int, SelectedKeys, SavePath="output_optimalfilter.csv"):
    def lowpass(F,fq,cf):
        F[(fq>cf)] = 0
        return F
    
    rate=config["Readout"]["Rate"]
    samples=config["Readout"]["Sample"]
    cf=config["Analysis"]["CutoffFrequency"]

    eta=input("eta:")
    eta=float(eta)

    df=pd.read_csv(f"{path}/CH{Channel}_pulse/output.csv")

    AveragePulse=np.array(0)*samples

    for key in SelectedKeys:
        pulse=general.LoadBin(f"{path}/CH{Channel}_pulse/rawdata/CH_{key}.dat")
        AveragePulse+=pulse

    AveragePulse/=len(SelectedKeys)

    F = scipy.fftpack.fft.fft(AveragePulse)
    fq=general.GetFreq(rate,samples)
    time=general.GetTime(rate,samples)

    F2 = lowpass(F,fq,cf)
    filt = scipy.fftpack.fft.ifft(F2/NoiseSPE)
    filt = filt.real

    plt.plot(time,filt)
    plt.show()

    for key in SelectedKeys:
        pulse=general.LoadBin(f"{path}/CH{Channel}_pulse/rawdata/CH_{key}.dat")
        pulse-= df.at[key,"base"]
        pulse=general.Bessel(pulse,rate,cf)
        df.at[key,"height_opt"] = np.sum(pulse*filt)

    df.to_csv(f"{path}/CH{Channel}_pulse/{SavePath}",index=False)

def Scatter2D(path):
    folders = glob.glob(os.path.join(path, "CH*_pulse"))
    chs = [re.search(r'CH(.*)_pulse', os.path.basename(f)).group(1) for f in folders]

    ResultList=["Base","Peak","Rise","Decay"]

    XChannel=questionary.select("Select X Channel:",choices=chs).ask()
    XKey=questionary.select("Select X Key:",choices=ResultList).ask()

    YChannel=questionary.select("Select Y Channel:",choices=chs).ask()
    YKey=questionary.select("Select Y Key:",choices=ResultList).ask()

    dfX=pd.read_csv(f"{path}/CH{XChannel}_pulse/output.csv")
    dfY=pd.read_csv(f"{path}/CH{YChannel}_pulse/output.csv")

    dfX,dfY=general.KeyIsin(dfX,dfY)

    general.Scatter2D(dfX[XKey],dfY[YKey])


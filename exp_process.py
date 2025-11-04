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
    folders=glob.glob(f"{path}/CH*_pulse")

    for folder in folders:
        if os.path.exists(f"{folder}/output.csv"):
            Skip=questionary.confirm(f"output.csv already exists in {folder}. Do you want to skip Pulse Analysis for this folder?").ask()
            if Skip:
                print(f"Skipped Pulse Analysis for {folder}.")
                continue

        pulse_pathes = glob.glob(f"{folder}/rawdata/CH*.dat")
        pulse_pathes = natsort.natsorted(pulse_pathes)
        results = []

        for pulse_path in tqdm.tqdm(pulse_pathes):
            # ファイル名からキーを抽出
            filename = os.path.basename(pulse_path)
            key = os.path.splitext(filename)[0].split("_")[-1]
            
            pulse = general.LoadBin(pulse_path)
            if len(pulse) != config["Readout"]["Sample"]:
                continue

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

    dfs = {}
    for folder in folders:
        output_path = f"{folder}/output.csv"
        if os.path.exists(output_path):
            df = pd.read_csv(output_path)
            if "key" in df.columns:
                dfs[folder] = df

    all_keys = [set(df["key"]) for df in dfs.values()]
    common_keys = set.intersection(*all_keys) if all_keys else set()

    for folder, df in dfs.items():
        df = df[df["key"].isin(common_keys)].reset_index(drop=True)
        output_path = f"{folder}/output.csv"
        df.to_csv(output_path, index=False)

def NoiseAnalysis(config:dict, path:str):
    
    # 設定値の取得
    sample = config["Readout"]["Sample"] # データ点数 (N)
    rate = config["Readout"]["Rate"]     # サンプリングレート (Fs)
    cutoff = config["Analysis"]["CutoffFrequency"] # ローパスフィルタのカットオフ周波数

    eta=float(input("eta:"))
    
    # フォルダごとの処理
    for folder in glob.glob(f"{path}/CH*_noise"):
        noise_pathes = glob.glob(f"{folder}/rawdata/CH*.dat")
        
        amplitude_model = np.zeros(sample)
        count = 0
        original_noise_list = []
        
        # メディアンフィルタのカーネルサイズ (窓サイズ) を定義
        # 奇数に設定し、ノイズの幅に応じて調整します。3, 5, 7 などが一般的です。
        # ここでは例としてカーネルサイズ3を使用します。
        median_kernel_size = 3
        
        for noise_path in tqdm.tqdm(noise_pathes):
            noise = general.LoadBin(noise_path)
            if len(noise) != sample:
                continue
            
            # 1. スパイクノイズ除去（メディアンフィルタ）を追加
            # scipy.signal.medfilt を使用するには、事前に import scipy.signal が必要です。
            #noise = scipy.signal.medfilt(noise, kernel_size=median_kernel_size)
            
            # 2. 既存の処理を続行
            noise = general.Bessel(noise, rate, cutoff)
            original_noise_list.append(noise)

            noise_fft = np.fft.fft(noise)
            noise_amp = np.abs(noise_fft)
            amplitude_model += noise_amp
            count += 1

        amplitude_model/=count

        np.savetxt(f"{folder}/noise_fft_Amplitude.txt",amplitude_model)

        df=rate/sample
        power=amplitude_model**2 / df
        amp_dens=np.sqrt(power)
        amp_dens = amp_dens[: int(sample / 2) + 1] * eta * 1e+6

        np.savetxt(f"{folder}/modelnoise.txt",amp_dens)

        fq=general.GetFreq(rate,sample)

        # スペクトルをグラフ化
        plt.plot(fq[: int(sample / 2) + 1], amp_dens, linestyle="-", linewidth=0.7)
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
            print(f"output.csv not found in {folder}/{LoadPath}, skipping TempCalib.")

def OptimalFilter(config: dict, path: str, NoiseSPE, Channel: int, SelectedKeys, SavePath="output_optimalfilter.csv"):
    rate = config["Readout"]["Rate"]
    cf = config["Analysis"]["CutoffFrequency"]

    eta = float(input("eta:"))

    df = pd.read_csv(f"{path}/CH{Channel}_pulse/output.csv")

    # --- SelectedKeys のみを残す ---
    df = df[df["key"].isin(SelectedKeys)].reset_index(drop=True)

    AveragePulse = np.zeros(config["Readout"]["Sample"], dtype=float)
    count = 0

    for key in tqdm.tqdm(SelectedKeys):
        pulse = general.LoadBin(f"{path}/CH{Channel}_pulse/rawdata/CH{Channel}_{key}.dat")
        if len(pulse) != config["Readout"]["Sample"]:
            continue
        AveragePulse += pulse
        count += 1

    if count > 0:
        AveragePulse /= count
    else:
        print("[警告] 有効なパルスがありません。")
        return

    filt= general.OptimalFilterTemplate(NoiseSPE, AveragePulse, config)

    pulse_length = len(AveragePulse)
    # 各配列の「もとのインデックス軸」
    x_avg = np.linspace(0, 1, pulse_length)

    # filt の補間処理
    if len(filt) != pulse_length:
        x_filt = np.linspace(0, 1, len(filt))
        filt = np.interp(x_avg, x_filt, filt)

    # SelectedKeys に対応する行だけ処理
    for key in tqdm.tqdm(SelectedKeys):
        pulse = general.LoadBin(f"{path}/CH{Channel}_pulse/rawdata/CH{Channel}_{key}.dat").copy()
        if len(pulse) != config["Readout"]["Sample"]:
            continue
        base_values = df.loc[df["key"] == key, "Base"].values

        if len(base_values) > 0 and not pd.isna(base_values[0]):
            pulse -= base_values[0]
        else:
            print(f"[警告] key={key} に対応するBase値が見つかりません。スキップします。")
            continue

        pulse = general.Bessel(pulse, rate, cf)
        df.loc[df["key"] == key, "PeakOpt"] = np.sum(pulse * filt)

    df.to_csv(f"{path}/CH{Channel}_pulse/{SavePath}", index=False)

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

    general.Scatter2D(dfX[XKey],dfY[YKey],xlabel=f"CH{XChannel}_{XKey}",ylabel=f"CH{YChannel}_{YKey}")

def ViewPulse(path:str,Channel:int,Key:int):
    config=general.LoadJson(f"{path}/PulseConfig.json")
    pulse=general.LoadBin(f"{path}/CH{Channel}_pulse/rawdata/CH{Channel}_{Key}.dat")
    print(f"path:{path}/CH{Channel}_pulse/rawdata/CH{Channel}_{Key}.dat")
    print(f"sample:{len(pulse)}")
    result=general.AnalyzePulse(pulse,config,Key,plot=True)
    print(result)
    
def Hist(csvpath:str,Key:str,binNum=None):
    df=pd.read_csv(csvpath)
    data=df[Key]
    fwhm,reso=general.MakeHistgram(data,bin_num=binNum)
    plt.show()
    print(f"FWHM:{fwhm}, Reso:{reso}%")
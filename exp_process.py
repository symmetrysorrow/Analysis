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
import numpy.fft as np_fft

def PulseAnalysis(config: dict, path: str):
    folders=glob.glob(f"{path}/CH*_pulse")

    for folder in folders:
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
    """
    複数ノイズのFFT解析を行い、平均振幅スペクトルを求めて同じ性質のノイズを再生成・比較します。
    ノイズエネルギーの調整 (eta) は行いません。再生成ノイズは元のノイズの平均RMSにスケーリングされます。
    時間領域プロットではDC成分を除去し、ゼロ基準で比較します。
    """
    
    # 【削除】 ユーザー入力 eta の取得を削除
    
    # 設定値の取得
    sample = config["Readout"]["Sample"] # データ点数 (N)
    rate = config["Readout"]["Rate"]     # サンプリングレート (Fs)
    cutoff = config["Analysis"]["CutoffFrequency"] # ローパスフィルタのカットオフ周波数
    
    # 周波数軸（プロット用）
    frequencies = np_fft.fftfreq(sample, d=1.0/rate)
    
    
    # フォルダごとの処理
    for folder in tqdm.tqdm(glob.glob(f"{path}/CH*_noise")):
        noise_pathes = glob.glob(f"{folder}/rawdata/CH*.dat")
        
        # ----------------------------------------------------
        # 1. FFT解析とモデル（平均振幅スペクトル）の作成
        # ----------------------------------------------------
        
        amplitude_model = None  # 振幅スペクトルの平均（モデル）
        count = 0
        original_noise_list = []
        original_energies = []
        
        print(f"\nProcessing folder: {folder}")
        
        for noise_path in noise_pathes:
            noise = general.LoadBin(noise_path)
            if len(noise) != sample:
                continue
            
            # フィルタリング (元のノイズと同様の処理)
            #noise = general.Bessel(noise, rate, cutoff)
            original_noise_list.append(noise)
            original_energies.append(np.std(noise)) # RMS値 (標準偏差) をエネルギー指標として記録

            # FFTの実行
            fft_result = np_fft.fft(noise)
            
            # 振幅スペクトル（絶対値）
            amplitude_spectrum = np.abs(fft_result)
            
            # 平均スペクトルの更新
            if amplitude_model is None:
                amplitude_model = amplitude_spectrum
            else:
                amplitude_model += amplitude_spectrum
            
            count += 1
        
        if count == 0:
            print("No valid noise files found in the folder.")
            continue
            
        # 平均振幅スペクトルの計算
        amplitude_model /= count
        
        # 元のノイズの平均RMS
        mean_original_rms = np.mean(original_energies) if original_energies else 0.0


        # ----------------------------------------------------
        # 2. ノイズの再生成 (IFFTによる合成)
        # ----------------------------------------------------
        
        N = sample
        
        # 独立な位相を生成する部分の長さ
        if N % 2 == 0:
            independent_len = N // 2 + 1
        else:
            independent_len = (N + 1) // 2

        # 独立なランダム位相を生成 (0から2pi)
        random_phases = np.random.uniform(0, 2 * np.pi, independent_len)
        
        # 新しい複素数スペクトル Y_new の構築
        Y_new = np.zeros(N, dtype=complex)
        
        # 独立な成分の適用 (平均振幅スペクトルとランダム位相)
        Y_new[:independent_len] = amplitude_model[:independent_len] * np.exp(1j * random_phases)
        
        # 共役対称性の適用
        if N % 2 == 0:
            # 偶数点: Nyquist成分は実数で（位相0）。
            # IFFTの結果が実数になるように、DC成分 (Y_new[0]) とNyquist成分 (Y_new[N//2]) の位相は0にする
            # 既にY_new[N//2]にamplitude_model[N//2]が入っているので、位相は0
            Y_new[N//2] = amplitude_model[N//2] 
            # 負の周波数成分
            Y_new[independent_len:] = np.conjugate(Y_new[1:N//2][::-1])
        else:
            # 奇数点: 負の周波数成分
            Y_new[independent_len:] = np.conjugate(Y_new[1:independent_len][::-1])

        # 【追加】DC成分をゼロにする: Y_new[0]をゼロに設定
        # 振幅モデルは平均振幅スペクトルなので、Y_new[0]は元のノイズの平均DC成分ではない。
        # 再生成ノイズの平均をゼロにするために、Y_new[0] (DC成分) をゼロに設定する。
        Y_new[0] = 0.0 + 0.0j

        # 逆FFTの実行
        reconstructed_noise = np_fft.ifft(Y_new).real

        # ノイズエネルギーの調整 (元の平均RMSに合わせる)
        reconstructed_std = np.std(reconstructed_noise)
        if reconstructed_std > 1e-9 and mean_original_rms > 1e-9:
            scale_factor = mean_original_rms / reconstructed_std 
            reconstructed_noise *= scale_factor
        
        # ----------------------------------------------------
        # 3. 比較、保存、プロット (general.SaveBinとgeneral.MakeFigureは不使用)
        # ----------------------------------------------------
        
        output_dir = f"{folder}/reconstructed_noise"
        os.makedirs(output_dir, exist_ok=True)
        
        # ノイズデータの保存 (NumPyの標準機能を使用)
        np.save(f"{output_dir}/reconstructed_CH.npy", reconstructed_noise)
        
        # 再構成ノイズのエネルギーを記録
        reconstructed_energy = np.std(reconstructed_noise)

        # 比較結果の表示
        print(f"--- Comparison Results for {os.path.basename(folder)} ---")
        print(f"Original Noise (RMS Mean): {mean_original_rms:.6f}")
        print(f"Reconstructed Noise (RMS): {reconstructed_energy:.6f}")

        # プロット (元のノイズと再構成ノイズの比較)
        
        ## 時間領域比較
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"Time Domain Comparison (DC Removed): {os.path.basename(folder)}")
        
        if len(original_noise_list) > 0:
            # 比較用ノイズの例を取得
            original_example = original_noise_list[0]
            
            # 【追加/修正】プロット前に平均値を差し引いてゼロ基準にする
            original_dc_removed = original_example - np.mean(original_example)
            reconstructed_dc_removed = reconstructed_noise - np.mean(reconstructed_noise) # IFFTでY_new[0]=0としたので、理論上はmean=0だが念のため

            plot_len = min(sample, 500) 
            ax.plot(original_dc_removed[:plot_len], label="Original Noise (Example, DC Removed)", alpha=0.7)
            ax.plot(reconstructed_dc_removed[:plot_len], label="Reconstructed Noise (DC Removed)", alpha=0.7)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Amplitude (Zero Mean)")
        ax.legend()
        fig.savefig(f"{output_dir}/time_domain_comparison_dc_removed.png")
        plt.close(fig)
        
        ## 振幅スペクトルの比較
        # ... (振幅スペクトルプロットのコードは変更なし) ...
        fig, ax = plt.subplots(figsize=(10, 6)) 
        ax.set_title(f"Amplitude Spectrum Comparison: {os.path.basename(folder)}")
        
        reconstructed_amp = np.abs(np_fft.fft(reconstructed_noise))

        positive_indices = np.where(frequencies >= 0)
        
        ax.loglog(frequencies[positive_indices], amplitude_model[positive_indices], 
                            label="Amplitude Model (Mean)", linestyle='--', color='k', linewidth=2)
        ax.loglog(frequencies[positive_indices], reconstructed_amp[positive_indices], 
                            label="Reconstructed Noise", alpha=0.7)

        if len(original_noise_list) > 0:
            original_fft_example = np_fft.fft(original_noise_list[0])
            original_amp_example = np.abs(original_fft_example)
            ax.loglog(frequencies[positive_indices], original_amp_example[positive_indices], 
                            label="Original (Example)", alpha=0.5, linestyle=':')
        
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True, which="both", ls="--")
        fig.savefig(f"{output_dir}/amplitude_spectrum_comparison.png")
        plt.close(fig)

    print("\nNoise Analysis and Reconstruction Complete.")
            

def TempCalib(path:str,SelectedKeys,SavePath="output_tempcalib.csv",LoadPath="output.csv"):
    for folder in tqdm.tqdm(glob.glob(f"{path}/CH*_pulse")):
        if os.path.exists(f"{folder}/{LoadPath}"):
            df=pd.read_csv(f"{folder}/{LoadPath}")
            df=df[df["key"].isin(SelectedKeys)].reset_index(drop=True)
            data_calib=general.TempCalib(df)
            data_calib.to_csv(f"{folder}/{SavePath}",index=False)
        else:
            print(f"output.csv not found in {folder}/{LoadPath}, skipping TempCalib.")

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

    AveragePulse=np.zeros(config["Readout"]["Sample"], dtype=float)

    count=0

    for key in SelectedKeys:
        pulse=general.LoadBin(f"{path}/CH{Channel}_pulse/rawdata/CH{Channel}_{key}.dat")
        if len(pulse)!=config["Readout"]["Sample"]:
            continue
        AveragePulse+=pulse
        count+=1

    AveragePulse/=count

    F = fft.fft(AveragePulse)
    fq=general.GetFreq(rate,samples)
    time=general.GetTime(rate,samples)

    F2 = lowpass(F,fq,cf)
    filt = fft.ifft(F2/NoiseSPE)
    filt = filt.real

    plt.plot(time,filt)
    plt.show()

    for key in SelectedKeys:
        pulse=general.LoadBin(f"{path}/CH{Channel}_pulse/rawdata/CH_{key}.dat")
        pulse-= df.at[key,"base"]
        pulse=general.Bessel(pulse,rate,cf)
        df.at[key,"PeakOpt"] = np.sum(pulse*filt)

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

def GenerateNoiseFromModel(model, rate, sample, seed=None):
    """
    model: 片側RMS振幅スペクトル (NoiseAnalysisで√2補正・平均化されたもの)
    rate: サンプリングレート [Hz]
    sample: サンプル数
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 片側周波数軸 (使用しないが、配列長確認のため)
    # freq = np.fft.rfftfreq(sample, 1/rate)
    
    # ランダム位相（DCとナイキストを除いてランダムに）
    # len(freq) = len(model)
    phase = np.random.uniform(0, 2*np.pi, len(model))
    phase[0] = 0  # DC成分は実数
    if sample % 2 == 0:
        phase[-1] = 0  # ナイキストも実数

    # --- スケーリングの修正 ---
    # model (RMS振幅スペクトル) から irfft が期待する非正規化rfftの絶対値に戻す

    # 1. √2補正を戻す: np.sqrt(sample)で正規化された振幅に戻す
    amp_normalized = np.array(model).copy()
    
    # AC成分 (DCとNyquistを除く) の √2 補正を戻す (÷√2)
    if sample % 2 == 0:
        amp_normalized[1:-1] /= np.sqrt(2)
    else:
        amp_normalized[1:] /= np.sqrt(2)
        
    # 2. np.sqrt(sample)正規化を戻す: 非正規化rfftの絶対値 X_half_amp = |rfft(noise)| に戻す (×√sample)
    # Parsevalの定理と rfft の正規化を考慮
    X_half_amp = amp_normalized * np.sqrt(sample)
    
    # 複素スペクトル
    X_half = X_half_amp * np.exp(1j * phase)

    # ifft (rfftの逆変換。irfftは自動で1/Nスケーリングを行うため、X_halfは非正規化rfftの結果である必要がある)
    noise_reconstructed = np.fft.irfft(X_half, n=sample)
    
    return noise_reconstructed

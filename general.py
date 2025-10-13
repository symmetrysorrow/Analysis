import numpy as np
import pandas as pd
import scipy
import json
import matplotlib.pyplot as plt
import tqdm

def LoadTxt(file_path:str):
    try:
        data = np.loadtxt(file_path)
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None
    
def LoadBin(file_path:str):
    try:
        with open(file_path, "rb") as fb:
            fb.seek(4)
            data = np.frombuffer(fb.read(), dtype="float64")
        return data
    except Exception as e:
        print(f"Error loading binary file {file_path}: {e}")
        return None
    
def LoadJson(file_path:str):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return None
    
def Bessel(data,rate:float,fs:float):
    ws=GetWs(rate,fs)
    b,a=scipy.signal.bessel(2,ws,"low")
    filtered_data=scipy.signal.filtfilt(b,a,data)
    return filtered_data

def gaussian(x, amp, mean, stddev):
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def OptimalBinCount(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1  # 四分位範囲
    bin_width = 2 * iqr / (len(data) ** (1/3))  # ビン幅
    bin_count = int(np.ceil((np.max(data) - np.min(data)) / bin_width))  # ビン数
    return max(bin_count, 1)  # ビン数が1未満にならないようにする

def MakeHistgram(data,label=None,HistColor=None):
    bin_num = OptimalBinCount(data)
    #bin_num=30
    hist, bin_edges = np.histogram(data, bins=bin_num, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # ビンの中心を計算
    initial_guess = [np.max(hist), np.mean(data), np.std(data)]
    if HistColor is not None:
        plt.hist(data, bins=bin_num, density=False, label=label,color=HistColor)
    else:
        plt.hist(data, bins=bin_num, density=False, label=label)
    # ガウスフィッティング
    popt, pcov = scipy.optimize.curve_fit(gaussian, bin_centers, hist, p0=initial_guess, maxfev=10000000)
    amp_fit, mean_fit, stddev_fit = popt
    fwhm = 2 * stddev_fit * np.sqrt(2 * np.log(2))

    #ヒストグラム
    x_fit = np.linspace(bin_edges[0], bin_edges[-1], 1000)  # フィッティング用のx
    plt.plot(x_fit, gaussian(x_fit, *popt),color="red",alpha=0.5)  # フィッティング曲線

    return fwhm,fwhm/mean_fit

def InputPath():
    path=input("Input file path:")
    return path

def GetWs(rate:float,fs:float):
    ws=fs/rate*2
    return ws

def GetFreq(rate:float,samples:int):
    fq = np.arange(0, rate, rate / samples)
    return fq

def GetTime(rate:float,samples:int):
    time = np.arange(0, samples) / rate
    return time

def AnalyzePulse(pulse,Json:dict,key):
    try:
        pulse = pulse.astype(float)

        pulse=Bessel(pulse, Json["Readout"]["Rate"], Json["Analysis"]["CutoffFrequency"])

        base = np.mean(pulse[0:Json["Readout"]["PreSample"]])
        pulse -= base

        peak_index=np.argmax(pulse)
        peak_av=np.mean(pulse[peak_index - Json["Analysis"]["PeakAveragePreSample"] : peak_index + Json["Analysis"]["PeakAveragePostSample"]])

        for i in reversed(range(0, peak_index)):
            if pulse[i] <= peak_av * Json["Analysis"]["RiseHighRatio"]:
                rise_high = i
                break
        try:
            rise_high+=0
        except:
            rise_high=0
        for j in reversed(range(0, rise_high)):
            if pulse[j] <= peak_av * Json["Analysis"]["RiseLowRatio"]:
                rise_low = j
                break
        try:
            rise_low+=0
        except:
            rise_low=0

        rise=(rise_high - rise_low)/Json["Readout"]["Rate"]

        for i in range(peak_index, len(pulse)):
            if pulse[i] <= peak_av * Json["Analysis"]["DecayHighRatio"]:
                decay_high = i
                break
        try:
            decay_high+=0

        except:
            decay_high=0

        for j in range(decay_high, len(pulse)):
            if pulse[j] <= peak_av * Json["Analysis"]["DecayLowRatio"]:
                decay_low = j
                break
        try:
            decay_low+=0
        except:
            decay_low=0

        decay=(decay_low - decay_high)/Json["Readout"]["Rate"]

        result = {
            "key": int(key),             # key が数字なら int
            "Base": float(base),
            "Peak": float(peak_av),
            "Rise": float(rise),
            "Decay": float(decay),
        }


        return result
    except Exception as e:
        print(f"Error in AnalyzePulse: {e}")
        return None

def TempCalib(data):
    def func(X, *params):
        Y = np.zeros_like(X)
        for i, param in enumerate(params):
            Y = Y + np.array(param * X ** i)
        return Y
    
    def Calibration(x,params):
        array = np.zeros(len(params))
        for i,param in enumerate(params):
            term = param * x ** i
            array[i] = term
            sum = np.sum(array)
        return sum
    
    bases=data["Base"]
    heights_opt=data["PeakOpt"]

    p0=[0.01,0.01,0.01,0.01,0.01,0.01]

    popt,_pcov = scipy.optimize.curve_fit(func,bases,heights_opt,p0)
    x_fit = np.linspace(np.min(bases),np.max(bases),100000)
    fitted = func(x_fit,*tuple(popt))

    plt.plot(bases,heights_opt,'o',color='blue',markersize=3,label='a')
    plt.plot(x_fit,fitted,color='red',linewidth=1.0,linestyle='-')
    plt.xlabel('baseline [V]',fontsize = 16)
    plt.ylabel('pulseheight [V]',fontsize = 16)
    plt.grid()
    plt.show()
    plt.cla()

    st=np.mean(heights_opt)

    for index,row in tqdm.tqdm(data.iterrows()):
        data.at[index,"PeakOptTemp"] = row['Peak']/Calibration(row['base'],popt)*st

    plt.plot(bases,data["PeakOptTemp"],'o',color='tab:blue',markersize=0.7,label='a')
    plt.xlabel('baseline [V]',fontsize = 16)
    plt.ylabel('pulseheight [V]',fontsize = 16)
    plt.grid()
    plt.show()
    plt.cla()

    return data

def GetSelectedIndex(x, y):
    def inpolygon(sx, sy, x, y):
        inside = False
        for i1 in range(len(x)):
            i2 = (i1 + 1) % len(x)
            if min(x[i1], x[i2]) < sx < max(x[i1], x[i2]):
                if (y[i1] + (y[i2] - y[i1]) / (x[i2] - x[i1]) * (sx - x[i1]) - sy) > 0:
                    inside = not inside
        return inside
    
    plt.plot(x, y, "bo", markersize=1)
    plt.grid()
    picked = plt.ginput(n=-1, timeout=-1)
    plt.show()
    plt.cla()

    picked = np.array(picked)
    inside = np.ndarray(len(x), dtype=bool)
    inside[:] = False
    for i, (sx, sy) in enumerate(zip(x, y)):
        inside[i] = inpolygon(sx, sy, picked[:, 0], picked[:, 1])

    selected_index = np.where(inside)[0]
    return selected_index

def SelectIDFrom2DF(dfX,dfY,key:str):
    selected_index = GetSelectedIndex(dfX[key], dfY[key])
    selected_ids = dfX.iloc[selected_index]["key"].values
    return selected_ids

def SelectIDFrom1DF(df,keyX:str,keyY:str):
    selected_index = GetSelectedIndex(df[keyX], df[keyY])
    selected_ids = df.iloc[selected_index]["key"].values
    return selected_ids

def Scatter2D(x,y,xlabel=None,ylabel=None,title=None):
    plt.plot(x, y, "bo", markersize=1)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.grid()
    plt.show()
    plt.cla()

def KeyIsin(df1,df2):
    common_keys = set(df1["key"]) & set(df2["key"])

    df1=df1[df1["key"].isin(common_keys)].reset_index(drop=True)
    df2=df2[df2["key"].isin(common_keys)].reset_index(drop=True)
    return df1,df2
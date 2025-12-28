import general
import os
import shutil
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import questionary
import exp_process
import re
import sys

def main(path):
    if not os.path.exists(f"{path}/PulseConfig.json"):
        shutil.copy("./PulseConfig.json",f"{path}/PulseConfig.json")
        print("PulseConfig.json is Copied.\nPlease set the config file and run again.")
        sys.exit(0)

    config=general.LoadJson(f"{path}/PulseConfig.json")

    command=["Pulse Analysis","Noise Analysis","Temp and Optimal","Scatter2D","Select from Scatter","ViewPulse", "Hist","Exit"]

    choice=questionary.select("Select analysis type:",choices=command).ask()
    if choice=="Pulse Analysis":
        exp_process.PulseAnalysis(config,path)
    elif choice=="Noise Analysis":
        exp_process.NoiseAnalysis(config,path)
    elif choice=="Temp and Optimal":
        folders = glob.glob(os.path.join(path, "CH*_pulse"))
        chs = [re.search(r'CH(.*)_pulse', os.path.basename(f)).group(1) for f in folders]
        # --- 1ch or 2ch を選択 ---
        mode = questionary.select(
            "Select mode:",
            choices=["Single Channel", "Two Channels"]
        ).ask()

        if mode == "Single Channel":
            # ---- 1CH選択 ----
            Channel = questionary.select("Select Channel:", choices=chs).ask()
            df = pd.read_csv(f"{path}/CH{Channel}_pulse/output.csv")
            XKey=questionary.select("Select X Key:",choices=["Peak","Base","Rise","Decay"]).ask()
            YKey=questionary.select("Select Y Key:",choices=["Peak","Base","Rise","Decay"]).ask()
            SelectedKeys = general.SelectIDFrom1DF(df, XKey, YKey)
            np.savetxt(f"{path}/SelectedKeys.txt", SelectedKeys, fmt="%d")

            print(f"selected key:{SelectedKeys}")

            NoiseSPE=general.LoadTxt(f"{path}/CH{Channel}_noise/modelnoise.txt")

            exp_process.OptimalFilter(config,path,NoiseSPE,Channel, SelectedKeys)
            exp_process.TempCalib(path, SelectedKeys)

        elif mode == "Two Channels":
            if len(chs)<2:
                print("2チャンネル以上のデータが必要です。")
                return
            elif len(chs)==2:
                Channels=chs
            else:
            # ---- 2CH選択 ----
                Channels = questionary.checkbox("Select TWO Channels:", choices=chs).ask()
                if len(Channels) != 2:
                    print("2チャンネルを選択してください。")
                    return

            # 2つのchのcsvを読み込み
            df1 = pd.read_csv(f"{path}/CH{Channels[0]}_pulse/output.csv")
            df2 = pd.read_csv(f"{path}/CH{Channels[1]}_pulse/output.csv")

            Key=questionary.select("Select Key:",choices=["Peak","Base","Rise","Decay"]).ask()

            SelectedKeys = general.SelectIDFrom2DF(df1,df2,Key)
            np.savetxt(f"{path}/SelectedKeys.txt", SelectedKeys, fmt="%d")

            for ch in Channels:
                noise=general.LoadTxt(f"{path}/CH{ch}_noise/modelnoise.txt")
                exp_process.OptimalFilter(config,path,noise,ch, SelectedKeys)
                exp_process.TempCalib(path, SelectedKeys)

    elif choice=="Scatter2D":
        exp_process.Scatter2D(path)

    elif choice=="Select from Scatter":
        folders = glob.glob(os.path.join(path, "CH*_pulse"))
        chs = [re.search(r'CH(.*)_pulse', os.path.basename(f)).group(1) for f in folders]
        # --- 1ch or 2ch を選択 ---
        mode = questionary.select(
            "Select mode:",
            choices=["Single Channel", "Two Channels"]
        ).ask()

        if mode == "Single Channel":
            # ---- 1CH選択 ----
            Channel = questionary.select("Select Channel:", choices=chs).ask()
            df = pd.read_csv(f"{path}/CH{Channel}_pulse/output.csv")
            XKey=questionary.select("Select X Key:",choices=["Peak","Base","Rise","Decay"]).ask()
            YKey=questionary.select("Select Y Key:",choices=["Peak","Base","Rise","Decay"]).ask()
            SelectedKeys = general.SelectIDFrom1DF(df, XKey, YKey)
            np.savetxt(f"{path}/SelectedKeys.txt", SelectedKeys, fmt="%d")           

        elif mode == "Two Channels":
            if len(chs)<2:
                print("2チャンネル以上のデータが必要です。")
                return
            elif len(chs)==2:
                Channels=chs
            else:
            # ---- 2CH選択 ----
                Channels = questionary.checkbox("Select TWO Channels:", choices=chs).ask()
                if len(Channels) != 2:
                    print("2チャンネルを選択してください。")
                    return

            # 2つのchのcsvを読み込み
            df1 = pd.read_csv(f"{path}/CH{Channels[0]}_pulse/output.csv")
            df2 = pd.read_csv(f"{path}/CH{Channels[1]}_pulse/output.csv")

            Key=questionary.select("Select Key:",choices=["Peak","Base","Rise","Decay"]).ask()

            SelectedKeys = general.SelectIDFrom2DF(df1,df2,Key)
            np.savetxt(f"{path}/SelectedKeys.txt", SelectedKeys, fmt="%d")

        print(f"Saved selected keys to {path}/SelectedKeys.txt")

    elif choice=="ViewPulse":
        folders = glob.glob(os.path.join(path, "CH*_pulse"))
        chs = [re.search(r'CH(.*)_pulse', os.path.basename(f)).group(1) for f in folders]
        # ---- チャンネル選択 ----
        Channel = questionary.select("Select Channel:", choices=chs).ask()
        Key = questionary.text("Input Key (integer):").ask()
        try:
            Key = int(Key)
        except ValueError:
            print("Keyは整数で入力してください。")
            return
        exp_process.ViewPulse(path,Channel,Key)

    elif choice=="Hist":
        folders = glob.glob(os.path.join(path, "CH*_pulse"))
        chs = [re.search(r'CH(.*)_pulse', os.path.basename(f)).group(1) for f in folders]
        Channel = questionary.select("Select Channel:", choices=chs).ask()

        csvs=glob.glob(os.path.join(path, f"CH{Channel}_pulse","*.csv"))
        files=[os.path.basename(f) for f in csvs]
        file=questionary.select("Select CSV File:", choices=files).ask()

        path=f"{path}/CH{Channel}_pulse/{file}"
        csv=pd.read_csv(path)
        Keys=list(csv.columns)
        Key = questionary.select("Select Key:", choices=Keys).ask()

        binChoose=questionary.select("Choose bin number option:",choices=["Auto","Manual"]).ask()
        if binChoose=="Manual":
            binNum=questionary.text("Input bin number (integer):").ask()
            try:
                binNum = int(binNum)
            except ValueError:
                print("Bin numberは整数で入力してください。")
                return
            exp_process.Hist(path,Key,binNum=binNum)
        else:
            binNum=None
            exp_process.Hist(path,Key)
    
    elif choice=="Exit":
        sys.exit(0)

path=general.InputPath()

while True:
    main(path)
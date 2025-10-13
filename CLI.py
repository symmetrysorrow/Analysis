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

    config=general.LoadJson(f"{path}/PulseConfig.json")

    command=["Pulse Analysis","Noise Analysis","Temp and Optimal","Scatter2D", "Exit"]

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
            YKey=questionary.select("Select X Key:",choices=["Peak","Base","Rise","Decay"]).ask()
            SelectedKeys = general.SelectIDFrom1DF(df, XKey, YKey)

            noise=general.LoadTxt(f"{path}/CH{Channel}_noise/modelnoise.txt")

            exp_process.OptimalFilter(config,path,noise,Channel, SelectedKeys)
            exp_process.TempCalib(path, SelectedKeys)

        elif mode == "Two Channels":
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

            for ch in Channels:
                noise=general.LoadTxt(f"{path}/CH{ch}_noise/modelnoise.txt")
                exp_process.OptimalFilter(config,path,noise,ch, SelectedKeys)
                exp_process.TempCalib(path, SelectedKeys)

    elif choice=="Scatter2D":
        exp_process.Scatter2D(path)
    
    elif choice=="Exit":
        sys.exit(0)

path=general.InputPath()

while True:
    main(path)
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

def main():
    path=general.InputPath()

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
        Channel=questionary.select("Select Channel:",choices=chs).ask()
        df=pd.read_csv(f"{path}/CH{Channel}_pulse/output.csv")
        SelectedKeys=general.SelectIDFrom1DF(df,"Peak","Rise")
        exp_process.TempCalib(path,SelectedKeys)
        exp_process.OptimalFilter(path,SelectedKeys)

    elif choice=="Scatter2D":
        exp_process.Scatter2D(path)
    
    elif choice=="Exit":
        sys.exit(0)

while True:
    main()




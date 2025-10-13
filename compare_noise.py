import general as gn
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

for i in [1,0]:
    path=input("path:")
    rate=int(input("rate:"))
    noise_pathes=glob(f"{path}/CH0_noise/rawdata/CH0_*.dat")
    noise_path=noise_pathes[0]
    noise=gn.LoadBin(noise_path).copy()
    noise-=np.mean(noise)
    samples=len(noise)
    time=gn.GetTime(rate,samples)
    plt.plot(time,noise,alpha=0.5,label=path)
plt.legend()
plt.show()
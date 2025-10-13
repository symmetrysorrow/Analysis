import numpy as np
import matplotlib.pyplot as plt

# サンプル信号の生成
fs = 1000  # サンプリング周波数 [Hz]
T = 1.0    # 信号長 [s]
N = int(fs * T)  # サンプル数
t = np.linspace(0, T, N, endpoint=False)

# テスト信号: 50Hzと120Hzの正弦波の和
f1, f2 = 50, 120
A1, A2 = 1.0, 0.5
signal = A1 * np.sin(2 * np.pi * f1 * t) + A2 * np.sin(2 * np.pi * f2 * t)

# 時間領域のエネルギー計算
energy_time = np.sum(signal**2)
print(f"時間領域エネルギー: {energy_time:.6f}")

# FFTの実行（3つの方法）

# 方法1: numpy.fft.fft（デフォルト）
fft_result = np.fft.fft(signal)
freq = np.fft.fftfreq(N, 1/fs)

# エネルギー保存のための正規化
# パーセバルの定理: Σ|x[n]|² = (1/N) * Σ|X[k]|²
energy_freq_raw = np.sum(np.abs(fft_result)**2) / N
print(f"周波数領域エネルギー (raw FFT): {energy_freq_raw:.6f}")

# 方法2: scipy.fft.fft with norm='forward'（推奨）
from scipy import fft as scipy_fft
fft_forward = scipy_fft.fft(signal, norm='forward')
energy_freq_forward = np.sum(np.abs(fft_forward)**2)
print(f"周波数領域エネルギー (norm='forward'): {energy_freq_forward:.6f}")

# 方法3: scipy.fft.fft with norm='ortho'（正規直交）
fft_ortho = scipy_fft.fft(signal, norm='ortho')
energy_freq_ortho = np.sum(np.abs(fft_ortho)**2)
print(f"周波数領域エネルギー (norm='ortho'): {energy_freq_ortho:.6f}")

# パワースペクトル密度（PSD）の計算
# 片側スペクトルの場合
positive_freq_idx = freq >= 0
psd = (np.abs(fft_result[positive_freq_idx])**2) / N
# DC成分とナイキスト周波数以外は2倍にする（両側→片側変換）
psd[1:-1] *= 2

# プロット
fig, axes = plt.subplots(3, 1, figsize=(10, 10))

# 時間領域信号
axes[0].plot(t[:200], signal[:200])
axes[0].set_xlabel('時間 [s]')
axes[0].set_ylabel('振幅')
axes[0].set_title('時間領域信号')
axes[0].grid(True)

# 振幅スペクトル
axes[1].plot(freq[positive_freq_idx], np.abs(fft_result[positive_freq_idx])/N)
axes[1].set_xlabel('周波数 [Hz]')
axes[1].set_ylabel('振幅')
axes[1].set_title('振幅スペクトル')
axes[1].set_xlim([0, 200])
axes[1].grid(True)

# パワースペクトル密度
axes[2].plot(freq[positive_freq_idx], psd)
axes[2].set_xlabel('周波数 [Hz]')
axes[2].set_ylabel('パワー')
axes[2].set_title('パワースペクトル密度（片側）')
axes[2].set_xlim([0, 200])
axes[2].grid(True)

plt.tight_layout()
plt.show()

# 検証: エネルギーの比較
print("\n=== エネルギー検証 ===")
print(f"誤差 (raw FFT):      {abs(energy_time - energy_freq_raw)/energy_time * 100:.2e} %")
print(f"誤差 (norm=forward): {abs(energy_time - energy_freq_forward)/energy_time * 100:.2e} %")
print(f"誤差 (norm=ortho):   {abs(energy_time - energy_freq_ortho)/energy_time * 100:.2e} %")
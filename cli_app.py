import glob
import os
import re
import shutil
import sys

import exp_process
import general
import numpy as np
import pandas as pd

import cli_prompts as prompts


def _ensure_config(path):
    config_path = os.path.join(path, "PulseConfig.json")
    if not os.path.exists(config_path):
        shutil.copy("./PulseConfig.json", config_path)
        print("PulseConfig.json is Copied.\nPlease set the config file and run again.")
        sys.exit(0)
    return general.LoadJson(config_path)


def _get_channels(path):
    folders = glob.glob(os.path.join(path, "CH*_pulse"))
    return [re.search(r"CH(.*)_pulse", os.path.basename(folder)).group(1) for folder in folders]


def _select_mode():
    return prompts.select_mode()


def _select_single_channel_keys(path, chs):
    channel = prompts.select_channel(chs)
    df = pd.read_csv(os.path.join(path, f"CH{channel}_pulse", "output.csv"))
    x_key = prompts.select_x_key()
    y_key = prompts.select_y_key()
    selected_keys = general.SelectIDFrom1DF(df, x_key, y_key)
    return channel, selected_keys


def _select_two_channel_keys(path, chs):
    if len(chs) < 2:
        print("2チャンネル以上のデータが必要です。")
        return None, None, None
    if len(chs) == 2:
        channels = chs
    else:
        channels = prompts.select_two_channels(chs)
        if not channels:
            return None, None, None
        if len(channels) != 2:
            print("2チャンネルを選択してください。")
            return None, None, None

    df1 = pd.read_csv(os.path.join(path, f"CH{channels[0]}_pulse", "output.csv"))
    df2 = pd.read_csv(os.path.join(path, f"CH{channels[1]}_pulse", "output.csv"))
    key = prompts.select_key()
    selected_keys = general.SelectIDFrom2DF(df1, df2, key)
    return channels, key, selected_keys


def _save_selected_keys(path, selected_keys):
    np.savetxt(os.path.join(path, "SelectedKeys.txt"), selected_keys, fmt="%d")
    print(f"Saved selected keys to {path}/SelectedKeys.txt")


def _run_temp_and_optimal(config, path):
    chs = _get_channels(path)
    mode = _select_mode()

    if mode == "Single Channel":
        channel, selected_keys = _select_single_channel_keys(path, chs)
        _save_selected_keys(path, selected_keys)
        print(f"selected key:{selected_keys}")

        noise_spe = general.LoadTxt(os.path.join(path, f"CH{channel}_noise", "modelnoise.txt"))
        exp_process.OptimalFilter(config, path, noise_spe, channel, selected_keys)
        exp_process.TempCalib(path, selected_keys)
        return

    if mode == "Two Channels":
        channels, _key, selected_keys = _select_two_channel_keys(path, chs)
        if channels is None:
            return

        _save_selected_keys(path, selected_keys)

        for ch in channels:
            noise = general.LoadTxt(os.path.join(path, f"CH{ch}_noise", "modelnoise.txt"))
            exp_process.OptimalFilter(config, path, noise, ch, selected_keys)
            exp_process.TempCalib(path, selected_keys)


def _run_select_from_scatter(path):
    chs = _get_channels(path)
    mode = _select_mode()

    if mode == "Single Channel":
        _channel, selected_keys = _select_single_channel_keys(path, chs)
        np.savetxt(os.path.join(path, "SelectedKeys.txt"), selected_keys, fmt="%d")
    elif mode == "Two Channels":
        channels, _key, selected_keys = _select_two_channel_keys(path, chs)
        if channels is None:
            return
        np.savetxt(os.path.join(path, "SelectedKeys.txt"), selected_keys, fmt="%d")

    print(f"Saved selected keys to {path}/SelectedKeys.txt")


def _run_view_pulse(path):
    chs = _get_channels(path)
    channel = prompts.select_channel(chs)
    key = prompts.input_integer("Input Key (integer):")
    try:
        key = int(key)
    except ValueError:
        print("Keyは整数で入力してください。")
        return

    exp_process.ViewPulse(path, channel, key)


def _run_hist(path):
    chs = _get_channels(path)
    channel = prompts.select_channel(chs)

    csvs = glob.glob(os.path.join(path, f"CH{channel}_pulse", "*.csv"))
    files = [os.path.basename(csv_path) for csv_path in csvs]
    file = prompts.select_csv_file(files)

    csv_path = os.path.join(path, f"CH{channel}_pulse", file)
    csv = pd.read_csv(csv_path)
    keys = list(csv.columns)
    key = prompts.select_csv_column(keys)

    bin_choose = prompts.select_bin_option()
    if bin_choose == "Manual":
        bin_num = prompts.input_integer("Input bin number (integer):")
        try:
            bin_num = int(bin_num)
        except ValueError:
            print("Bin numberは整数で入力してください。")
            return
        exp_process.Hist(csv_path, key, binNum=bin_num)
        return

    exp_process.Hist(csv_path, key)


def main(path):
    config = _ensure_config(path)

    choice = prompts.select_analysis_type()
    if choice == "Pulse Analysis":
        exp_process.PulseAnalysis(config, path)
    elif choice == "Noise Analysis":
        exp_process.NoiseAnalysis(config, path)
    elif choice == "Temp and Optimal":
        _run_temp_and_optimal(config, path)
    elif choice == "Scatter2D":
        exp_process.Scatter2D(path)
    elif choice == "Select from Scatter":
        _run_select_from_scatter(path)
    elif choice == "ViewPulse":
        _run_view_pulse(path)
    elif choice == "Hist":
        _run_hist(path)
    elif choice == "Exit":
        sys.exit(0)


def run():
    path = general.InputPath()
    while True:
        main(path)


if __name__ == "__main__":
    run()

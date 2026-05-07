import glob
import os
import re
import shutil
from contextlib import contextmanager

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
from scipy.optimize import curve_fit


R_SH = 3.9e-3


@contextmanager
def _cd(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _load_dat(path):
    try:
        return np.loadtxt(path, comments="#", skiprows=6)
    except Exception:
        return np.loadtxt(path, comments="#")


def _linear(x, a, b):
    return a * x + b


def _offset(data):
    data = np.asarray(data) - data[0]
    if np.mean(data[:10]) < 0:
        data = data * -1
    return data


def _list_child_dirs(path):
    return [entry.path for entry in os.scandir(path) if entry.is_dir()]


def _dirs_with_direct_dat(path):
    data_dirs = []
    for child in _list_child_dirs(path):
        if glob.glob(os.path.join(child, "*.dat")):
            data_dirs.append(child)
    return data_dirs


def _has_subdirs(path):
    return any(entry.is_dir() for entry in os.scandir(path))


def _has_pulse_layout(path):
    return any(
        entry.is_dir() and entry.name.endswith("_pulse")
        for entry in os.scandir(path)
    )


def _has_rt_layout(path):
    data_dirs = _dirs_with_direct_dat(path)
    if len(data_dirs) != 1:
        return False

    dats = glob.glob(os.path.join(data_dirs[0], "*.dat"))
    if not dats:
        return False
    sample = os.path.basename(dats[0])
    return re.search(r"^CH\d+_\d+mK_\d+uA\.dat$", sample) is not None


def _has_iv_layout(path):
    data_dirs = _dirs_with_direct_dat(path)
    if len(data_dirs) >= 2:
        return True
    if len(data_dirs) == 1 and _has_subdirs(data_dirs[0]):
        return True
    return False


def detect_dataset_kind(path):
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        return None
    if _has_pulse_layout(path):
        return "pulse"
    if _has_iv_layout(path):
        return "iv"
    if _has_rt_layout(path):
        return "rt"
    return None


def _extract_int(pattern, text):
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"Could not parse value from {text}")
    return int(match.group(1))


def run_iv(path):
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        print(f"Path not found: {path}")
        return

    with _cd(path):
        temps = natsorted(
            [
                name
                for name in os.listdir(".")
                if os.path.isdir(name) and glob.glob(os.path.join(name, "*.dat"))
            ]
        )

        if not temps:
            print("IV folder not found.")
            return

        os.makedirs("calibration", exist_ok=True)
        os.makedirs("rawdata", exist_ok=True)

        eta = None
        for t in temps:
            files = natsorted(glob.glob(os.path.join(t, "*.dat")))
            I_bias = []
            V_out = []
            for file_path in files:
                data = _load_dat(file_path)
                V_out.append(np.mean(data))
                I_bias.append(_extract_int(r"_(\d+)uA", os.path.basename(file_path)))

            V_out = np.array(V_out)
            I_bias = np.array(I_bias)
            V_out = _offset(V_out)

            if eta is None:
                fit_count = min(10, len(I_bias))
                popt, _cov = curve_fit(_linear, I_bias[:fit_count], V_out[:fit_count])
                eta = 1 / popt[0]
                print(f"eta (uA/V): {eta}")

            I_tes = eta * V_out
            I_sh = I_bias - I_tes
            V_tes = I_sh * R_SH
            R_tes = np.zeros_like(I_tes)
            if len(I_tes) > 1:
                R_tes[1:] = V_tes[1:] / I_tes[1:]

            np.savetxt(f"rawdata/IV_{t}.txt", [I_bias, V_out, R_tes])
            np.savetxt(f"calibration/IV_{t}.txt", [I_bias, V_out, R_tes])

            plt.figure()
            plt.plot(I_bias, V_out, marker="o", c="red", linewidth=1, markersize=6)
            plt.title(f"I-V at {t}")
            plt.xlabel("I_bias[uA]")
            plt.ylabel("V_out[V]")
            plt.grid(True)
            plt.savefig(f"rawdata/IV_{t}.png")
            plt.close()

            plt.figure()
            plt.plot(I_bias, R_tes, marker="o", c="red", linewidth=1, markersize=6)
            plt.title(f"I-R at {t}")
            plt.xlabel("I_bias[uA]")
            plt.ylabel("R_tes[$\\Omega$]")
            plt.grid(True)
            plt.savefig(f"rawdata/IR_{t}.png")
            plt.close()

        plt.figure()
        cnt = 0
        for t in temps:
            files = natsorted(glob.glob(os.path.join(t, "*.dat")))
            I_bias = []
            V_out = []
            for file_path in files:
                data = _load_dat(file_path)
                V_out.append(np.mean(data))
                I_bias.append(_extract_int(r"_(\d+)uA", os.path.basename(file_path)))

            V_out = np.array(V_out)
            I_bias = np.array(I_bias)
            V_out = _offset(V_out)
            plt.plot(
                I_bias,
                V_out,
                marker="o",
                linewidth=1,
                markersize=6,
                label=t,
                color=cm.hsv(float(cnt) / float(len(temps))),
            )
            cnt += 1

        plt.title("I-V")
        plt.xlabel("I_bias[uA]")
        plt.ylabel("V_out[V]")
        plt.grid(True)
        plt.legend()
        plt.savefig("rawdata/IV_matome.png")
        plt.show()
        plt.close()


def run_rt(path):
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        print(f"Path not found: {path}")
        return

    with _cd(path):
        if not os.path.exists("output"):
            os.mkdir("output")

        if not os.path.exists("rawdata"):
            files = natsorted(glob.glob("*.dat"))
            os.mkdir("rawdata")
            for file_path in files:
                shutil.move(file_path, "rawdata")

        files = natsorted(glob.glob(os.path.join("rawdata", "*.dat")))
        if not files:
            print("RT data not found.")
            return

        channel_match = re.search(r"CH(\d+)_", os.path.basename(files[0]))
        channel = channel_match.group(1) if channel_match else "1"

        I_bias = []
        V_out = []
        T = []
        for file_path in files:
            data = _load_dat(file_path)
            name = os.path.splitext(os.path.basename(file_path))[0]
            V_out.append(np.mean(data))
            T.append(_extract_int(r"_(\d+)mK", name))
            I_bias.append(float(_extract_int(r"_(\d+)uA", name)))

        low_temp = T.count(min(T))
        popt, _cov = curve_fit(_linear, I_bias[:low_temp], V_out[:low_temp])
        eta = 1 / popt[0]
        print(f"eta (uA/V): {eta}")

        T_unique = sorted(set(T[low_temp:]))
        I_bias_unique = sorted(set(I_bias[low_temp:]))

        V_out_by_current = []
        for current in I_bias_unique:
            values = []
            for temp in T_unique:
                file_path = f"rawdata/CH{channel}_{temp}mK_{int(current)}uA.dat"
                data = _load_dat(file_path)
                values.append(np.mean(data))
            V_out_by_current.append(values)

        V_out_by_current = np.array(V_out_by_current)

        plt.figure()
        for idx, values in enumerate(V_out_by_current):
            if idx > 0:
                V_out_base = values - V_out_by_current[0]
                R = R_SH * (I_bias_unique[idx] / (eta * V_out_base) - 1)
                plt.title("R-T")
                plt.plot(
                    T_unique,
                    R,
                    marker="o",
                    linewidth=1,
                    label=f"{I_bias_unique[idx]}uA",
                    markersize=4,
                )
                plt.xlabel("Temperature[mK]")
                plt.ylabel("Resistance[m$\\Omega$]")
                plt.grid(True)
                plt.legend(loc="best", fancybox=True, shadow=True)
                np.savetxt(f"output/{int(I_bias_unique[idx])}uA.txt", [T_unique, R])

        plt.savefig("output/RT.png")
        plt.show()
        plt.close()


def dispatch(path, pulse_runner):
    kind = detect_dataset_kind(path)
    if kind == "pulse":
        pulse_runner(path)
        return kind
    if kind == "iv":
        run_iv(path)
        return kind
    if kind == "rt":
        run_rt(path)
        return kind

    print("Could not detect the analysis target from the path.")
    print("Please choose a typical pulse, IV, or RT folder.")
    return None

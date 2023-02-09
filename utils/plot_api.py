from utils.auxfunctions import superimposed, moving_window, fourier
from utils.preprocess import windows_creator
from utils.detection import detection_iter
import numpy as np
from itertools import repeat
import matplotlib.pyplot as plt


def img_signal(request_information, no_return=False):
    signal_name = request_information["signal_name"]
    signals = request_information["signals"]
    signal, t, params = signals.load_data(signal_name)
    request_information["signal"] = signal
    request_information["t"] = t
    request_information["params"] = params
    request_information["plots"]["img_signal"] = True
    if no_return:
        print("no return img_signal")
        return

    return t.tolist(), signal.tolist(), "linear", "img"


def img_si_signal(request_information, no_return=False):
    signal = request_information.get("signal", "")

    if len(signal) == 0:

        img_signal(request_information, no_return=True)

    signal = request_information["signal"]
    fs = request_information["params"]["fs"]
    t = request_information["t"]

    si_signal = superimposed(signal, fs)

    request_information["si_signal"] = si_signal
    request_information["plots"]["img_si_signal"] = True
    if no_return:
        print("no return img_si_signal")
        return

    return t.tolist(), si_signal.tolist(), "linear", "img"


def anim_signal(request_information, no_return=False):
    signal = request_information.get("signal", "")

    if len(signal) == 0:
        img_signal(request_information, no_return=True)

    signal = request_information["signal"]
    t = request_information["t"]
    window_length = request_information["window_length"]
    step = request_information["step"]
    signal_windows, t_windows = list(
        map(moving_window, [signal, t], repeat(window_length), repeat(step))
    )

    request_information["signal_windows"] = signal_windows
    request_information["t_windows"] = t_windows
    request_information["plots"]["anim_signal"] = True
    max_in_windows = signal_windows.max()
    min_in_windows = signal_windows.min()
    max_min = [max_in_windows, min_in_windows]
    if no_return:
        print("no return anim_signal")
        return

    return t_windows.tolist(), signal_windows.tolist(), max_min, "anim"


def anim_si_signal(request_information, no_return=False):
    si_signal = request_information.get("si_signal", "")
    if len(si_signal) == 0:
        print("No existe la señal")
        img_si_signal(request_information, no_return=True)

    si_signal = request_information["si_signal"]
    t = request_information["t"]
    window_length = request_information["window_length"]
    step = request_information["step"]
    si_signal_windows, t_windows = list(
        map(moving_window, [si_signal, t], repeat(window_length), repeat(step))
    )

    request_information["si_signal_windows"] = si_signal_windows
    request_information["t_windows"] = t_windows
    request_information["plots"]["anim_si_signal"] = True
    max_in_windows = si_signal_windows.max()
    min_in_windows = si_signal_windows.min()
    max_min = [max_in_windows, min_in_windows]

    if no_return:
        print("no return anim_si_signal")
        return

    return t_windows.tolist(), si_signal_windows.tolist(), max_min, "anim"


def anim_fft(request_information, no_return=False):
    signal_windows = request_information.get("signal_windows", "")
    if len(signal_windows) == 0:
        print("No existe la señal")
        anim_signal(request_information, no_return=True)

    signal_windows = request_information["signal_windows"]
    dt = request_information["params"]["dt"]
    window_length = request_information["window_length"]

    xf, fft_windows = fourier(signal_windows, window_length, dt)

    request_information["xf"] = xf
    request_information["fft_windows"] = fft_windows
    request_information["fft_windows_fundamental"] = fft_windows[:, 1]
    request_information["plots"]["anim_fft"] = True
    max_in_windows = fft_windows.max()
    min_in_windows = fft_windows.min()
    max_min = [max_in_windows, min_in_windows]

    if no_return:
        print("no return anim_fft")
        return

    number_of_windows = len(fft_windows)

    return (
        np.tile(xf, (number_of_windows, 1)).tolist(),
        fft_windows.tolist(),
        max_min,
        "anim",
    )


def anim_si_fft(request_information, no_return=False):
    si_signal_windows = request_information.get("si_signal_windows", "")
    if len(si_signal_windows) == 0:
        print("No existe la señal")
        anim_si_signal(request_information, no_return=True)

    si_signal_windows = request_information["si_signal_windows"]
    dt = request_information["params"]["dt"]
    window_length = request_information["window_length"]

    xf, si_fft_windows = fourier(si_signal_windows, window_length, dt)

    request_information["xf"] = xf
    request_information["si_fft_windows"] = si_fft_windows
    request_information["si_fft_windows_fundamental"] = si_fft_windows[:, 1]
    request_information["plots"]["anim_si_fft"] = True
    max_in_windows = si_fft_windows.max()
    min_in_windows = si_fft_windows.min()
    max_min = [max_in_windows, min_in_windows]

    if no_return:
        print("no return anim_fft")
        return

    return xf, si_fft_windows, max_min, "anim"


def anim_trip(request_information, no_return=False):
    fft_windows = request_information.get("fft_windows", "")
    if len(fft_windows) == 0:
        print("No existe la señal")
        anim_fft(request_information, no_return=True)

    fft_windows = request_information["fft_windows"]
    fundamental = request_information["fft_windows_fundamental"]
    trip_windows = detection_iter(fft_windows, fundamental)

    request_information["trip_windows"] = trip_windows
    request_information["plots"]["anim_trip"] = True
    max_in_windows = max(trip_windows)
    min_in_windows = min(trip_windows)
    max_min = [max_in_windows, min_in_windows]

    if no_return:
        print("no return anim_trip")
        return

    return trip_windows, max_min, "anim"


def anim_si_trip(request_information, no_return=False):
    si_fft_windows = request_information.get("si_fft_windows", "")
    fundamental = request_information.get("fft_windows_fundamental", "")
    if len(si_fft_windows) == 0:
        print("No existe la señal")
        anim_si_fft(request_information, no_return=True)
    if len(fundamental) == 0:
        print("No fundamental")
        anim_fft(request_information, no_return=True)

    si_fft_windows = request_information["si_fft_windows"]
    fundamental = request_information["fft_windows_fundamental"]
    si_trip_windows = detection_iter(si_fft_windows, fundamental)

    request_information["si_trip_windows"] = si_trip_windows
    request_information["plots"]["anim_si_trip"] = True

    if no_return:
        print("no return anim_trip")
        return

    return si_trip_windows, "", "anim"


# def img_trip(request_information, no_return=False):
#     signals = request_information["signals"]
#     signal_name = request_information["signal_name"]
#     (signal_window, signal_si_window, t_window), (
#         signal_fft,
#         signal_si_fft,
#         xf,
#     ) = windows_creator(
#         64,
#         signals=signals,
#         signal_name=signal_name,
#         windows_fourier=True,
#     )
#     signal_fundamental = signal_fft[:, 1]
#     si_fundamental = signal_si_fft[:, 1]
#     trip = detection_iter(signal_fft, signal_fundamental)
#     t_window = np.insert(t_window[:, -1], 0, 0)
#     trip = np.insert(trip, 0, 0)
#     return t_window.tolist(), trip.tolist(), "hv", "img"


def img_trip(request_information, no_return=False):
    trip_windows = request_information.get("trip_windows", "")
    if len(trip_windows) == 0:
        print("No existe la señal")
        anim_trip(request_information, no_return=True)

    trip_windows = request_information["trip_windows"]
    t_windows = request_information["t_windows"]
    t_window = np.insert(t_windows[:, -1], 0, 0)
    trip_windows = np.insert(trip_windows, 0, 0)
    return t_window.tolist(), trip_windows.tolist(), "hv", "img"

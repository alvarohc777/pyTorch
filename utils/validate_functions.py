# Archivo basado en detector_form_1.py

from scipy.signal.windows import get_window as wndw
from scipy.fft import rfftfreq
import numpy as np

# import concurrent.futures
# from itertools import repeat

if __name__ == "__main__":
    from signalload import CSV, CSV_prueba, CSV_PATH
    from auxfunctions import superimposed, synthetic_signal, moving_window, fourier
else:
    from .signalload import CSV, CSV_prueba, CSV_PATH
    from .auxfunctions import superimposed, synthetic_signal, moving_window, fourier


relay_list = ["R01", "R10", "R15"]
signal_list = ["Ia", "Ib", "Ic"]


def validate_csv(PATH, signal_name):
    detection = 0
    signal = CSV_PATH(PATH)

    # Parámetros de la señal
    dt = signal.dt
    t = signal.t
    fs = signal.fs
    signal = eval(f"signal.{signal_name}")

    # signal = signal.R1I1
    si_signal = superimposed(signal, fs)

    # Creador de la Window Function
    window_len_t = 1 / 60
    window_len = int((window_len_t) // dt)
    N = window_len
    step = 8

    window_name = "boxcar"
    window_function = wndw(window_name, window_len, fftbins=True)

    # Construcción de las ventanas móviles utilizando Multiprocessing
    # Construcción de las ventanas móviles
    windows = moving_window(signal, window_len, step)
    windows_si = moving_window(si_signal, window_len, step)
    windows_t = moving_window(t, window_len, step)

    # Parámetros del sistema
    I_avg_60Hz = 44.375
    I_F_60Hz = 72.95
    I_F_si = 67
    I_HIF_180Hz = 13.294  # Solo para falla en cabecera
    I_HIF_120Hz = 8.133

    # Parámetros de detector
    SI_420Hz_threshold = 5  # El valor normal es de 2
    Fault_current_threshold = I_avg_60Hz * 1.5  # 50% de la corriente nominal
    HIF_thld_180 = I_HIF_180Hz * 0.3

    xf = rfftfreq(N, dt)[: N // 2]
    # fault_max = np.empty(len(xf))
    # caps_max = np.empty(len(xf))
    fault_max = 0
    caps_max = 0

    for i, time_window in enumerate(windows_t):

        # Obtener la ventana i
        window = windows[i]
        window_si = windows_si[i]

        # Multiplicar por la función de ventana
        window = window * window_function
        window_si = window_si * window_function
        FFT_STFT = fourier(window, N)
        FFT_SI = fourier(window_si, N)

        # Se utiliza el valor porcentual basado en la componente de 60Hz
        # # Componentes para fallas
        I_per_120 = FFT_SI[2] / FFT_STFT[1]
        I_per_180 = FFT_SI[3] / FFT_STFT[1]
        I_per_240 = FFT_SI[4] / FFT_STFT[1]

        # Componentes para capacitores
        I_per_300 = FFT_SI[5] / FFT_STFT[1]
        I_per_360 = FFT_SI[6] / FFT_STFT[1]
        I_per_420 = FFT_SI[7] / FFT_STFT[1]
        I_per_480 = FFT_SI[8] / FFT_STFT[1]
        # # Componentes para fallas
        # I_per_120 = FFT_STFT[2] / FFT_STFT[1]
        # I_per_180 = FFT_STFT[3] / FFT_STFT[1]
        # I_per_240 = FFT_STFT[4] / FFT_STFT[1]

        # # Componentes para capacitores
        # I_per_300 = FFT_STFT[5] / FFT_STFT[1]
        # I_per_360 = FFT_STFT[6] / FFT_STFT[1]
        # I_per_420 = FFT_STFT[7] / FFT_STFT[1]
        # I_per_480 = FFT_STFT[8] / FFT_STFT[1]

        delta_h_f = np.sqrt(
            I_per_120 * I_per_120 + I_per_180 * I_per_180 + I_per_240 * I_per_240
        )
        delta_h_cap = np.sqrt(
            I_per_300 * I_per_300
            + I_per_360 * I_per_360
            + I_per_420 * I_per_420
            + I_per_480 * I_per_480
        )
        if delta_h_f > fault_max:
            fault_max = delta_h_f
        if delta_h_cap > caps_max:
            caps_max = delta_h_cap

        if delta_h_cap > 0.35 or delta_h_f > 0.35:
            if delta_h_f > delta_h_cap:
                window_span = time_window[-1] - time_window[0]
                detect_duration = time_window[-1] - 0.1
                # print("Detección fallas")
                return 1, i

            if delta_h_f < delta_h_cap:
                # print("Detección caps")
                return 0, i

    return 0, i


def load_events(PATH):
    for relay in relay_list:
        print(f"{relay}-----------------> NUEVO CASO")
        for phase in signal_list:
            signal_name = f"{relay}{phase}"


def main():
    print(__name__)


if __name__ == "__main__":
    main()

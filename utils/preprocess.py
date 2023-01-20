from utils.auxfunctions import superimposed, moving_window, fourier, wndw
from utils.signalload import CSV_pandas_path, CSV_pandas
import matplotlib.pyplot as plt
from itertools import repeat
import inspect
from numpy import tile


def windows_creator(
    window_len,
    signals=None,
    signal_name=None,
    step=4,
    window_name="boxcar",
    windows_return=True,
    windows_fourier=False,
):
    """
    Returns Moving windows for a signal

    Parameters
    ----------

    window_len: int

    signals: Numpy array (optional)
        Array of shape (1, n)

    signal_name: string (optional)
        Signal name contained within signals object

    step: int (optional)

    window_name: string
        windowing function name from scipy

    windows_return: boolean (optional)
        default = True, wether it returns moving windows for original signal
        and super imposed components signal

    windows_fourier: boolean (optional)
        wether to return fourier transform windows

    returns
    -------

    if (windows_fourier == True) and (windows_return == True)
    tuple of numpy array
        variables -> (signal, signal_si, t), (signal_fft, signal_si_fft, xf)
        dimensions-> ([n, N], [n, N], [n, N]), ([n, N/2], [n, N/2], [n])

    if windows_fourier == False
    numpy arrays
        variables ->  signal, signal_si, t
        dimensions -> [n, N], [n, N], [n, N]

    if windows_return == False
    numpy arrays
        variables -> signal_fft, signal_si_fft, xf
        dimensions-> [n, N/2], [n, N/2], [n]
    """
    if not signals:
        signals = CSV_pandas_path()
        signal_name = signals.labels_list[0]
    signal, t, params = signals.load_data(signal_name)

    # Preprocesamiento - componentes superimpuestas
    signal_si = superimposed(signal, params["fs"])
    dt = params["dt"]
    window_len_t = 1 / 60
    window_function = wndw(window_name, window_len, fftbins=True)
    # Creación de ventanas

    windows, windows_si, windows_t = list(
        map(moving_window, [signal, signal_si, t], repeat(window_len), repeat(step))
    )

    if windows_fourier == False:
        return windows, windows_si, windows_t
    xf, windows_fft = fourier(windows, window_len, dt)
    _, windows_si_fft = fourier(windows_si, window_len, dt)
    if windows_return == False:
        return windows_fft, windows_si_fft, xf
    return (windows, windows_si, windows_t), (windows_fft, windows_si_fft, xf)


if __name__ == "__main__":
    print("Preprocess module")

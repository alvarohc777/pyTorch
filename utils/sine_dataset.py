import numpy as np
import torch
import matplotlib.pyplot as plt
from random import getrandbits


def sine_harmonic(t, fs=3840, N=64, fundamental=60, harmonic=1, magnitude=1, phi=0):
    wt = 2 * np.pi * fs * t
    n = harmonic
    phi = phi
    phi_a = float(0) * np.pi / 180
    phi_a = phi_a + phi
    signal = magnitude * np.sin(n * (wt + phi_a))
    return signal


def sine_harmonics(signals, target, t, fs, m, mag_i=0.01, mag_f=100):
    for magnitude in np.linspace(mag_i, mag_f, int(m / 2)):
        phi = np.random.normal(-np.pi, np.pi)
        harmonic = np.random.normal(1, 8)
        signal = sine_harmonic(t, phi=phi, harmonic=harmonic, magnitude=magnitude)
        signals = np.vstack((signals, signal))
        target = np.vstack((target, np.array(1)))
    return signals, target


def sine_creator(signals, target, t, fs, m, mag_i=0.01, mag_f=100):
    for magnitude in np.linspace(mag_i, mag_f, int(m / 2)):
        signal = np.array(magnitude * np.sin(2 * np.pi * fs * t))
        signals = np.vstack((signals, signal))
        target = np.vstack((target, np.array(1)))
    return signals, target


def sine_phase_creator(signals, target, t, fs, m, mag_i=0.01, mag_f=100):
    for magnitude in np.linspace(mag_i, mag_f, int(m / 2)):
        phi = np.random.normal(-np.pi, np.pi)
        signal = magnitude * np.sin(2 * np.pi * fs * t + phi)
        signals = np.vstack((signals, signal))
        target = np.vstack((target, np.array(1)))
    return signals, target


# Crear esto como una transformación para los datos de entrada
def sine_noise(signals, target, t, fs, m, mag_i=0.01, mag_f=100):
    for magnitude in np.linspace(mag_i, mag_f, int(m / 2)):
        x = np.random.uniform()
        scale_var = 30 * x + 20
        gauss = np.random.normal(scale=magnitude / scale_var, size=len(t))
        phi = np.random.normal(-np.pi, np.pi)
        signal = magnitude * np.sin(2 * np.pi * fs * t + phi)
        signals = np.vstack((signals, signal + gauss))
        target = np.vstack((target, np.array(1)))
    return signals, target


def synthetic_signal(t, harmonics=[60], fundamental=60):
    """
    Function for the creation of a synthetic signal



    Returns
    -------
    numpy.array
        (3, n) three phase matrix
    """
    wt = 2 * np.pi * fundamental * t
    ang_A = float(0) * np.pi / 180
    ang_B = float(240) * np.pi / 180
    ang_C = float(120) * np.pi / 180

    A, B, C = 0, 0, 0

    for s in harmonics:
        n = s / fundamental
        A += 100 * np.sin(n * (wt + ang_A))
        B += 100 * np.sin(n * (wt + ang_B))
        C += 100 * np.sin(n * (wt + ang_C))

    return np.array((A, B, C))


def gaussian_creator(
    signals,
    target,
    N,
    m,
    scale=3,
):
    for i in range(int(m / 6)):
        signal = np.random.normal(scale=scale, size=N)
        signals = np.vstack((signals, signal))
        target = np.vstack((target, np.array(0)))
    return signals, target


def constant_creator(signals, target, N, m):
    for magnitude in range(int(m / 6)):
        signal = np.ones(N) * magnitude / 10
        signals = np.vstack((signals, signal))
        target = np.vstack((target, np.array(0)))
    return signals, target


def slope_creator(signals, target, t, N, m):
    for slope in range(int(m / 6)):
        positive_slope = bool(getrandbits(1))
        if not positive_slope:
            slope = -slope
        signal = slope * t
        signals = np.vstack((signals, signal))
        target = np.vstack((target, np.array(0)))
    return signals, target


def signal_dataset_creator(fs, N, m, mag_i=0.01, mag_f=100):

    t = np.linspace(0, N / fs, N)

    # Initialize signals vector
    signals = t
    target = np.array([0])

    # signals, target = sine_creator(signals, target, t, fs, m)
    signals, target = sine_phase_creator(signals, target, t, fs, m)
    signals, target = gaussian_creator(signals, target, N, m)
    signals, target = constant_creator(signals, target, N, m)
    signals, target = slope_creator(signals, target, t, N, m)

    return signals, target


def main():

    fs = 3840
    N = 64
    t = np.linspace(0, N / fs, N)
    m = 1000

    # Initialize signals vector
    signals = t
    target = np.array([0])
    print(t[-1])
    mag_i = 0.01
    mag_f = 100
    signals, target = signal_dataset_creator(fs, N, m, mag_i, mag_f)
    print(signals.shape)
    print(target.shape)

    plt.stem(signals[10])
    plt.show()

    # Para poder utilizar como input de LSTM
    #   esta espera (m, N, s), donde s son cantidad de señales
    #   en este caso solo hay una señal
    signals = np.expand_dims(signals, axis=2)
    signals = torch.from_numpy(signals).float()
    target = torch.from_numpy(target).float()
    print(target.shape)
    print(signals.shape)


if __name__ == "__main__":
    main()

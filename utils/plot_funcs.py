import numpy as np
import matplotlib.pyplot as plt


def signal_plt(t, signal):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.plot(t, signal)
    return fig


def plot_signal_fft(t, signal, t_window, trip):
    """
    Plotting function shows signal in time domain and the trip signal in time

    parameters
    ----------

    t: np.array

    signal: np.array

    t_window: np.array

    trip: list

    returns
    -------

    fig: plt.fig

    """
    t_window = np.insert(t_window[:, -1], 0, 0)
    fig = plt.figure()
    axis1 = fig.add_subplot(2, 1, 1)
    axis2 = fig.add_subplot(2, 1, 2)

    axis1.step(t_window, trip)
    axis2.step(t, signal)

    return fig

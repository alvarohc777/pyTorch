from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np


def signal_animation(x, y):
    """
    Function to create animation object for pair of signal and time windows

    parameters
    ----------

    x: np.array
        time windows array

    y: np.array
        signal windows array

    returns
    -------

    fig: matploblib fig

    animate: function

    ini: function

    frames: int
        number of frames in animation
    """
    # Setting de la gráfica
    ymin, ymax = np.min(y), np.max(y)
    frames = len(x)

    fig = plt.figure()
    axis = fig.add_subplot(111)
    (line,) = axis.plot([], [])

    def ini():
        line.set_data([], [])
        return (line,)

    def animate(i):
        line.set_data(x[i], y[i])
        axis.set_xlim(x[i, 0], x[i, -1])
        axis.set_ylim(ymin, ymax)
        return (line,)

    return fig, animate, ini, frames


def fft_animation(x, y):
    # Setting de la gráfica
    x = np.tile(x, (len(y), 1))
    ymin, ymax = np.min(y), np.max(y)
    frames = len(x)

    fig = plt.figure()
    axis = fig.add_subplot(111)
    (line,) = axis.plot([], [])

    def ini():
        line.set_data([], [])
        return (line,)

    def animate(i):
        line.set_data(x[i], y[i])
        axis.set_xlim(x[i, 0], x[i, -1])
        axis.set_ylim(ymin, ymax)
        return (line,)

    return fig, animate, ini, frames


def signal_render(anims_list, interval=200):
    """
    Function to animate moving windows

    parameters
    ----------

    anims_list: list of tuples
        list of tuples (fig, animate, ini, frames), these are outputs of
        signal_animation function

    interval: int (optional)
        speed of the animations in ms

    returns
    -------

    anim: FuncAnimation object
    """

    anim = []
    for figcomps in anims_list:
        fig, animate, ini, frames = figcomps
        anim.append(
            FuncAnimation(
                fig, animate, init_func=ini, frames=frames, interval=200, blit=True
            )
        )
    return anim

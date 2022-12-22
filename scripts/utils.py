import matplotlib.pyplot as plt
from matplotlib import animation
import bisect
import numpy as np
from tsmoothie.smoother import *

from IPython.display import clear_output


def convert_state(state):
    ranges = [
        np.arange(-4.8, 4.8, 0.01),
        np.arange(-100, 100, 0.01),
        np.arange(-0.5, 0.5, 0.001),
        np.arange(-100, 100, 0.01),
    ]
    output = []
    for i in range(len(state)):
        index = int(bisect.bisect_left(ranges[i], state[i]))
        output += [str(ranges[i][min(index, len(ranges[i]) - 1)])]
    return ",".join(output)


def plot_smooth(y, axis, label):
    # operate smoothing
    smoother = SplineSmoother(n_knots=10, spline_type="natural_cubic_spline")
    smoother.smooth(y)
    # plot the first smoothed timeseries with intervals
    axis.plot(smoother.smooth_data[0], linewidth=3, color="blue", label=label)
    axis.plot(smoother.data[0], ".k")


def plot_res(values, title=""):
    try:
        clear_output(wait=True)
        f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        f.suptitle(title)
        xs = range(len(values))
        plot_smooth(values, ax[0], label="score per run")
        ax[0].axhline(195, c="red", ls="--", label="goal")
        ax[0].set_xlabel("Episodes")
        ax[0].set_ylabel("Reward")
        ax[0].legend()
        try:
            z = np.polyfit(xs, values, 1)
            p = np.poly1d(z)
            ax[0].plot(xs, p(xs), "--", label="trend")
        except:
            print("error")

        ax[1].hist(values[-50:])
        ax[1].axvline(195, c="red", label="goal")
        ax[1].set_xlabel("Scores per Last 50 Episodes")
        ax[1].set_ylabel("Frequency")
        ax[1].legend()
        plt.show()
    except:
        pass


def save_frames_as_gif(frames, path="./images", filename="cart_pole.gif"):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer="imagemagick", fps=60)

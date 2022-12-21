
import matplotlib.pyplot as plt
import bisect
import numpy as np

from IPython.display import clear_output


def convert_state(state):
    ranges = [
        np.arange(-4.8, 4.8, 0.01), 
        np.arange(-100,100, 0.01),
        np.arange(-0.5, 0.5, 0.001),
        np.arange(-100,100, 0.01)
    ]
    output = []
    for i in range(len(state)):
        index = int(bisect.bisect_left(ranges[i], state[i]))
        output += [str(ranges[i][min(index, len(ranges[i])-1)])]
    return ",".join(output)


def plot_res(values, title=''):   
    clear_output(wait=True)

    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    f.suptitle(title)
    ax[0].plot(values, label='score per run')
    ax[0].axhline(195, c='red',ls='--', label='goal')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    x = range(len(values))
    ax[0].legend()
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x,p(x),"--", label='trend')
    except:
        print('')
    
    ax[1].hist(values[-50:])
    ax[1].axvline(195, c='red', label='goal')
    ax[1].set_xlabel('Scores per Last 50 Episodes')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    plt.show()
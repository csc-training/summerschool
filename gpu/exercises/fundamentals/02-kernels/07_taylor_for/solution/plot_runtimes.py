import numpy as np
import matplotlib.pyplot as plt

def plot_runtimes(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    taylor_iters            = data[:, 0]
    vec_size                = data[:, 1]
    microseconds            = data[:, 2]
    relative_vec            = data[:, 3]
    relative_strided        = data[:, 4]
    relative_consecutive    = data[:, 5]
    relative_vec_for        = data[:, 6]

    num_vec_sizes = np.argwhere(np.diff(taylor_iters))[0, 0] + 1
    num_taylor_iters = len(taylor_iters) / num_vec_sizes

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    fig.tight_layout()

    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)

    ax1.set_title("Strided loop", fontsize=40)
    ax2.set_title("Consecutive elements loop", fontsize=40)
    ax3.set_title("Vectorized loads, strided loop", fontsize=40)
    ax4.set_title("Vectorized loads", fontsize=40)

    ax1.tick_params(labelsize=30)
    ax2.tick_params(labelsize=30)
    ax3.tick_params(labelsize=30)
    ax4.tick_params(labelsize=30)

    ax1.set_xlabel("number of elements", fontsize=30)
    ax2.set_xlabel("number of elements", fontsize=30)
    ax3.set_xlabel("number of elements", fontsize=30)
    ax4.set_xlabel("number of elements", fontsize=30)

    ax1.set_ylabel("runtime relative to base", fontsize=30)
    ax2.set_ylabel("runtime relative to base", fontsize=30)
    ax3.set_ylabel("runtime relative to base", fontsize=30)
    ax4.set_ylabel("runtime relative to base", fontsize=30)

    for i in np.arange(0, num_taylor_iters):
        start = int(i * num_vec_sizes)
        stop = int((i + 1) * num_vec_sizes)

        ax1.semilogx(vec_size[start:stop], relative_strided[start:stop], 'o-', lw=3.0, label='Taylor N = ' + str(1 << int(i)))
        ax2.semilogx(vec_size[start:stop], relative_consecutive[start:stop], 'o-', lw=3.0)
        ax3.semilogx(vec_size[start:stop], relative_vec_for[start:stop], 'o-', lw=3.0)
        ax4.semilogx(vec_size[start:stop], relative_vec[start:stop], 'o-', lw=3.0)

    fig.legend(loc='outside right upper', fontsize=30)
    plt.show()

if __name__ == "__main__":
    plot_runtimes("runtimes.dat")

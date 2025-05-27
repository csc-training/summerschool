import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def make_fig(byte_sizes, block_sizes, titles, ki):
    fig_width = 36.0
    aspect_ratio = 9.0 / 16.0
    fig, _ = plt.subplots(nrows=2,
                          ncols=2,
                          figsize=(fig_width, fig_width * aspect_ratio),
                          sharex=True,
                          sharey=True,
                          layout='constrained')

    fig.suptitle(titles[ki], fontsize=40)

    xticks = [str(int(i)) for i in block_sizes]
    yticks = [str(int(i / 1e6)) + "MB" for i in byte_sizes]

    # Remove x ticks from top row
    fig.axes[0].tick_params(axis='x', which='both', bottom=False, top=False)
    fig.axes[1].tick_params(axis='x', which='both', bottom=False, top=False)

    # Add x labels to bottom row
    fig.axes[2].set_xlabel("block size", fontsize=30)
    fig.axes[3].set_xlabel("block size", fontsize=30)
    fig.axes[2].set_xticks(range(len(xticks)), labels=xticks, rotation=45)
    fig.axes[3].set_xticks(range(len(xticks)), labels=xticks, rotation=45)

    # Remove y ticks from right col
    fig.axes[1].tick_params(axis='y', which='both', left=False, right=False)
    fig.axes[3].tick_params(axis='y', which='both', left=False, right=False)

    # Add y labels to left col
    fig.axes[0].set_ylabel("bytes", fontsize=30)
    fig.axes[2].set_ylabel("bytes", fontsize=30)
    fig.axes[0].set_yticks(range(len(yticks)), labels=yticks, rotation=45)
    fig.axes[2].set_yticks(range(len(yticks)), labels=yticks, rotation=45)

    fig.axes[0].tick_params(labelsize=30)
    fig.axes[2].tick_params(labelsize=30)
    fig.axes[3].tick_params(labelsize=30)

    return fig

def plot_data(title, subplot_titles, data, fig, colorbar_title):
    norm = colors.Normalize(vmin=np.min(data), vmax=np.max(data))
    images = []
    for page_index in np.arange(data.shape[0]):
        fig.axes[page_index].set_title(subplot_titles[page_index], fontsize=30)
        images.append(fig.axes[page_index].imshow(data[page_index], cmap='RdBu', origin='lower', norm=norm))

    cb = plt.colorbar(images[0], ax=fig.axes, aspect=80)
    cb.set_label(colorbar_title, fontsize=30)
    cb.ax.tick_params(labelsize=30)

    fig.get_layout_engine().set(w_pad=0.1, h_pad=0.1, hspace=0, wspace=0)

    plt.savefig(title.lower().replace(' ', '_') + ".svg")

def plot_relative_runtimes(taylor_iters, byte_sizes, block_sizes, runtimes, titles):
    assert len(taylor_iters) >= 4
    subplot_titles = ["N = " + str(int(i)) for i in taylor_iters]

    base = runtimes[0]
    colorbar_title = "relative runtime"
    for ki in np.arange(1, runtimes.shape[0]):
        fig = make_fig(byte_sizes, block_sizes, titles, ki)

        kernel = runtimes[ki]
        relative_cube = kernel / base
        plot_data("runtime_" + titles[ki], subplot_titles, relative_cube, fig, colorbar_title)

    colorbar_title = "relative deviation from row average"
    for ki in np.arange(runtimes.shape[0]):
        fig = make_fig(byte_sizes, block_sizes, titles, ki)
        kernel = runtimes[ki]
        averages = np.average(kernel, axis=2).reshape((kernel.shape[0], kernel.shape[1], 1))
        deviations = (kernel - averages) / averages
        plot_data("deviation_" + titles[ki], subplot_titles, deviations, fig, colorbar_title)

def main():
    filename = "runtimes.dat"
    with open(filename, 'r') as f:
        line = f.readline()
        words = line.split(',')
        titles = [word.strip().rstrip() for word in words[3:]]

    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    taylor_iters = data[:, 0]
    byte_sizes = data[:, 1]

    num_block_sizes = int(np.argwhere(np.diff(byte_sizes))[0, 0] + 1)
    num_vec_sizes = int((np.argwhere(np.diff(taylor_iters))[0, 0] + 1) / num_block_sizes)
    num_taylor_iters = int(len(taylor_iters) / (num_block_sizes * num_vec_sizes))

    # Reshape to 4D data: indexing to data gives 3D data cubes of runtimes
    data = data.ravel(order='F').reshape(
            (data.shape[1], num_taylor_iters, num_vec_sizes, num_block_sizes), order='C')

    # 1D arrays describing the values of the axes
    taylor_iters = data[0, :, 0, 0]
    byte_sizes = data[1, 0, :, 0]
    block_sizes = data[2, 0, 0, :]

    # N x 3D data cubes
    # Where N is the number of different kernels that have been measured
    # For each cube
    # - first index: Taylor iteration count
    # - second index: bytes taken by the data
    # - third index: block size
    # The values are average runtimes of 19 kernel invocations in microseconds
    runtimes = data[3:]

    plot_relative_runtimes(taylor_iters, byte_sizes, block_sizes, runtimes, titles)

if __name__ == "__main__":
    main()

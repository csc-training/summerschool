import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def plot(data, kernel_index, title):
    fig_width = 36.0
    aspect_ratio = 9.0 / 16.0

    taylor_iters    = data[:, 0]
    vec_size        = data[:, 1]
    block_size      = data[:, 2]
    base            = data[:, 3]
    kernel          = data[:, kernel_index]

    num_block_sizes = int(np.argwhere(np.diff(vec_size))[0, 0] + 1)
    num_vec_sizes = int((np.argwhere(np.diff(taylor_iters))[0, 0] + 1) / num_block_sizes)
    num_taylor_iters = int(len(taylor_iters) / (num_block_sizes * num_vec_sizes))
    data_stride = num_vec_sizes * num_block_sizes

    xticks = [str(int(i)) for i in block_size[:num_block_sizes]]
    yticks = [str(int(i / 1e6)) + "MB" for i in vec_size[:num_vec_sizes * num_block_sizes:num_block_sizes]]

    # Only plotting the first 4 taylor iters
    fig, _ = plt.subplots(nrows=2,
                          ncols=2,
                          figsize=(fig_width, fig_width * aspect_ratio),
                          sharex=True,
                          sharey=True,
                          layout='constrained')
                          
    fig.suptitle(title, fontsize=40)

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

    norm = colors.Normalize(vmin=np.min(kernel / base), vmax=np.max(kernel / base))
    images = []
    for nti in np.arange(num_taylor_iters):
        start = nti * data_stride
        stop = (nti + 1) * data_stride

        base_data = base[start:stop]
        kernel_data = kernel[start:stop]

        data = (kernel_data / base_data).reshape(num_vec_sizes, num_block_sizes)

        fig.axes[nti].set_title("N = " + str(int(1 << nti)), fontsize=30)
        images.append(fig.axes[nti].imshow(data, cmap='RdBu', origin='lower', norm=norm))

    cb = plt.colorbar(images[0], ax=fig.axes, aspect=80)
    cb.set_label("relative runtime", fontsize=30)
    cb.ax.tick_params(labelsize=30)

    fig.get_layout_engine().set(w_pad=0.1, h_pad=0.1, hspace=0, wspace=0)

    plt.savefig(title.lower().replace(' ', '_') + ".svg")
    #plt.show()

def plot_runtimes_new(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    plot(data, 4, "Vectorized")
    plot(data, 5, "Strided loop")
    plot(data, 6, "Consecutive values loop")
    plot(data, 7, "Vectorized loop")

def plot_runtimes(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    taylor_iters    = data[:, 0]
    vec_size        = data[:, 1]
    block_size      = data[:, 2]
    base            = data[:, 3]
    vec             = data[:, 4]
    strided         = data[:, 5]
    consecutive     = data[:, 6]
    vec_for         = data[:, 7]

    num_block_sizes = int(np.argwhere(np.diff(vec_size))[0, 0] + 1)
    num_vec_sizes = int((np.argwhere(np.diff(taylor_iters))[0, 0] + 1) / num_block_sizes)
    num_taylor_iters = int(len(taylor_iters) / (num_block_sizes * num_vec_sizes))
    data_stride = num_vec_sizes * num_block_sizes

    titles = ["Strided loop", "Consecutive elements loop", "Vectorized loads, strided loop", "Vectorized loads"]
    xticks = [str(int(i)) for i in block_size[:num_block_sizes]]
    yticks = [str(int(i / 1e6)) + "e6" for i in vec_size[:num_vec_sizes * num_block_sizes:num_block_sizes]]

    fig_width = 36.0
    rect = (0.05, 0.05, 1, 1)
    aspect_ratio = 9.0 / 16.0

    for i in np.arange(num_taylor_iters):
        fig, _ = plt.subplots(nrows=2, ncols=2, figsize=(fig_width, fig_width * aspect_ratio), sharex=True, sharey=True)
        fig.tight_layout(rect=rect)

        fig.axes[2].set_xlabel("block size", fontsize=30)
        fig.axes[3].set_xlabel("block size", fontsize=30)

        fig.axes[0].set_ylabel("#elements", fontsize=30)
        fig.axes[2].set_ylabel("#elements", fontsize=30)

        fig.axes[2].set_xticks(range(len(xticks)), labels=xticks, rotation=45)
        fig.axes[3].set_xticks(range(len(xticks)), labels=xticks, rotation=45)

        fig.axes[0].set_yticks(range(len(yticks)), labels=yticks, rotation=45)
        fig.axes[2].set_yticks(range(len(yticks)), labels=yticks, rotation=45)
        
        fig.axes[0].tick_params(labelsize=30)
        fig.axes[2].tick_params(labelsize=30)
        fig.axes[3].tick_params(labelsize=30)

        fig.axes[0].tick_params(axis='x', which='both', bottom=False, top=False)
        fig.axes[1].tick_params(axis='x', which='both', bottom=False, top=False)
        fig.axes[1].tick_params(axis='y', which='both', left=False, right=False)
        fig.axes[3].tick_params(axis='y', which='both', left=False, right=False)

        start = i * data_stride
        stop = (i + 1) * data_stride

        base_data = base[start:stop]
        datas = [strided[start:stop], consecutive[start:stop], vec_for[start:stop], vec[start:stop]]

        for j in np.arange(len(fig.axes)):
            fig.axes[j].set_title(titles[j], fontsize=40)

            data = (datas[j] / base_data).reshape(num_vec_sizes, num_block_sizes)

            r = fig.axes[j].imshow(data, cmap='RdBu', origin='lower', norm='linear')
            if j & 1:
                cb = fig.colorbar(r, ax=fig.axes[j], extend='both')
                cb.set_label("relative runtime", fontsize=30)
                cb.ax.tick_params(labelsize=30)

        plt.savefig("runtimes_" + str(int(taylor_iters[start])) + ".svg")

        fig, _ = plt.subplots(nrows=1, ncols=1, figsize=(fig_width, fig_width * aspect_ratio))
        fig.tight_layout(rect=rect)
        fig.axes[0].set_title(r"Base kernel runtime [$\mu s$]", fontsize=40)
        fig.axes[0].set_xticks(range(len(xticks)), labels=xticks, rotation=45)
        fig.axes[0].set_yticks(range(len(yticks)), labels=yticks, rotation=45)
        fig.axes[0].tick_params(labelsize=30)
        fig.axes[0].set_xlabel("block size", fontsize=30)
        fig.axes[0].set_ylabel("#elements", fontsize=30)

        data = base_data.reshape(num_vec_sizes, num_block_sizes)
        r = fig.axes[0].imshow(data, cmap='RdBu', origin='lower', norm='log')
        cb = fig.colorbar(r, ax=fig.axes[0])
        cb.set_label(r"$\mu s$", fontsize=30)
        cb.ax.tick_params(labelsize=30)
        plt.savefig("base_runtime_" + str(int(taylor_iters[start])) + ".svg")

if __name__ == "__main__":
    plot_runtimes_new("runtimes.dat")

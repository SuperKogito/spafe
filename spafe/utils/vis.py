import matplotlib
import matplotlib.pyplot as plt


def visualize_fbanks(fbanks, ylabel, xlabel):
    """
    visualize a matrix including the filterbanks coordinates. Each row corresponds
    to a filter.

    Args:
        fbanks (array) : 2d array including the the filterbanks coordinates.
        ylabel   (str) : y-axis label.
        xlabel   (str) : x-axis label.
    """
    for fbank in fbanks:
        plt.plot(fbank)
        plt.ylim(0, 1.1)
        plt.grid(True)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
    plt.show(block=False)
    plt.close()


def visualize_features(feats, ylabel, xlabel, cmap='viridis'):
    """
    visualize a matrix including the features coefficients. Each row corresponds
    to a frame.

    Args:
        feats  (array) : 2d array including the the features coefficients.
        ylabel   (str) : y-axis label.
        xlabel   (str) : x-axis label.
        cmap     (str) : matplotlib colormap to use.
    """
    plt.imshow(feats.T,
               origin='lower',
               aspect='auto',
               cmap=cmap,
               interpolation='nearest')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show(block=False)
    plt.close()


def plot(y, ylabel, xlabel):
    """
    plot an array y.

    Args:
        y      (array) : 1d array to plot.
        ylabel   (str) : y-axis label.
        xlabel   (str) : x-axis label.
    """
    plt.plot(y)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show(block=False)
    plt.close()


def spectogram(sig, fs):
    """
    visualize a the spectogram of the given mono signal.

    Args:
        sig (array) : a mono audio signal (Nx1) from which to compute features.
        fs    (int) : the sampling frequency of the signal we are working with.
    """
    plt.specgram(sig, NFFT=1024, Fs=fs)
    plt.ylabel("Frequency (kHz)")
    plt.xlabel("Time (s)")
    plt.show(block=False)
    plt.close()


def multi_plots(data, fs, plot_rows, step,
                colors=["b", "r", "m", "g", "b", "y", "r-."]):
    """
    Generate multiple plots related to same signal in one figure.

    Args:
        data    (array) : array of arrays to plot.
        fs        (int) : the sampling frequency of the signal we are working with.
        plot_rows (int) : number of rows to plot.
        step      (int) : array reading step.
        colors   (list) : list of colors for the plots.
    """

    font = {'family' : 'normal',
            'size'   : 7}
    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(50, 25))
    plt.subplots(plot_rows, 1)
    plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.25, hspace=0.75)

    for i in range(plot_rows):
        plt.subplot(plot_rows, 1, i+1)
        y = data[i]
        plt.plot([i/fs for i in range(0, len(y), step)], y, colors[i])
    plt.show()

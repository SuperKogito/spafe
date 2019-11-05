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
    for i in range(len(fbanks)):
        plt.plot(fbanks[i])
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
    plt.imshow(feats.T, origin='lower', aspect='auto',
               cmap=cmap,  interpolation='nearest')
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

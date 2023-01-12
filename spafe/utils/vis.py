"""

- Description : Visualization implementation.
- Copyright (c) 2019-2023 Ayoub Malek.
  This source code is licensed under the terms of the BSD 3-Clause License.
  For a copy, see <https://github.com/SuperKogito/spafe/blob/master/LICENSE>.

"""
import numpy as np

from spafe.utils.converters import hz2mel, hz2bark, hz2erb


def tick_function(X, fb_type):
    """
    Return correct converted ticks.

    Args:
        X (numpy.ndarray) : input ticks.
        fb_type     (str) : filter banks type to choose the correct conversion.

    Returns:
        (numpy.ndarray) : converted ticks to the correct scale.
    """
    return {
        "mel": ["%.1f" % z for z in [hz2mel(y) for y in X]],
        "bark": ["%.1f" % z for z in [hz2bark(y) for y in X]],
        "gamma": ["%.1f" % z for z in [hz2erb(y) for y in X]],
    }.get(fb_type, ["%.1f" % z for z in X])


def show_fbanks(
    fbanks,
    center_freqs,
    ref_freqs,
    title="Mel filter banks",
    ylabel="Weight",
    x1label="Frequency / Hz",
    x2label="Frequency / mel",
    figsize=(12, 5),
    fb_type="mel",
    show_center_freqs=True,
):
    """
    visualize a matrix including the filter banks coordinates. Each row corresponds
    to a filter.

    Args:
        fbanks    (numpy.ndarray) : 2d array including the the filter banks coordinates.
        ref_freqs (numpy.ndarray) : 1d array reference frequencies
        title               (str) : plot title.
                                    (Default is "Mel filter banks").
        ylabel              (str) : y-axis label.
                                    (Default is "Weight").
        x1label             (str) : lower x-axis label.
                                    (Default is "Frequency/ Hz").
        x2label             (str) : upper x-axis label.
                                    (Default is "Frequency/ mel").
        figsize           (tuple) : size of figure.
                                    (Default is (12, 5)).
        fb_type             (str) : type of filter banks.
                                    (Default is "mel").
        show_center_freqs  (bool) : if true show center frequencies.
                                    (Default is True).
    """
    import matplotlib.pyplot as plt

    # init plot
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    # format and plot data
    xvals = ref_freqs
    data = fbanks.T
    ax1.plot(xvals, data, "-")

    if show_center_freqs:
        for f in center_freqs:
            plt.vlines(f, 0, 1, "grey", ":")
            plt.text(f, 1, "({:.1f}, {})".format(f, 1))

    ax1Ticks = ax1.get_xticks()
    ax2Ticks = ax1Ticks

    ax2.set_xticks(ax2Ticks)
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticklabels(tick_function(ax2Ticks, fb_type))

    # fix axes
    ax1.set_title(title)
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel(x1label)
    ax2.set_xlabel(x2label)
    ax1.grid(True)
    plt.show()


def show_spectrogram(
    spectrogram,
    fs,
    xmin,
    xmax,
    ymin,
    ymax,
    dbf=80.0,
    xlabel="Time (s)",
    ylabel="Frequency (Hz)",
    title="Mel spectrogram (dB)",
    figsize=(14, 4),
    cmap="jet",
    colorbar=True,
):
    """
    Visualize the spectrogram.


    Args:
        feats (numpy.ndarray) : 2d array including the the features coefficients.
        fs              (int) : sampling rate.
        xmin          (float) : minimum x-axis value.
        xmax          (float) : maximum x-axis value.
        ymin          (float) : minimum y-axis value.
        ymax          (float) : maximum y-axis value.
        dbf           (float) : db reference value.
                                (Default is 80.0).
        xlabel          (str) : x-axis label.
                                (Default is "Time (s)").
        ylabel          (str) : y-axis label.
                                (Default is "Frequency (Hz)").
        title           (str) : plot title.
                                (Default is "Mel spectrogram (dB)").
        figsize       (tuple) : size of figure.
                                (Default is (14, 4)).
        cmap            (str) : matplotlib colormap to use.
                                (Default is "jet").
        colorbar       (bool) : if true add colorbar.
                                (Default is True).
    """
    import matplotlib.pyplot as plt

    # init vars
    amin = 1e-10
    magnitude = np.abs(spectrogram)
    ref_value = np.max(magnitude)

    # compute log spectrum (in dB)
    log_spec = 10.0 * np.log10(
        np.maximum(amin, magnitude) / np.maximum(amin, ref_value)
    )
    log_spec = np.maximum(log_spec, log_spec.max() - dbf)

    # Display the mel spectrogram in dB, seconds, and Hz
    plt.figure(figsize=figsize)
    mSpec_img = plt.imshow(
        log_spec,
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        aspect="auto",
        cmap=cmap,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    if colorbar:
        plt.colorbar(mSpec_img, format="%+2.0f dB")
    plt.show()


def show_features(feats, title, ylabel, xlabel, figsize=(14, 4), cmap="jet"):
    """
    visualize a matrix including the features coefficients. Each row corresponds
    to a frame.

    Args:
        feats (numpy.ndarray) : 2d array including the the features coefficients.
        title           (str) : plot title.
        ylabel          (str) : y-axis label.
        xlabel          (str) : x-axis label.
        figsize       (tuple) : size of figure.
                                (Default is (14, 4)).
        cmap            (str) : matplotlib colormap to use.
                                (Default is "jet").
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    plt.imshow(
        feats.T, origin="lower", aspect="auto", cmap=cmap, interpolation="nearest"
    )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show(block=False)

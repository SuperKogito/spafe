import numpy as np
import functions, conversion
from __future__ import division

################################################################################
#                      Mel-filter-banks implementation
################################################################################

def hz2mel(hz):
    """
    Convert a value in Hertz to Mels

    Args:
         hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.

    Returns:
        a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * numpy.log10(1 + hz / 700.)

def mel2hz(mel):
    """
    Convert a value in Mels to Hertz

    Args:
        mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.

    Returns:
        a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700 * (10**(mel / 2595.0) - 1)

def mel_filter_banks(nfilt=20, nfft= 512, samplerate=16000, lowfreq=0, highfreq=None):
    """
    Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    Args:
        nfilt: the number of filters in the filterbank, default 20.
        nfft: the FFT size. Default is 512.
        samplerate: the sample rate of the signal we are working with, in Hz. Affects mel spacing.
        lowfreq: lowest band edge of mel filters, default 0 Hz
        highfreq: highest band edge of mel filters, default samplerate/2

    Returns:
        A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq  = highfreq or samplerate/2
    # compute points evenly spaced in mels
    lowmel    = hz2mel(lowfreq)
    highmel   = hz2mel(highfreq)
    melpoints = numpy.linspace(lowmel, highmel, nfilt + 2)
    # our points are in Hz, but we use fft bins, so we have to convert from Hz to fft bin number
    bin   = numpy.floor((nfft + 1) * conversion.mel2hz(melpoints) / samplerate)
    fbank = numpy.zeros([nfilt, nfft // 2 + 1])
ear_q
    for j in range(0, nfilt):
        for i in range(int(bin[j]),   int(bin[j+1])): fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])): fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank

################################################################################
#                    Gammatone-filter-banks implementation
################################################################################
"""
based on:
    https://github.com/jthiem/gtfblib
"""

def Hz2ERBnum(Hz):
    """
    Return the ERB filter number for a given frequency.

    Args:
         hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    """
    return 21.4 * np.log10(Hz * 4.37e-3 + 1.0)

def ERBnum2Hz(ERB):
    """
    Return the frequency of given ERB filter.
    """
    return (10**(ERB / 21.4) - 1.0) / 4.37e-3

def ERBspacing_given_N(cf_first, cf_last, N):
    ERB_first = Hz2ERBnum(cf_first)
    ERB_last  = Hz2ERBnum(cf_last)
    cfs       = ERBnum2Hz(np.linspace(ERB_first, ERB_last, N))
    return cfs

def ERBspacing_given_spacing(cf_first, cf_last, ERBstep):
    ERB_first = Hz2ERBnum(cf_first)
    ERB_last  = Hz2ERBnum(cf_last)
    cfs       = ERBnum2Hz(np.arange(ERB_first, ERB_last, ERBstep))
    return cfs

def gammatone_filter_banks(nfilt=20, nfft= 512, fs=16000, lowfreq=0, highfreq=None,
                           cfs=None, EarQ=(1/0.108), Bfact=1.0186):
    """
    Compute the Gammatone filter banks

    Args:
        nfilt: the number of filters in the filterbank, default 20.
        nfft: the FFT size. Default is 512.
        samplerate: the sample rate of the signal we are working with, in Hz. Affects mel spacing.
        lowfreq: lowest band edge of mel filters, default 0 Hz
        highfreq: highest band edge of mel filters, default samplerate/2

    Returns:
        X
    """
    if cfs is None:
        cfs = ERBspacing_given_N(80, 0.9*fs/2, 32)

    ERB = cfs/EarQ+24.7
    nfilt = cfs.shape[0]

    # shortcuts used by derived methods
    _omega = 2*np.pi*cfs/fs
    _normB = 2*np.pi*Bfact*self.ERB/fs



###############################################################################
#
###############################################################################
"""This module implements gammatone filters and a filtering routine.

A filterbank is coming soon [Hohmann2002]_.

.. plot::

    import gammatone
    gammatone.example()


TODO:
    - Tests,
    - nice introduction with example,
    - implementing the filterbank class

References
----------

.. [Hohmann2002]
   Hohmann, V., Frequency analysis and synthesis using a Gammatone filterbank,
   Acta Acustica, Vol 88 (2002), 433--442


Functions
---------
"""

import numpy as np
from numpy.fft import rfft, rfftfreq
from numpy import (arange, array, pi, cos, exp, log10, ones_like, sqrt, zeros)
from scipy.special import factorial
from scipy.signal import lfilter


# ERB means "Equivalent retangular band(-width)"
# Constants:
_ERB_L = 24.7
_ERB_Q = 9.265


def erb_count(centerfrequency):
    """Returns the equivalent rectangular band count up to centerfrequency.

    Parameters
    ----------
    centerfrequency : scalar /Hz
        The center frequency in Hertz of the
        desired auditory filter.

    Returns
    -------
    count : scalar
        Number of equivalent bandwidths below `centerfrequency`.

    """
    return 21.4 * log10(4.37 * 0.001 * centerfrequency + 1)


def erb_aud(centerfrequency):
    """Retrurns equivalent rectangular band width of an auditory filter.
    Implements Equation 13 in [Hohmann2002]_.

    Parameters
    ----------
    centerfrequency : scalar /Hz
        The center frequency in Hertz of the
        desired auditory filter.

    Returns
    -------
    erb : scalar
        Equivalent rectangular bandwidth of
        an auditory filter at `centerfrequency`.

    """
    return _ERB_L + centerfrequency / _ERB_Q


def hertz_to_erbscale(frequency):
    """Returns ERB-frequency from frequency in Hz.
    Implements Equation 16 in [Hohmann2002]_.

    Parameters
    ----------
    frequency : scalar
        The Frequency in Hertz.

    Returns
    -------
    erb : scalar
        The corresponding value on the ERB-Scale.

    """
    return _ERB_Q * np.log(1 + frequency / (_ERB_L * _ERB_Q))


def erbscale_to_hertz(erb):
    """Returns frequency in Hertz from ERB value.
    Implements Equation 17 in [Hohmann2002]_.

    Parameters
    ----------
    erb : scalar
        The corresponding value on the ERB-Scale.

    Returns
    -------
    frequency : scalar
        The Frequency in Hertz.

    """
    return (exp(erb/_ERB_Q) - 1) * _ERB_L * _ERB_Q


def frequencies_gammatone_bank(start_band, end_band, norm_freq, density):
    """Returns centerfrequencies and auditory Bandwidths
    for a range of gamatone filters.

    Parameters
    ----------
    start_band : int
        Erb counts below norm_freq.
    end_band : int
        Erb counts  over norm_freq.
    norm_freq : scalar
        The reference frequency where all filters are around
    density : scalar
        ERB density 1would be `erb_aud`.

    Returns
    -------
    centerfrequency_array : ndarray

    """
    norm_erb = hertz_to_erbscale(norm_freq)
    centerfrequencies = erbscale_to_hertz(
        arange(start_band, end_band, density) + norm_erb)
    return centerfrequencies


def design_filter(
        sample_rate=44100,
        order=4,
        centerfrequency=1000.0,
        band_width=None,
        band_width_factor=1.0,
        attenuation_half_bandwidth_db=-3):
    """Returns filter coefficient of a gammatone filter
    [Hohmann2002]_.

    Parameters
    ----------
    sample_rate : int/scalar
    order : int
    centerfrequency : scalar
    band_width : scalar
    band_width_factor : scalar
    attenuation_half_bandwidth_db : scalar

    Returns
    -------
    b, a : ndarray, ndarray

    """
    if band_width:
        phi = pi * band_width / sample_rate
        # alpha = 10**(0.1 * attenuation_half_bandwidth_db / order)
        # p = (-2 + 2 * alpha * cos(phi)) / (1 - alpha)
        # lambda_ = -p/2 - sqrt(p*p/4 - 1)

    elif band_width_factor:
        erb_audiological = band_width_factor * erb_aud(centerfrequency)
        phi = pi * erb_audiological / sample_rate
        # a_gamma = ((factorial(pi * (2*order - 2)) *
        #             2**(-(2*order - 2))) / (factorial(order - 1)**2))
        # b = erb_audiological / a_gamma
        # lambda_ = exp(-2 * pi * b / sample_rate)

    else:
        raise ValueError(
            'You need to specify either `band_width` or `band_width_factor!`')

    alpha = 10**(0.1 * attenuation_half_bandwidth_db / order)
    p = (-2 + 2 * alpha * cos(phi)) / (1 - alpha)
    lambda_ = -p/2 - sqrt(p*p/4 - 1)
    beta = 2*pi * centerfrequency / sample_rate
    coef = lambda_ * exp(1j*beta)
    factor = 2 * (1 - abs(coef))**order
    b, a = array([factor]), array([1., -coef])
    return b, a


def fosfilter(b, a, order, signal, states=None):
    """Return signal filtered with `b` and `a` (first order section)
    by filtering the signal `order` times.

    This Function was created for filtering signals by first order section
    cascaded complex gammatone filters.

    Parameters
    ----------
    b, a : ndarray, ndarray
        Filter coefficients of a first order section filter.
        Can be complex valued.
    order : int
        Order of the filter to be applied. This will
        be the count of refiltering the signal order times
        with the given coefficients.
    signal : ndarray
        Input signal to be filtered.
    states : ndarray, default None
        Array with the filter states of length `order`.
        Initial you can set it to None.

    Returns
    -------
    signal : ndarray
        Output signal, that is filtered and complex valued
        (analytical signal).
    states : ndarray
        Array with the filter states of length `order`.
        You need to loop it back into this function when block
        processing.

    """
    if not states:
        states = zeros(order, dtype=np.complex128)

    for i in range(order):
        state = [states[i]]
        signal, state = lfilter(b, a, signal, zi=state)
        states[i] = state[0]
        b = ones_like(b)
    return signal, states


def freqz_fos(b, a, order, nfft, plotfun=None):
    impulse = _create_impulse(nfft)
    response, states = fosfilter(b, a, order, impulse)
    freqresponse = rfft(np.real(response))
    frequencies = rfftfreq(nfft)
    if plotfun:
        plotfun(frequencies, freqresponse)
    return freqresponse, frequencies, response


def design_filtbank_coeffs(
        samplerate,
        order,
        centerfrequencies,
        bandwidths=None,
        bandwidth_factor=None,
        attenuation_half_bandwidth_db=-3):

    for i, cf in enumerate(centerfrequencies):
        if bandwidths:
            bw = bandwidths[i]
            bwf = None
        else:
            bw = None
            bwf = bandwidth_factor

        yield design_filter(
            samplerate, order, cf, band_width=bw,
            band_width_factor=bwf,
            attenuation_half_bandwidth_db=attenuation_half_bandwidth_db)


class GammatoneFilterbank:

    def __init__(
            self,
            samplerate=44100,
            order=4,
            startband=-12,
            endband=12,
            normfreq=1000.0,
            density=1.0,
            bandwidth_factor=1.0,
            desired_delay_sec=0.02):

        self.samplerate = samplerate
        self.order = order
        self.centerfrequencies = frequencies_gammatone_bank(
            startband, endband, normfreq, density)
        self._coeffs = tuple(design_filtbank_coeffs(
            samplerate,
            order,
            self.centerfrequencies,
            bandwidth_factor=bandwidth_factor))
        self.init_delay(desired_delay_sec)
        self.init_gains()

    def init_delay(self, desired_delay_sec):
        self.desired_delay_sec = desired_delay_sec
        self.desired_delay_samples = int(self.samplerate*desired_delay_sec)
        self.max_indices, self.slopes = self.estimate_max_indices_and_slopes(
            delay_samples=self.desired_delay_samples)
        self.delay_samples = self.desired_delay_samples - self.max_indices
        self.delay_memory = np.zeros((len(self.centerfrequencies),
                                      np.max(self.delay_samples)))

    def init_gains(self):
        self.gains = np.ones(len(self.centerfrequencies))
        # not correct until now:
        # x, s = list(zip(*self.analyze(_create_impulse(self.samplerate/10))))
        # rss = [np.sqrt(np.sum(np.real(b)**2)) for b in x]
        # self.gains = 1/np.array(rss)

    def analyze(self, signal, states=None):
        for i, (b, a) in enumerate(self._coeffs):
            st = None if not states else states[i]
            yield fosfilter(b, a, self.order, signal, states=st)

    def reanalyze(self, bands, states=None):
        for i, ((b, a), band) in enumerate(zip(self._coeffs, bands)):
            st = None if not states else states[i]
            yield fosfilter(b, a, self.order, band, states=st)

    def synthesize(self, bands):
        return np.array(list(self.delay(
            [b*g for b, g in zip(bands, self.gains)]))).sum(axis=0)

    def delay(self, bands):
        self.phase_factors = np.abs(self.slopes)*1j / self.slopes
        for i, band in enumerate(bands):
            phase_factor = self.phase_factors[i]
            delay_samples = self.delay_samples[i]
            if delay_samples == 0:
                yield np.real(band) * phase_factor
            else:
                yield np.concatenate(
                    (self.delay_memory[i, :delay_samples],
                     np.real(band[:-delay_samples])),
                    axis=0)
                self.delay_memory[i, :delay_samples] = np.real(
                    band[-delay_samples:])

    def estimate_max_indices_and_slopes(self, delay_samples=None):
        if not delay_samples:
            delay_samples = int(self.samplerate/10)
        sig = _create_impulse(delay_samples)
        bands = list(zip(*self.analyze(sig)))[0]
        ibandmax = [np.argmax(np.abs(b[:delay_samples])) for b in bands]
        slopes = [b[i+1]-b[i-1] for (b, i) in zip(bands, ibandmax)]
        return np.array(ibandmax), np.array(slopes)

    def freqz(self, nfft=4096, plotfun=None):
        def gen_freqz():
            for b, a in self._coeffs:
                yield freqz_fos(b, a, self.order, nfft, plotfun)
        return list(gen_freqz())


def _create_impulse(num_samples, arr = None):
    sig = zeros(num_samples) + 0j
    if not (np.any(arr)):
        sig[0] = 1.0
    else:
        sig = arr
    return sig


def example_filterbank():
    from pylab import plt
    import numpy as np

    x = _create_impulse(2000)
    gfb = GammatoneFilterbank(density=1)

    analyse = gfb.analyze(x)
    imax, slopes = gfb.estimate_max_indices_and_slopes()
    fig, axs = plt.subplots(len(gfb.centerfrequencies), 1)
    for (band, state), imx, ax in zip(analyse, imax, axs):
        ax.plot(np.real(band))
        ax.plot(np.imag(band))
        ax.plot(np.abs(band))
        ax.plot(imx, 0, 'o')
        ax.set_yticklabels([])
        [ax.set_xticklabels([]) for ax in axs[:-1]]

    axs[0].set_title('Impulse responses of gammatone bands')

    fig, ax = plt.subplots()

    def plotfun(x, y):
        ax.semilogx(x, 20*np.log10(np.abs(y)**2))
        plt.hold(True)

    gfb.freqz(nfft=2*4096, plotfun=plotfun)
    plt.grid(True)
    plt.title('Absolute spectra of gammatone bands.')
    plt.xlabel('Normalized Frequency (log)')
    plt.ylabel('Attenuation /dB(FS)')
    plt.axis('Tight')
    plt.ylim([-90, 1])
    plt.show()

    return gfb


def example_gammatone_filter():
    from pylab import plt, np
    sample_rate = 44100
    order = 4
    b, a = design_filter(
        sample_rate=sample_rate,
        order=order,
        centerfrequency=1000.0,
        attenuation_half_bandwidth_db=-3,
        band_width_factor=1.0)

    x = _create_impulse(1000)
    y, states = fosfilter(b, a, order, x)
    y = y[:800]
    plt.plot(np.real(y), label='Re(z)')
    plt.plot(np.imag(y), label='Im(z)')
    plt.plot(np.abs(y), label='|z|')
    plt.legend()
    plt.show()
    return y, b, a


if __name__ == '__main__':

    gfb = example_filterbank()
    y = example_gammatone_filter()

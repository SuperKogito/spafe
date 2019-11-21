import scipy
import warnings
import numpy as np
import matplotlib.pyplot as plt
from spafe.utils.converters import (hz2bark, bark2hz, hz2mel, mel2hz, fft2barkmx, fft2melmx)


NFFT = 512


def cqt(sig, fs=16000, low_freq=10, high_freq=3000, b=48):
    """
    Compute the constant Q-transform.

          - take the absolute value of the FFT
          - warp to a Mel frequency scale
          - take the DCT of the log-Mel-spectrum
          - return the first <num_ceps> components

    Args:
        sig     (array) : a mono audio signal (Nx1) from which to compute features.
        fs        (int) : the sampling frequency of the signal we are working with.
                          Default is 16000.
        low_freq  (int) : lowest band edge of mel filters (Hz).
                          Default is 10.
        high_freq (int) : highest band edge of mel filters (Hz).
                          Default is 3000.
        b         (int) : number of bins per octave.
                          Default is 48.
    Returns:
        array including the Q-transform coefficients.
    """

    # define lambda funcs for clarity
    def f(k):
        return low_freq * 2**((k - 1) / b)

    def w(N):
        return np.hamming(N)

    def nk(k):
        return np.ceil(Q * fs / f(k))

    def t(Nk, k):
        return (1 / Nk) * w(Nk) * np.exp(
            2 * np.pi * 1j * Q * np.arange(Nk) / Nk)

    # init vars
    Q = 1 / (2**(1 / b) - 1)
    K = int(np.ceil(b * np.log2(high_freq / low_freq)))
    nfft = int(2**np.ceil(np.log2(Q * fs / low_freq)))

    # define temporal kernal and sparse kernal variables
    S = [
        scipy.sparse.coo_matrix(np.fft.fft(t(nk(k), k), nfft))
        for k in range(K, 0, -1)
    ]
    S = scipy.sparse.vstack(S[::-1]).tocsc().transpose().conj() / nfft

    # compute the constant Q-transform
    xcq = (np.fft.fft(sig, nfft).reshape(1, nfft) * S)[0]
    return xcq


def pre_process_x(sig, fs=16000, win_type="hann", win_len=0.025, win_hop=0.01):
    """
    Prepare window and pad signal audio
    """
    # convert integer to double
    # sig = np.double(sig) / 2.**15

    # STFT parameters
    # convert win_len and win_hop from seconds to samples
    win_length = int(win_len * fs)
    hop_length = int(win_hop * fs)

    # compute window
    window = np.hanning(win_length)
    if win_type == "hamm":
        window = np.hamming(win_length)

    # normalization step to ensure that the STFT is self-inverting (or a Parseval frame)
    normalized_window = normalize_window(win=window, hop=hop_length)

    # Compute the STFT
    # zero pad to ensure that there are no partial overlap windows in the STFT computation
    sig = np.pad(sig, (window.size + hop_length, window.size + hop_length),
                 'constant',
                 constant_values=(0, 0))
    return sig, normalized_window, hop_length


def stft(sig, fs=16000, win_type="hann", win_len=0.025, win_hop=0.01):
    """
    Compute the short time Fourrier transform of an audio signal x.

    Args:
        x   (array) : audio signal in the time domain
        win   (int) : window to be used for the STFT
        hop   (int) : hop-size

    Returns:
        X : 2d array of the STFT coefficients of x
    """
    sig, normalized_window, hop_length = pre_process_x(sig,
                                                       fs=fs,
                                                       win_type=win_type,
                                                       win_len=win_len,
                                                       win_hop=win_hop)

    X = compute_stft(x=sig, win=normalized_window, hop=hop_length)
    return X, sig


def compute_stft(x, win, hop):
    """
    Compute the short time Fourrier transform of an audio signal x.

    Args:
        x   (array) : audio signal in the time domain
        win   (int) : window to be used for the STFT
        hop   (int) : hop-size

    Returns:
        X : 2d array of the STFT coefficients of x
    """
    # length of the audio signal
    sig_len = x.size

    # length of the window = fft size
    win_len = win.size

    # number of steps to take
    num_steps = (np.ceil((sig_len - win_len) / hop) + 1).astype(int)

    # init STFT coefficients
    X = np.zeros((win_len, num_steps), dtype=complex)

    # normalizing factor
    nf = np.sqrt(win_len)

    for k in range(num_steps - 1):
        d = x[k * hop:k * hop + win_len] * win
        X[:, k] = np.fft.fft(d) / nf

    # the last window may partially overlap with the signal
    d = x[num_steps * hop:]
    X[:, k] = np.fft.fft(d * win[:d.size], n=win_len) / nf
    return X


def istft(X, fs=16000, win_type="hann", win_len=0.025, win_hop=0.01):
    """
    Args:
        X : STFT coefficients
        win : window to be used for the STFT
        hop : hop-size

    Returns :
        x : inverse STFT of X
    """
    # STFT parameters
    # convert win_len and win_hop from seconds to samples
    win_length = int(win_len * fs)
    hop_length = int(win_hop * fs)

    # compute window
    if win_type == "hann":
        window = np.hanning(win_length)
    elif win_type == "hamm":
        window = np.hamming(win_length)

    # normalization step to ensure that the STFT is self-inverting (or a Parseval frame)
    normalized_window = normalize_window(win=window, hop=hop_length)

    # Compute the ISTFT
    # win_len: length of the window +   num_steps: number of frames
    win_len, num_steps = X.shape[0], X.shape[1]

    # length of the output signal
    sig_len = win_len + (num_steps - 1) * hop_length

    # init output variable
    x = np.zeros((sig_len), dtype=complex)

    # normalizing factor
    nf = np.sqrt(win_len)

    for k in range(num_steps):
        d = nf * np.fft.ifft(X[:, k]) * normalized_window
        x[k * hop_length:k * hop_length + win_len] += d
    return x


def normalize_window(win, hop):
    """
    Normalize the window according to the provided hop-size so that the STFT is
    a tight frame.

    Args:
        win   (int) : window to be used for the STFT
        hop   (int) : hop-size
    """
    N = win.size
    K = int(N / hop)
    win2 = win * win
    z = 1 * win2
    k = 1
    ind1 = N - hop
    ind2 = hop
    while (k < K):
        z[0:ind1] += win2[ind2:N]
        z[ind2:N] += win2[0:ind1]
        ind1 -= hop
        ind2 += hop
        k += 1
    win2 = win / np.sqrt(z)
    return win2


def display_stft(X,
                 fs,
                 len_sig,
                 low_freq=0,
                 high_freq=3000,
                 min_db=-10,
                 max_db=0,
                 normalize=True):
    """
    Plot the stft of an audio signal in the time-frequency plane.

    Args:
        X        (array) : STFT coefficients
        fs         (int) : sampling frequency in Hz (assumed to be integer)
        hop        (int) : hop-size used in the STFT (for labeling the time axis)
        low_freq   (int) : minimun frequency to plot in hz.
                           Default is 0 Hz.
        high_freq  (int) : maximum frequency tp plot in Hz.
                           Default is 3000 Hz.
        min_db     (int) : minimun magnitude to display in dB
                           Default is 0 dB.
        max_db     (int) : maximum magnitude to display in dB.
                           Default is -10 dB.
        normalize (bool) : Normalize input.
                           Default is True.
    """
    # normalize : largest coefficient magnitude is unity
    X_temp = X.copy()
    if normalize:
        X_temp /= np.amax(abs(X_temp))

    # compute frequencies array
    Freqs = np.array([low_freq, high_freq])
    Fd = (Freqs * X_temp.shape[0] / fs).astype(int)

    # compute values matrix
    Z = X_temp[Fd[1]:Fd[0]:-1, :]
    Z = np.clip(np.log(np.abs(Z) + 1e-50), min_db, max_db)
    Z = 255 * (Z - min_db) / (max_db - min_db)

    # compute duration
    time = float(len_sig) / float(fs)

    # plotting
    plt.imshow(Z,
               extent=[0, time, low_freq / 1000, high_freq / 1000],
               aspect="auto")
    plt.ylabel('Frequency (Khz)')
    plt.xlabel('Time (sec)')
    plt.show()
    

def power_spectrum(fourrier_transform, nfft=NFFT):
    magnitude_frames = np.absolute(fourrier_transform)  # Magnitude of the FFT
    power_frames = ((1.0 / nfft) * ((magnitude_frames)**2))  # Power Spectrum
    return power_frames


def rfft(x, n=NFFT):
    """
    compute the fourrier transform of a certain signal frames.
    """
    return np.fft.rfft(a=x, n=n)


def dct(x, type=2, axis=1, norm='ortho'):
    from scipy.fftpack import dct
    return scipy.fftpack.dct(x=x, type=type, axis=axis, norm=norm)


def powspec(sig,
            fs=16000,
            nfft=512,
            win_type="hann",
            win_len=0.025,
            win_hop=0.01,
            dither=1):
    """
     compute the powerspectrum and frame energy of the input signal.
     basically outputs a power spectrogram

     each column represents a power spectrum for a given frame
     each row represents a frequency

     default values:
         fs = 8000Hz
         wintime = 25ms (200 samps)
         steptime = 10ms (80 samps)
         which means use 256 point fft
         hamming window

     $Header: /Users/dpwe/matlab/rastamat/RCS/powspec.m,v 1.3 2012/09/03 14:02:01 dpwe Exp dpwe $

     for fs = 8000
         NFFT = 256;
         NOVERLAP = 120;
         SAMPRATE = 8000;
         WINDOW = hamming(200);
    """
    # convert win_len and win_hop from seconds to samples
    win_length = int(win_len * fs)
    hop_length = int(win_hop * fs)
    fft_length = int(np.power(2, np.ceil(np.log2(win_len * fs))))

    # compute stft
    X, _ = stft(sig=sig,
                fs=fs,
                win_type=win_type,
                win_len=win_len,
                win_hop=win_hop)

    pow_X = np.abs(X)**2
    if dither:
        pow_X = pow_X + win_length

    e = np.log(np.sum(pow_X, axis=0))
    return pow_X, e


def lifter(x, lift=0.6, invs=False):
    """
    apply lifter to matrix of cepstra (one per column)

    Args:
      lift   (float) : exponent of x i^n liftering or, as a negative integer, the
                       length of HTK-style sin-curve liftering.
      inverse (bool) : if inverse == 1 (default 0), undo the liftering.

    Returns:
        liftered cepstra.
    """
    ncep = x.shape[0]

    if lift == 0:
        y = x
    else:
        if lift < 0:
            warnings.warn(
                'HTK liftering does not support yet; default liftering')
            lift = 0.6
        liftwts = np.arange(1, ncep)**lift
        liftwts = np.append(1, liftwts)

        if (invs):
            liftwts = 1 / liftwts

        y = np.matmul(np.diag(liftwts), x)

    return y


def audspec(p_spectrum,
            fs=16000,
            nfilts=0,
            fb_type='bark',
            low_freq=0,
            high_freq=0,
            sumpower=1,
            bwidth=1):
    """
    perform critical band analysis (see PLP) based on the power spectrogram.

    Args:
        aspectrum (array) : the power spectrum array.
        nfft        (int) : the FFT size.
                            (Default is 512)
        fs          (int) : sample rate/ sampling frequency of the signal.
                            (Default 16000 Hz)
        nfilts      (int) : the number of filters in the filterbank.
                            (Default 20)
        fb_type     (str) : type of bins [Mel, Bark, ...].
        bwidth      (int) : the constant width of each band relative to standard Mel (default 1).
                            Default is 1.
        low_freq    (int) : lowest band edge of mel filters.
                            (Default 0 Hz)
        high_freq   (int) : highest band edge of mel filters.
                            (Default samplerate/2)
        sumpower   (bool) : sum power if True.
                            Default is True.

    Returns:
        auditory spectrum array.
    """
    if nfilts == 0:
        np.ceil(hz2bark(fs / 2)) + 1
    if high_freq == 0:
        high_freq = fs / 2
    nfreqs = p_spectrum.shape[0]
    nfft = (int(nfreqs) - 1) * 2

    if fb_type == 'bark':
        wts = fft2barkmx(nfft, fs, nfilts, bwidth, low_freq, high_freq)
    elif fb_type == 'mel':
        wts = fft2melmx(nfft, fs, nfilts, bwidth, low_freq, high_freq)
    elif fb_type == 'htkmel':
        wts = fft2melmx(nfft,
                        fs,
                        nfilts,
                        bwidth,
                        low_freq,
                        high_freq,
                        htk=True,
                        constamp=True)
    elif fb_type == 'fcmel':
        wts = fft2melmx(nfft,
                        fs,
                        nfilts,
                        bwidth,
                        low_freq,
                        high_freq,
                        htk=True,
                        constamp=False)

    wts = wts[:, 0:nfreqs]

    if sumpower:
        aspectrum = np.matmul(wts, p_spectrum)
    else:
        aspectrum = np.matmul(wts, np.sqrt(p_spectrum))**2
    return aspectrum


def postaud(x, fmax, fb_type='bark', broaden=0):
    """
    do loudness equalization and cube root compression
        - x = critical band filters
        - rows = critical bands
        - cols = frames
    """
    nbands, nframes = x.shape
    nfpts = int(nbands + 2 * broaden)

    if fb_type == 'bark':
        bandcfhz = bark2hz(np.linspace(0, hz2bark(fmax), nfpts))
    elif fb_type == 'mel':
        bandcfhz = mel2hz(np.linspace(0, hz2mel(fmax), nfpts))
    elif fb_type == 'htkmel' or fb_type == 'fcmel':
        bandcfhz = mel2hz(np.linspace(0, hz2mel(fmax, htk=True), nfpts),
                          htk=True)

    bandcfhz = bandcfhz[broaden:(nfpts - broaden)]

    fsq = np.power(bandcfhz, 2)
    ftmp = np.add(fsq, 1.6e5)
    eql = np.multiply((fsq / ftmp)**2, (fsq + 1.44e6) / (fsq + 9.61e6))

    z = np.multiply(np.tile(eql, (nframes, 1)).T, x)
    z = np.power(z, 0.33)

    if broaden:
        y = np.zeros((z.shape[0] + 2, z.shape[1]))
        y[0, :] = z[0, :]
        y[1:nbands + 1, :] = z
        y[nbands + 1, :] = z[z.shape[0] - 1, :]
    else:
        y = np.zeros((z.shape[0], z.shape[1]))
        y[0, :] = z[1, :]
        y[1:nbands - 1, :] = z[1:z.shape[0] - 1, :]
        y[nbands - 1, :] = z[z.shape[0] - 2, :]

    return y, eql


def invpostaud(y, fmax, fb_type='bark', broaden=0):
    """
    invert the effects of postaud (loudness equalization and cube
        - root compression)
        - y = postaud output
        - x = reconstructed critical band filters
        - rows = critical bands
        - cols = frames
    """
    nbands, nframes = y.shape

    if fb_type == 'bark':
        bandcfhz = bark2hz(np.linspace(0, hz2bark(fmax), nbands))
    elif fb_type == 'mel':
        bandcfhz = mel2hz(np.linspace(0, hz2mel(fmax), nbands))
    elif fb_type == 'htkmel' or fb_type == 'fcmel':
        bandcfhz = mel2hz(np.linspace(0, hz2mel(fmax, htk=True), nbands),
                          htk=True)

    bandcfhz = bandcfhz[broaden:(nbands - broaden)]

    fsq = bandcfhz**2
    ftmp = fsq + 1.6e5
    eql = np.multiply((fsq / ftmp)**2, (fsq + 1.44e6) / (fsq + 9.61e6))

    x = y**(1 / 0.33)

    if eql[0] == 0:
        eql[0] = eql[1]
        eql[-1] = eql[-2]

    x = np.divide(x[broaden:(nbands - broaden + 1), :],
                  np.add(np.tile(eql.T, (nframes, 1)).T, 1e-8))

    return x, eql


def invpowspec(y, fs, win_len, win_hop, excit=[]):
    """
    x = invpowspec(y, fs, wintime, steptime, excit)

    Attempt to go back from specgram-like power spectrum to audio waveform by
    scaling specgram of white noise

    default values:
        fs = 8000Hz
        wintime = 25ms (200 samps)
        steptime = 10ms (80 samps)
        which means use 256 point fft
        hamming window

    excit is input excitation; white noise is used if not specified
        for fs = 8000
        NFFT = 256;
        NOVERLAP = 120;
        SAMPRATE = 8000;
        WINDOW = hamming(200);
    """
    nrow, ncol = y.shape
    r = excit

    winpts = int(win_len * fs)
    steppts = int(win_hop * fs)
    nfft = int(np.power(2, np.ceil(np.divide(np.log(winpts), np.log(2)))))

    # Can't predict librosa stft length...
    tmp = istft(X=y, fs=fs, win_type="hann", win_len=0.025, win_hop=0.01)

    # # Can't predict librosa stft length...
    # tmp = librosa.istft(y,
    #                     hop_length=steppts,
    #                     win_length=winpts,
    #                     window='hann',
    #                     center=False)
    xlen = len(tmp)
    # xlen = int(np.add(winpts, np.multiply(steppts, np.subtract(ncol, 1))))
    # xlen = int(np.multiply(steppts, np.subtract(ncol, 1)))

    if len(r) == 0:
        r = np.squeeze(np.random.randn(xlen, 1))
    r = r[0:xlen]

    R, _ = stft(sig=r,
                fs=fs,
                win_type=win_type,
                win_len=win_len,
                win_hop=win_hop)
    # R = librosa.stft(np.divide(r, 32768 * 12),
    #                  n_fft=nfft,
    #                  hop_length=steppts,
    #                  win_length=winpts,
    #                  window='hann',
    #                  center=False)

    R *= np.sqrt(y)
    x = istft(X=R, fs=fs, win_type="hann", win_len=0.025, win_hop=0.01)
    # x = librosa.istft(R,
    #                   hop_length=steppts,
    #                   win_length=winpts,
    #                   window='hann',
    #                   center=False)

    return x


def invaudspec(aspectrum,
               fs=16000,
               nfft=512,
               fb_type='bark',
               low_freq=0,
               high_freq=None,
               sumpower=True,
               bwidth=1):
    """
    Compute the power spectrum from the auditory spectrum.
    Invert (~might not be that accurate) the effects of audspec()

    Args:
        aspectrum (array) : the auditory spectrum array.
        nfft        (int) : the FFT size.
                            (Default is 512)
        fs          (int) : sample rate/ sampling frequency of the signal.
                            (Default 16000 Hz)
        nfilts      (int) : the number of filters in the filterbank.
                            (Default 20)
        fb_type     (str) : type of bins [Mel, Bark, ...].
        bwidth      (int) : the constant width of each band relative to standard Mel (default 1).
                            Default is 1.
        low_freq    (int) : lowest band edge of mel filters.
                            (Default 0 Hz)
        high_freq   (int) : highest band edge of mel filters.
                            (Default samplerate/2)
        sumpower   (bool) : sum power if True.
                            Default is True.

    Returns:
        power spectrum array.
    """
    if high_freq is None:
        high_freq = fs / 2
    nfilts, nframes = aspectrum.shape

    if fb_type == 'bark':
        wts = fft2barkmx(nfft, fs, nfilts, bwidth, low_freq, high_freq)
    elif fb_type == 'mel':
        wts = fft2melmx(nfft, fs, nfilts, bwidth, low_freq, high_freq)
    elif fb_type == 'htkmel':
        wts = fft2melmx(nfft,
                        fs,
                        nfilts,
                        bwidth,
                        low_freq,
                        high_freq,
                        htk=True,
                        constamp=True)
    elif fb_type == 'fcmel':
        wts = fft2melmx(nfft,
                        fs,
                        nfilts,
                        bwidth,
                        low_freq,
                        high_freq,
                        htk=True,
                        constamp=False)

    wts = wts[:, 0:int(nfft / 2 + 1)]
    ww = np.matmul(wts.T, wts)
    itws = wts.T / np.tile(
        np.maximum(np.mean(np.diag(ww)) / 100, np.sum(ww, axis=0)),
        (nfilts, 1)).T

    if sumpower:
        spec = np.matmul(itws, aspectrum)
    else:
        spec = np.power(np.matmul(itws, np.sqrt(aspectrum)))

    return spec, wts, itws

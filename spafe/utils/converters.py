import numpy as np
from ..utils.exceptions import ParameterError, ErrorMsgs

# init vars
F0 = 0
FSP = 200 / 3
BARK_FREQ = 1000
BARK_PT = (BARK_FREQ - F0) / FSP
LOGSTEP = np.exp(np.log(6.4) / 27.0)

def hz2erb(f):
    """
    Convert Hz frequencies to Bark.

    Args:
        f (np.array) : input frequencies [Hz].

    Returns:
        (np.array): frequencies in Bark [Bark].
    """
    return 24.7 * (4.37 * (f / 1000) + 1)

def erb2hz(fe):
    """
    Convert Bark frequencies to Hz.

    Args:
        fb (np.array) : input frequencies [Bark].

    Returns:
        (np.array)  : frequencies in Hz [Hz].
    """
    return ((fe / 24.7) - 1) * (1000. / 4.37)

def fft2erb(fft, fs=16000, nfft=512):
    """
    Convert Bark frequencies to Hz.

    Args:
        fft (np.array) : fft bin numbers.

    Returns:
        (np.array): frequencies in Bark [Bark].
    """
    return hz2erb((fft * fs) / (nfft + 1))

def erb2fft(fb, fs=16000, nfft=512):
    """
    Convert Bark frequencies to fft bins.

    Args:
        fb (np.array): frequencies in Bark [Bark].

    Returns:
        (np.array) : fft bin numbers.
    """
    return (nfft + 1) * erb2hz(fb) / fs

def hz2bark(f):
    """
    Convert Hz frequencies to Bark acoording to Wang, Sekey & Gersho, 1992.

    Args:
        f (np.array) : input frequencies [Hz].

    Returns:
        (np.array): frequencies in Bark [Bark].
    """
    return 6. * np.arcsinh(f / 600.)

def bark2hz(fb):
    """
    Convert Bark frequencies to Hz.

    Args:
        fb (np.array) : input frequencies [Bark].

    Returns:
        (np.array)  : frequencies in Hz [Hz].
    """
    return 600. * np.sinh(fb / 6.)

def fft2hz(fft, fs=16000, nfft=512):
    """
    Convert Bark frequencies to Hz.

    Args:
        fft (np.array) : fft bin numbers.

    Returns:
        (np.array): frequencies in Bark [Bark].
    """
    return (fft * fs) / (nfft + 1)

def hz2fft(fb, fs=16000, nfft=512):
    """
    Convert Bark frequencies to fft bins.

    Args:
        fb (np.array): frequencies in Bark [Bark].

    Returns:
        (np.array) : fft bin numbers.
    """
    return (nfft + 1) * fb / fs


def fft2bark(fft, fs=16000, nfft=512):
    """
    Convert Bark frequencies to Hz.

    Args:
        fft (np.array) : fft bin numbers.

    Returns:
        (np.array): frequencies in Bark [Bark].
    """
    return hz2bark((fft * fs) / (nfft + 1))

def bark2fft(fb, fs=16000, nfft=512):
    """
    Convert Bark frequencies to fft bins.

    Args:
        fb (np.array): frequencies in Bark [Bark].

    Returns:
        (np.array) : fft bin numbers.
    """
    return (nfft + 1) * bark2hz(fb) / fs


def hz2mel(hz, htk=1):
    """
    Convert a value in Hertz to Mels

    Args:
         hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
         htk: Optional variable, if htk = 1 uses the mel axis defined in the HTKBook otherwise use Slaney's formula.

    Returns:
        a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    if htk == 1:
        return 2595 * np.log10(1 + hz / 700.)

    else:
        # format variable
        hz = np.array(hz, ndmin=1)

        # definee lambda functions to simplify code
        def e(i):
            return (hz[i] - F0) / FSP

        def g(i):
            return BARK_PT + (np.log(hz[i] / BARK_FREQ) / np.log(LOGSTEP))

        mel = [e(i) if hz[i] < BARK_PT else g(i) for i in range(hz.shape[0])]
        return np.array(mel)


def mel2hz(mel, htk=1):
    """
    Convert a value in Mels to Hertz

    Args:
        mel: a value in Mels. This can also be a numpy array, conversion
             proceeds element-wise.
        htk: Optional variable, if htk = 1 uses the mel axis defined in the
             HTKBook otherwise use Slaney's formula.

    Returns:
        a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    if htk == 1:
        return 700 * (10**(mel / 2595.0) - 1)
    else:
        # format variable
        mel = np.array(mel, ndmin=1)

        # definee lambda functions to simplify code
        def e(i):
            return F0 + FSP * mel[i]

        def g(i):
            return BARK_FREQ * np.exp(np.log(LOGSTEP) * (mel[i] - BARK_PT))

        f = [e(i) if mel[i] < BARK_PT else g(i) for i in range(mel.shape[0])]
        return np.array(f)


def fft2melmx(nfft,
              fs,
              nfilts=0,
              bwidth=1,
              low_freq=0,
              high_freq=0,
              htk=False,
              constamp=False):
    """
    Generate a matrix of weights to combine FFT bins into Mel bins.

    Args:
        nfft      (int) : the FFT size.
                          (Default is 512)
        fs        (int) : sample rate/ sampling frequency of the signal.
                          (Default 16000 Hz)
        nfilts    (int) : the number of filters in the filterbank.
                          (Default 20)
        bwidth    (int) : the constant width of each band relative to standard Mel (default 1).
                          Default is 1.
        low_freq  (int) : lowest band edge of mel filters.
                          (Default 0 Hz)
        high_freq (int) : highest band edge of mel filters.
                          (Default samplerate/2)
        htkmel   (bool) : use HTK's version of the mel curve, not Slaney's.
                          Default is False.
        constamp (bool) : if True then make integration windows peak at 1, not sum to 1.
                          Default is False.

    Notes
        `low_freq` default is 0, but 133.33 is a common standard (to skip low frequencies).
        `high_freq` default is fs/2
        You can exactly duplicate the mel matrix in Slaney'ss using
        `fft2melmx(nfft=512, fs=8000, nfilts=40, bwidth=1, low_freq=133.33, high_freq=6855.5, 0)`

    Returns:
        matrix of weights to combine FFT bins into Mel bins.

    """
    if high_freq == 0:
        high_freq = fs / 2

    if nfilts == 0:
        nfilts = int(np.ceil(hz2mel(high_freq, htk) / 2))

    if isinstance(nfilts, int) == int:
        raise ParameterError(ErrorMsgs["nfilts"])

    if isinstance(nfft, int) == int:
        raise ParameterError(ErrorMsgs["nfft"])

    wts = np.zeros((nfilts, nfft))
    fftfrqs = (fs / nfft) * np.arange(0, nfft / 2 + 1)

    min_mel = hz2mel(low_freq, htk)
    max_mel = hz2mel(high_freq, htk)
    dif_mel = max_mel - min_mel
    binfrqs = mel2hz(
        min_mel + np.arange(0, nfilts + 2) * dif_mel / (nfilts + 1), htk)

    for i in range(nfilts):
        fs_tmp = binfrqs[np.arange(0, 3) + i]
        fs_tmp = fs_tmp[1] + bwidth * (fs_tmp - fs_tmp[1])
        # slopes
        loslope = (fftfrqs - fs_tmp[0]) / (fs_tmp[1] - fs_tmp[0])
        hislope = (fs_tmp[2] - fftfrqs) / (fs_tmp[2] - fs_tmp[1])
        wts[i, 0:nfft // 2 + 1] = np.maximum(0, np.minimum(loslope, hislope))

    if not constamp:
        wts = np.matmul(
            np.diag(2 / (binfrqs[2:nfilts + 2] - binfrqs[0:nfilts])), wts)

    return wts


def fft2barkmx(nfft, fs, nfilts=0, bwidth=1, low_freq=0, high_freq=0):
    """
    Generate a matrix of weights to combine FFT bins into Bark bins.

    Args:
        nfft      (int) : the FFT size.
                          (Default is 512)
        fs        (int) : sample rate/ sampling frequency of the signal.
                          (Default 16000 Hz)
        nfilts    (int) : the number of filters in the filterbank.
                          (Default 20)
        bwidth    (int) : the constant width of each band relative to standard Mel (default 1).
                          Default is 1.
        low_freq  (int) : lowest band edge of mel filters.
                          (Default 0 Hz)
        high_freq (int) : highest band edge of mel filters.
                          (Default sample rate/2)

    Notes:
        Optional nfilts specifies the number of output bands required
        (else one per bark), and width is the constant width of each
        band in Bark (default 1).

    Returns:
        matrix of weights to combine FFT bins into Bark bins.
    """
    if high_freq == 0:
        high_freq = fs / 2

    min_bark = hz2bark(low_freq)
    nyqbark = hz2bark(high_freq) - min_bark

    if nfilts == 0:
        nfilts = int(np.add(np.ceil(nyqbark), 1))

    if isinstance(nfilts, int) == int:
        raise ParameterError(ErrorMsgs["nfilts"])

    if isinstance(nfft, int) == int:
        raise ParameterError(ErrorMsgs["nfft"])

    wts = np.zeros((nfilts, nfft))
    step_barks = nyqbark / (nfilts - 1)
    binbarks = hz2bark((fs / nfft) * np.arange(0, nfft / 2 + 1))

    for i in range(nfilts):
        f_bark_mid = min_bark + i * step_barks
        lof = binbarks - f_bark_mid - 0.5
        hif = binbarks - f_bark_mid + 0.5
        wts[i, 0:nfft // 2 + 1] = 10**np.minimum(
            0,
            np.minimum(hif, -2.5 * lof) / bwidth)
    return wts

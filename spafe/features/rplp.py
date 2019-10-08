"""
based on:
    https://github.com/ZhihaoDU/speech_feature_extractor/blob/master/rasta_plp_extractor.py
"""
import numpy as np
from scipy.io import wavfile
from matplotlib import pylab
import matplotlib.pyplot as plt
from scikits.talkbox import lpc
from scipy.signal import lfilter, lfilter_zi, lfiltic
from feature_extractor import log_power_spectrum_extractor


def freq2bark(f):
    return 7.*np.log(f/650.+np.sqrt(np.power(1.+(f/650.), 2.)))


def bark2freq(b):
    return 650.*np.sinh(b/7.)


def get_fft_bark_mat(sr, fft_len, barks, min_frq=20, max_frq=None):
    if max_frq is None:
        max_frq = sr // 2
    fft_frqs  = np.arange(0, fft_len//2+1) / (1.*fft_len) * sr
    min_bark  = freq2bark(min_frq)
    max_bark  = freq2bark(max_frq)
    bark_bins = bark2freq(min_bark + np.arange(0, barks+2) / (barks + 1.) * (max_bark - min_bark))
    wts       = np.zeros((barks, fft_len//2+1))
    for i in range(barks):
        fs        = bark_bins[[i+0, i+1, i+2]]
        loslope   = (fft_frqs - fs[0]) / (fs[1] - fs[0])
        hislope   = (fs[2] - fft_frqs) / (fs[2] - fs[1])
        wts[i, :] = np.maximum(0, np.minimum(loslope, hislope))
    return wts


def rasta_filt(x):
    number = np.arange(-2., 3., 1.)
    number = -1. * number / np.sum(number*number)
    denom  = np.array([1., -0.94])
    zi     = lfilter_zi(number, 1)
    zi     = zi.reshape(1, len(zi))
    zi     = np.repeat(zi, np.size(x, 0), 0)
    y, zf  = lfilter(number, 1, x[:,0:4], axis=1, zi=zi)
    y, zf  = lfilter(number, denom, x, axis=1, zi=zf)
    return y


def postaud(x, fmax, fbtype=None):
    if fbtype is None:
        fbtype = 'bark'

    nbands  = x.shape[0]
    nframes = x.shape[1]
    nfpts   = nbands

    if fbtype == 'bark':
        bancfhz = bark2freq(np.linspace(0, freq2bark(fmax), nfpts))

    fsq  = bancfhz * bancfhz
    ftmp = fsq + 1.6e5
    eql  = ((fsq/ftmp)**2) * ((fsq + 1.44e6)/(fsq + 9.61e6))
    '''
    plt.figure()
    plt.plot(eql)
    plt.show()
    '''
    eql = eql.reshape(np.size(eql), 1)
    z   = np.repeat(eql, nframes, axis=1) * x
    z   = z ** (1./3.)
    y   = np.vstack((z[1, :], z[1:nbands-1, :], z[nbands-2, :]))
    return y


def do_lpc(spec, order, error_normal=False):
    coeff, error, k = lpc(spec, order, axis=0)
    if error_normal:
        error = np.reshape(error, (1, len(error)))
        error = np.repeat(error, order+1, axis=0)
        return coeff / error
    else:
        return coeff[1:, :]


def get_dct_coeff(in_channel, out_channel):
    dct_coef = np.zeros((out_channel, in_channel), dtype=np.float32)
    for i in range(out_channel):
        n = np.linspace(0, in_channel - 1, in_channel)
        dct_coef[i, :] = np.cos((2 * n + 1) * i * np.pi / (2 * in_channel))
    return dct_coef

# I cannot understand it, maybe it works...
def lpc2cep(a, nout=None):
    nin   = np.size(a, 0)
    ncol  = np.size(a, 1)
    order = nin - 1
    if nout is None:
        nout = order + 1
    c = np.zeros((nout, ncol))
    c[0, :] = -1. * np.log(a[0, :])
    renormal_coef = np.reshape(a[0,:], (1, ncol))
    renormal_coef = np.repeat(renormal_coef, nin, axis=0)
    a = a / renormal_coef
    for n in range(1, nout):
        sumn = np.zeros(ncol)
        for m in range(1, n+1):
            sumn = sumn + (n-m) * a[m, :] * c[n-m, :]
        c[n, :] = -1. * (a[n, :] + 1. / n * sumn)
    return c


def rasta_plp_extractor(x, sr, plp_order=0, do_rasta=True):
    spec         = log_power_spectrum_extractor(x, int(sr*0.02), int(sr*0.01), 'hamming', False)
    bark_filters = int(np.ceil(freq2bark(sr//2)))
    wts          = get_fft_bark_mat(sr, int(sr*0.02), bark_filters)
    '''
    plt.figure()
    plt.subplot(211)
    plt.imshow(wts)
    plt.subplot(212)
    plt.hold(True)
    for i in range(18):
        plt.plot(wts[i, :])
    plt.show()
    '''
    bark_spec = np.matmul(wts, spec)
    if do_rasta:
        bark_spec     = np.where(bark_spec == 0.0, np.finfo(float).eps, bark_spec)
        log_bark_spec = np.log(bark_spec)
        rasta_log_bark_spec = rasta_filt(log_bark_spec)
        bark_spec           = np.exp(rasta_log_bark_spec)
    post_spec = postaud(bark_spec, sr/2.)

    if plp_order > 0:
        lpcas = do_lpc(post_spec, plp_order)
        # lpcas = do_lpc(spec, plp_order) # just for test
    else:
        lpcas = post_spec
    return lpcas

if __name__ == '__main__':
    sr, wav_data = wavfile.read("../test.wav")
    lpcas = rasta_plp_extractor(wav_data, sr, 16, True)
    pylab.figure()
    #pylab.subplot(211)
    pylab.imshow(lpcas)
    pylab.show()

import numpy as np
from spafe.utils import levinsondr
from scipy.fftpack import fft, ifft
from spafe.utils.filters import rasta_filter
from spafe.utils.cepstral import cms, cmvn, lifter_ceps
from ..utils.preprocessing import pre_emphasis, framing, windowing, zero_handling
from spafe.utils.spectral import (powspec, lifter, audspec, postaud,
                                  invpostaud, invpowspec, invaudspec)


def lpcc(sig,
         fs=16000,
         num_ceps=13,
         pre_emph=1,
         pre_emph_coeff=0.97,
         win_type="hann",
         win_len=0.025,
         win_hop=0.01,
         do_rasta=True,
         lifter=1,
         normalize=1,
         dither=1):
    """
    Compute the LINEAR PREDICTIVE CEPSTRAL COEFFICIENTS (LPCC) from an audio signal.

    Args:
        sig            (array) : a mono audio signal (Nx1) from which to compute features.
        fs               (int) : the sampling frequency of the signal we are working with.
                                 Default is 16000.
        num_ceps       (float) : number of cepstra to return (order of the model to compute).
                                 Default is 13.
        pre_emph         (int) : apply pre-emphasis if 1.
                                 Default is 1.
        pre_emph_coeff (float) : apply pre-emphasis filter [1 -pre_emph] (0 = none).
                                 Default is 0.97.
        win_type       (float) : window type to apply for the windowing.
                                 Default is hanning.
        win_len        (float) : window length in sec.
                                 Default is 0.025.
        win_hop        (float) : step between successive windows in sec.
                                 Default is 0.01.
        do_rasta         (int) : if 1 then apply rasta filtering.
                                 Default is 0.
        lifter           (int) : apply liftering if value > 0.
                                 Default is 22.
        normalize        (int) : apply normalization if 1.
                                 Default is 0.
        dither           (int) : 1 = add offset to spectrum as if dither noise.
                                 Default is 0.
    Returns:
        (array) : 2d array of LPCC features (num_frames x num_ceps)
    """
    lpcs = lpc(sig=sig,
               fs=fs,
               num_ceps=num_ceps,
               pre_emph=pre_emph,
               pre_emph_coeff=pre_emph_coeff,
               win_len=win_len,
               win_hop=win_hop,
               do_rasta=True,
               dither=dither)
    lpccs = lpc2cep(lpcs.T)

    # liftering
    if lifter > 0:
        lpccs = lifter_ceps(lpccs, lifter)

    # normalization
    if normalize:
        lpccs = cmvn(cms(lpccs))

    lpccs = lpccs.T
    return lpccs[:, :]


def lpc(sig,
        fs=16000,
        num_ceps=13,
        pre_emph=0,
        pre_emph_coeff=0.97,
        win_type="hann",
        win_len=0.025,
        win_hop=0.01,
        do_rasta=True,
        dither=1):
    """
    Compute the LINEAR PREDICTIVE COEFFICIENTS (LPC) from an audio signal.

    Args:
        sig            (array) : a mono audio signal (Nx1) from which to compute features.
        fs               (int) : the sampling frequency of the signal we are working with.
                                 Default is 16000.
        num_ceps       (int) : number of cepstra to return(order of the model to compute).
                                 Default is 13.
        pre_emph         (int) : apply pre-emphasis if 1.
                                 Default is 1.
        pre_emph_coeff (float) : apply pre-emphasis filter [1 -pre_emph] (0 = none).
                                 Default is 0.97.
        win_type       (float) : window type to apply for the windowing.
                                 Default is hanning.
        win_len        (float) : window length in sec.
                                 Default is 0.025.
        win_hop        (float) : step between successive windows in sec.
                                 Default is 0.01.
        do_rasta         (int) : if 1 then apply rasta filtering.
                                 Default is 0.
        lifter           (int) : apply liftering if value > 0.
                                 Default is 22.
        normalize        (int) : apply normalization if 1.
                                 Default is 0.
        dither           (int) : 1 = add offset to spectrum as if dither noise.
                                 Default is 0.
    Returns:
        (array) : 2d array of LPC features (num_frames x num_ceps)
    """
    # pre-emphasis
    if pre_emph:
        sig = pre_emphasis(sig=sig, pre_emph_coeff=pre_emph_coeff)

    # compute power spectrum
    power_spectrum, _ = powspec(sig=sig,
                                fs=fs,
                                win_type=win_type,
                                win_len=win_len,
                                win_hop=win_hop,
                                dither=dither)

    # group to critical bands
    auditory_spectrum = audspec(power_spectrum, fs)
    nbands = auditory_spectrum.shape[0]

    if do_rasta:
        # put in log domain
        log_auditory_spectrum = np.log(auditory_spectrum)
        # next do rasta filtering
        rasta_filtered_log_auditory_spectrum = rasta_filter(
            log_auditory_spectrum)
        # do inverse log
        auditory_spectrum = np.exp(rasta_filtered_log_auditory_spectrum)

    post_processing_spectrum, _ = postaud(auditory_spectrum, fs / 2)
    lpcs = do_lpc(x=post_processing_spectrum, model_order=num_ceps)
    lpcs = lpcs.T
    return lpcs[:, :num_ceps]


def do_lpc(x, model_order=8):
    """
    Compute the autoregressive model from spectral magnitude samples.

    Args:
        x         (array) : array of the audio signal to process.
        model_order (int) : order of the model to compute.

    Returns:
        array of the autoregressive model
    """
    nbands, nframes = x.shape
    ncorr = 2 * (nbands - 1)
    R = np.zeros((ncorr, nframes))

    R[0:nbands, :] = x
    for i in range(nbands - 1):
        R[i + nbands - 1, :] = x[nbands - (i + 1), :]

    r = ifft(R.T).real.T
    r = r[0:nbands, :]

    y = np.ones((nframes, model_order + 1))
    e = np.zeros((nframes, 1))

    if model_order == 0:
        for i in range(nframes):
            _, e_tmp, _ = levinsondr.LEVINSON(r[:, i],
                                              model_order,
                                              allow_singularity=True)
            e[i, 0] = e_tmp
    else:
        for i in range(nframes):
            y_tmp, e_tmp, _ = levinsondr.LEVINSON(r[:, i],
                                                  model_order,
                                                  allow_singularity=True)
            y[i, 1:model_order + 1] = y_tmp
            e[i, 0] = e_tmp

    y = y.T / (np.tile(e.T, (model_order + 1, 1)) + 1e-8)

    return y


def lpc2cep(a, nout=0):
    """
    convert LPC coefficients directly to cepstral values.
     - convert the LPC 'a' coefficients in each column of lpcs into frames of cepstra.

    Args:
        a  (array) : cepstral values.
        nout (int) : number of cepstra to produce

    Returns:
        array of LPC coefficients.
        Default size(lpcs, 1)
    """
    nin, ncol = a.shape

    order = nin - 1

    if nout == 0:
        nout = order + 1

    cep = np.zeros((nout, ncol))
    cep[0, :] = -np.log(a[0, :])

    norm_a = np.divide(a, np.add(np.tile(a[0, :], (nin, 1)), 1e-8))

    for n in range(1, nout):
        sum = 0
        for m in range(1, n):
            sum = np.add(
                sum,
                np.multiply(np.multiply((n - m), norm_a[m, :]),
                            cep[(n - m), :]))

        cep[n, :] = -np.add(norm_a[n, :], np.divide(sum, n))

    return cep


def lpc2spec(lpcs, nout=17, FMout=False):
    """
    convert LPC coefficients back into spectra by sampling the z-plane.

    Args:
        lpcs (array) : array including the LPC coefficients.
        nout   (int) : number of freq channels, default 17 (i.e. for 8 kHz)
        FMout (bool) :

    Returns:
        list including the features, F and M
    """
    rows, cols = lpcs.shape
    order = rows - 1

    gg = lpcs[0, :]
    aa = lpcs / np.tile(gg, (rows, 1))

    # Calculate the actual z-plane polyvals: nout points around unit circle
    tmp_1 = np.array(np.arange(0, nout), ndmin=2).T
    tmp_1 = (-1j * tmp_1 * np.pi) / (nout - 1)
    tmp_2 = np.array(np.arange(0, order + 1), ndmin=2)
    zz = np.exp(np.matmul(tmp_1, tmp_2))

    # Actual polyvals, in power (mag^2)
    features = np.tile(gg, (nout, 1)) / np.abs(np.matmul(zz, aa))**2
    F = np.zeros((cols, int(np.ceil(rows / 2))))
    M = F

    if FMout:
        for c in range(cols):
            aaa = aa[:, c]
            rr = np.roots(aaa)
            ff_tmp = np.angle(rr)
            ff = np.array(ff_tmp, ndmin=2).T
            zz = np.exp(
                1j *
                np.matmul(ff, np.array(np.arange(0, aaa.shape[0]), ndmin=2)))
            mags = np.sqrt(gg[c] /
                           np.abs(np.matmul(zz,
                                            np.array(aaa, ndmin=2).T))**2)

            ix = np.argsort(ff_tmp)
            dummy = np.sort(ff_tmp)
            tmp_F_list = []
            tmp_M_list = []

            for i in range(ff.shape[0]):
                if dummy[i] > 0:
                    tmp_F_list = np.append(tmp_F_list, dummy[i])
                    tmp_M_list = np.append(tmp_M_list, mags[ix[i]])

            M[c, 0:tmp_M_list.shape[0]] = tmp_M_list
            F[c, 0:tmp_F_list.shape[0]] = tmp_F_list

    return features, F, M

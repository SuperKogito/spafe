import pytest
import scipy.io.wavfile
from spafe.utils import vis
from spafe.features.pncc import pncc
from spafe.features.lfcc import lfcc
from spafe.features.gfcc import gfcc
from spafe.features.ngcc import ngcc
from spafe.features.bfcc import bfcc
from spafe.features.msrcc import msrcc
from spafe.features.psrcc import psrcc
from spafe.features.lpc import lpc, lpcc
from spafe.features.rplp import rplp, plp
from spafe.utils.exceptions import ParameterError
from spafe.utils.spectral import stft, display_stft
from spafe.features.mfcc import mfcc, imfcc, mfe, melfcc

DEBUG_MODE = True


def get_data(fname):
    return scipy.io.wavfile.read(fname)


@pytest.fixture
def sig():
    __EXAMPLE_FILE = 'test.wav'
    return scipy.io.wavfile.read(__EXAMPLE_FILE)[1]


@pytest.fixture
def fs():
    __EXAMPLE_FILE = 'test.wav'
    return scipy.io.wavfile.read(__EXAMPLE_FILE)[0]


@pytest.mark.test_id(201)
@pytest.mark.parametrize('num_ceps', [13, 19, 26])
@pytest.mark.parametrize('low_freq', [0, 50, 300])
@pytest.mark.parametrize('high_freq', [2000, 4000])
@pytest.mark.parametrize('n_bands', [12, 18, 24])
@pytest.mark.parametrize('nfft', [256, 512, 1024])
def test_mfcc(sig,
              fs,
              num_ceps,
              low_freq,
              high_freq,
              n_bands,
              nfft,
              lifter_exp=0.6,
              fb_type='fcmel',
              dct_type=1,
              use_cmp=True,
              win_len=0.025,
              win_hop=0.01,
              pre_emph=0.97,
              dither=1,
              sumpower=1,
              band_width=1,
              model_order=0,
              broaden=0,
              use_energy=False):
    # compute mfccs and mfes
    mfccs = mfcc(sig,
         fs=16000,
         num_ceps=13,
         pre_emph=0,
         pre_emph_coeff=0.97,
         win_len=0.025,
         win_hop=0.01,
         win_type="hamming",
         nfilts=26,
         nfft=512,
         low_freq=None,
         high_freq=None,
         scale="constant",
         dct_type=2,
         use_energy=False,
         lifter=22,
         normalize=1)
    imfccs = imfcc(sig,
         fs=16000,
         num_ceps=13,
         pre_emph=0,
         pre_emph_coeff=0.97,
         win_len=0.025,
         win_hop=0.01,
         win_type="hamming",
         nfilts=26,
         nfft=512,
         low_freq=None,
         high_freq=None,
         scale="constant",
         dct_type=2,
         use_energy=False,
         lifter=22,
         normalize=1)
    mfes = mfe(sig, fs)
    # visualize the results
    if DEBUG_MODE:
        vis.visualize_features(mfccs, 'MFCC Index', 'Frame Index')

    # compute mfccs and mfes
    melfccs = melfcc(sig, fs)
    # visualize the results
    if DEBUG_MODE:
        vis.visualize_features(melfccs.T, 'DE-MFCC Coefficient Index',
                               'Frame Index')

    # visualize the results
    if DEBUG_MODE:
        vis.visualize_features(imfccs, 'IMFCC Index', 'Frame Index')
        vis.plot(mfes, 'MFE Coefficient Index', 'Frame Index')
    assert True


@pytest.mark.test_id(202)
@pytest.mark.parametrize('num_ceps', [13, 19, 26])
@pytest.mark.parametrize('low_freq', [0, 50, 300])
@pytest.mark.parametrize('high_freq', [2000, 4000])
@pytest.mark.parametrize('nfilts', [32, 48, 64])
@pytest.mark.parametrize('nfft', [256, 512, 1024])
def test_lfcc(sig,
              fs,
              num_ceps,
              low_freq,
              high_freq,
              nfilts,
              nfft,
              lifter_exp=0.6,
              fb_type='fcmel',
              dct_type=1,
              use_cmp=True,
              win_len=0.025,
              win_hop=0.01,
              pre_emph=0.97,
              dither=1,
              sumpower=1,
              band_width=1,
              model_order=0,
              broaden=0):

    # compute mfccs and mfes
    lfccs = lfcc(
        sig=sig,
        fs=fs,
        num_ceps=num_ceps,
        low_freq=low_freq,
        high_freq=high_freq,
        nfilts=nfilts,
        nfft=nfft)

    # assert number of returned cepstrum coefficients
    assert lfccs.shape[1] == num_ceps

    # assert number of filters is bigger than number of cepstrums
    with pytest.raises(ParameterError):
        # compute mfccs and mfes
        lfccs = lfcc(
            sig=sig,
            fs=fs,
            num_ceps=num_ceps,
            low_freq=low_freq,
            high_freq=high_freq,
            nfilts=num_ceps - 1,
            nfft=nfft)

    # check lifter Parameter error for low freq
    with pytest.raises(ParameterError):
        lfccs = lfcc(
            sig=sig,
            fs=fs,
            num_ceps=num_ceps,
            low_freq=-5,
            high_freq=high_freq,
            nfilts=nfilts,
            nfft=nfft)

    # check lifter Parameter error for high freq
    with pytest.raises(ParameterError):
        lfccs = lfcc(
            sig=sig,
            fs=fs,
            num_ceps=num_ceps,
            low_freq=low_freq,
            high_freq=16000,
            nfilts=nfilts,
            nfft=nfft)

    if DEBUG_MODE:
        vis.visualize_features(lfccs, 'LFCC Index', 'Frame Index')
    assert True


@pytest.mark.test_id(203)
@pytest.mark.parametrize('num_ceps', [13, 19, 26])
@pytest.mark.parametrize('low_freq', [0, 50, 300])
@pytest.mark.parametrize('high_freq', [2000, 4000])
@pytest.mark.parametrize('nfilts', [32, 48, 64])
@pytest.mark.parametrize('nfft', [256, 512, 1024])
def test_gfcc(sig,
              fs,
              num_ceps,
              low_freq,
              high_freq,
              nfilts,
              nfft,
              lifter_exp=0.6,
              fb_type='fcmel',
              dct_type=1,
              use_cmp=True,
              win_len=0.025,
              win_hop=0.01,
              pre_emph=0.97,
              dither=1,
              sumpower=1,
              band_width=1,
              model_order=0,
              broaden=0):

    # compute mfccs and mfes
    gfccs = gfcc(
        sig=sig,
        fs=fs,
        num_ceps=num_ceps,
        low_freq=low_freq,
        high_freq=high_freq,
        nfilts=nfilts,
        nfft=nfft)

    # assert number of returned cepstrum coefficients
    assert gfccs.shape[1] == num_ceps

    # assert number of filters is bigger than number of cepstrums
    with pytest.raises(ParameterError):
        # compute mfccs and mfes
        gfccs = gfcc(
            sig=sig,
            fs=fs,
            num_ceps=num_ceps,
            low_freq=low_freq,
            high_freq=high_freq,
            nfilts=num_ceps - 1,
            nfft=nfft)

    # check lifter Parameter error for low freq
    with pytest.raises(ParameterError):
        gfccs = gfcc(
            sig=sig,
            fs=fs,
            num_ceps=num_ceps,
            low_freq=-5,
            high_freq=high_freq,
            nfilts=nfilts,
            nfft=nfft)

    # check lifter Parameter error for high freq
    with pytest.raises(ParameterError):
        gfccs = gfcc(
            sig=sig,
            fs=fs,
            num_ceps=num_ceps,
            low_freq=low_freq,
            high_freq=16000,
            nfilts=nfilts,
            nfft=nfft)

    if DEBUG_MODE:
        vis.visualize_features(gfccs, 'GFCC Index', 'Frame Index')
    assert True


@pytest.mark.test_id(204)
@pytest.mark.parametrize('num_ceps', [13, 19, 26])
@pytest.mark.parametrize('low_freq', [0, 50, 300])
@pytest.mark.parametrize('high_freq', [2000, 4000])
@pytest.mark.parametrize('nfilts', [32, 48, 64])
@pytest.mark.parametrize('nfft', [256, 512, 1024])
def test_ngcc(sig,
              fs,
              num_ceps,
              low_freq,
              high_freq,
              nfilts,
              nfft,
              lifter_exp=0.6,
              fb_type='fcmel',
              dct_type=1,
              use_cmp=True,
              win_len=0.025,
              win_hop=0.01,
              pre_emph=0.97,
              dither=1,
              sumpower=1,
              band_width=1,
              model_order=0,
              broaden=0):

    # compute mfccs and mfes
    ngccs = ngcc(
        sig=sig,
        fs=fs,
        num_ceps=num_ceps,
        low_freq=low_freq,
        high_freq=high_freq,
        nfilts=nfilts,
        nfft=nfft)

    # assert number of returned cepstrum coefficients
    assert ngccs.shape[1] == num_ceps

    # assert number of filters is bigger than number of cepstrums
    with pytest.raises(ParameterError):
        # compute mfccs and mfes
        ngccs = ngcc(
            sig=sig,
            fs=fs,
            num_ceps=num_ceps,
            low_freq=low_freq,
            high_freq=high_freq,
            nfilts=num_ceps - 1,
            nfft=nfft)

    # check lifter Parameter error for low freq
    with pytest.raises(ParameterError):
        ngccs = ngcc(
            sig=sig,
            fs=fs,
            num_ceps=num_ceps,
            low_freq=-5,
            high_freq=high_freq,
            nfilts=nfilts,
            nfft=nfft)

    # check lifter Parameter error for high freq
    with pytest.raises(ParameterError):
        ngccs = ngcc(
            sig=sig,
            fs=fs,
            num_ceps=num_ceps,
            low_freq=low_freq,
            high_freq=16000,
            nfilts=nfilts,
            nfft=nfft)
    if DEBUG_MODE:
        vis.visualize_features(ngccs, 'NGCC Index', 'Frame Index')
    assert True


@pytest.mark.test_id(205)
@pytest.mark.parametrize('num_ceps', [13, 19, 26])
@pytest.mark.parametrize('low_freq', [0, 50, 300])
@pytest.mark.parametrize('high_freq', [2000, 4000])
@pytest.mark.parametrize('nfilts', [32, 48, 64])
@pytest.mark.parametrize('nfft', [256, 512, 1024])
def test_bfcc(sig,
              fs,
              num_ceps,
              low_freq,
              high_freq,
              nfilts,
              nfft,
              lifter_exp=0.6,
              fb_type='fcmel',
              dct_type=1,
              use_cmp=True,
              win_len=0.025,
              win_hop=0.01,
              pre_emph=0.97,
              dither=1,
              sumpower=1,
              band_width=1,
              model_order=0,
              broaden=0):

    # compute mfccs and mfes
    bfccs = bfcc(
        sig=sig,
        fs=fs,
        num_ceps=num_ceps,
        low_freq=low_freq,
        high_freq=high_freq,
        nfilts=nfilts,
        nfft=nfft)

    # assert number of returned cepstrum coefficients
    assert bfccs.shape[1] == num_ceps

    # assert number of filters is bigger than number of cepstrums
    with pytest.raises(ParameterError):
        # compute mfccs and mfes
        bfccs = bfcc(
            sig=sig,
            fs=fs,
            num_ceps=num_ceps,
            low_freq=low_freq,
            high_freq=high_freq,
            nfilts=num_ceps - 1,
            nfft=nfft)

    # check lifter Parameter error for low freq
    with pytest.raises(ParameterError):
        bfccs = bfcc(
            sig=sig,
            fs=fs,
            num_ceps=num_ceps,
            low_freq=-5,
            high_freq=high_freq,
            nfilts=nfilts,
            nfft=nfft)

    # check lifter Parameter error for high freq
    with pytest.raises(ParameterError):
        bfccs = bfcc(
            sig=sig,
            fs=fs,
            num_ceps=num_ceps,
            low_freq=low_freq,
            high_freq=16000,
            nfilts=nfilts,
            nfft=nfft)

    if DEBUG_MODE:
        vis.visualize_features(bfccs, 'BFCC Index', 'Frame Index')
    assert True


@pytest.mark.test_id(206)
@pytest.mark.parametrize('num_ceps', [13, 19, 26])
@pytest.mark.parametrize('low_freq', [0, 50, 300])
@pytest.mark.parametrize('high_freq', [2000, 4000])
@pytest.mark.parametrize('nfilts', [32, 48, 64])
@pytest.mark.parametrize('nfft', [256, 512, 1024])
def test_pncc(sig,
              fs,
              num_ceps,
              low_freq,
              high_freq,
              nfilts,
              nfft,
              lifter_exp=0.6,
              fb_type='fcmel',
              dct_type=1,
              use_cmp=True,
              win_len=0.025,
              win_hop=0.01,
              pre_emph=0.97,
              dither=1,
              sumpower=1,
              band_width=1,
              model_order=0,
              broaden=0):

    # compute mfccs and mfes
    pnccs = pncc(
        sig=sig,
        fs=fs,
        num_ceps=num_ceps,
        low_freq=low_freq,
        high_freq=high_freq,
        nfilts=nfilts,
        nfft=nfft)

    # assert number of returned cepstrum coefficients
    assert pnccs.shape[1] == num_ceps

    # assert number of filters is bigger than number of cepstrums
    with pytest.raises(ParameterError):
        # compute mfccs and mfes
        pnccs = pncc(
            sig=sig,
            fs=fs,
            num_ceps=num_ceps,
            low_freq=low_freq,
            high_freq=high_freq,
            nfilts=num_ceps - 1,
            nfft=nfft)

    # check lifter Parameter error for low freq
    with pytest.raises(ParameterError):
        pnccs = pncc(
            sig=sig,
            fs=fs,
            num_ceps=num_ceps,
            low_freq=-5,
            high_freq=high_freq,
            nfilts=nfilts,
            nfft=nfft)

    # check lifter Parameter error for high freq
    with pytest.raises(ParameterError):
        pnccs = pncc(
            sig=sig,
            fs=fs,
            num_ceps=num_ceps,
            low_freq=low_freq,
            high_freq=16000,
            nfilts=nfilts,
            nfft=nfft)

    if DEBUG_MODE:
        vis.visualize_features(pnccs, 'PNCC Index', 'Frame Index')
    assert True


@pytest.mark.test_id(207)
@pytest.mark.parametrize('num_ceps', [13, 19, 26])
@pytest.mark.parametrize('low_freq', [0, 50, 300])
@pytest.mark.parametrize('high_freq', [2000, 4000])
@pytest.mark.parametrize('nfilts', [32, 48, 64])
@pytest.mark.parametrize('nfft', [256, 512, 1024])
def test_rplp(sig,
              fs,
              num_ceps,
              low_freq,
              high_freq,
              nfilts,
              nfft,
              lifter_exp=0.6,
              fb_type='fcmel',
              dct_type=1,
              use_cmp=True,
              win_len=0.025,
              win_hop=0.01,
              pre_emph=0.97,
              dither=1,
              sumpower=1,
              band_width=1,
              model_order=0,
              broaden=0):
    # compute plps
    plps = plp(sig, fs, 13)
    if DEBUG_MODE:
        vis.visualize_features(plps, 'PLP Coefficient Index', 'Frame Index')
    # compute bfccs
    rplps = rplp(sig, fs, 13)
    if DEBUG_MODE:
        vis.visualize_features(rplps, 'RPLP Coefficient Index', 'Frame Index')
    assert True


@pytest.mark.test_id(208)
@pytest.mark.parametrize('num_ceps', [13, 19, 26])
@pytest.mark.parametrize('low_freq', [0, 50, 300])
@pytest.mark.parametrize('high_freq', [2000, 4000])
@pytest.mark.parametrize('nfilts', [32, 48, 64])
@pytest.mark.parametrize('nfft', [256, 512, 1024])
def test_lpc(sig,
             fs,
             num_ceps,
             low_freq,
             high_freq,
             nfilts,
             nfft,
             lifter_exp=0.6,
             fb_type='fcmel',
             dct_type=1,
             use_cmp=True,
             win_len=0.025,
             win_hop=0.01,
             pre_emph=0.97,
             dither=1,
             sumpower=1,
             band_width=1,
             model_order=0,
             broaden=0):
    # compute lpcs and lsps
    lpcs = lpc(sig, fs, 13)
    lpccs = lpcc(sig, fs, 13)
    if DEBUG_MODE:
        vis.visualize_features(lpcs, 'LPC Index', 'Frame Index')
    if DEBUG_MODE:
        vis.visualize_features(lpccs, 'LPCC Index', 'Frame Index')
    assert True


@pytest.mark.test_id(209)
@pytest.mark.parametrize('num_ceps', [13, 19, 26])
@pytest.mark.parametrize('low_freq', [0, 50, 300])
@pytest.mark.parametrize('high_freq', [2000, 4000])
@pytest.mark.parametrize('nfilts', [32, 48, 64])
@pytest.mark.parametrize('nfft', [256, 512, 1024])
def test_msrcc(sig,
               fs,
               num_ceps,
               low_freq,
               high_freq,
               nfilts,
               nfft,
               lifter_exp=0.6,
               fb_type='fcmel',
               dct_type=1,
               use_cmp=True,
               win_len=0.025,
               win_hop=0.01,
               pre_emph=0.97,
               dither=1,
               sumpower=1,
               band_width=1,
               model_order=0,
               broaden=0):

    # compute mfccs and mfes
    msrccs = msrcc(
        sig=sig,
        fs=fs,
        num_ceps=num_ceps,
        low_freq=low_freq,
        high_freq=high_freq,
        nfilts=nfilts,
        nfft=nfft)

    # assert number of returned cepstrum coefficients
    assert msrccs.shape[1] == num_ceps

    # assert number of filters is bigger than number of cepstrums
    with pytest.raises(ParameterError):
        # compute mfccs and mfes
        msrccs = msrcc(
            sig=sig,
            fs=fs,
            num_ceps=num_ceps,
            low_freq=low_freq,
            high_freq=high_freq,
            nfilts=num_ceps - 1,
            nfft=nfft)

    # check lifter Parameter error for low freq
    with pytest.raises(ParameterError):
        msrccs = msrcc(
            sig=sig,
            fs=fs,
            num_ceps=num_ceps,
            low_freq=-5,
            high_freq=high_freq,
            nfilts=nfilts,
            nfft=nfft)

    # check lifter Parameter error for high freq
    with pytest.raises(ParameterError):
        msrccs = msrcc(
            sig=sig,
            fs=fs,
            num_ceps=num_ceps,
            low_freq=low_freq,
            high_freq=16000,
            nfilts=nfilts,
            nfft=nfft)

    if DEBUG_MODE:
        vis.visualize_features(msrccs, 'MSRCC Index', 'Frame Index')
    assert True


@pytest.mark.test_id(210)
@pytest.mark.parametrize('num_ceps', [13, 19, 26])
@pytest.mark.parametrize('low_freq', [0, 50, 300])
@pytest.mark.parametrize('high_freq', [2000, 4000])
@pytest.mark.parametrize('nfilts', [32, 48, 64])
@pytest.mark.parametrize('nfft', [256, 512, 1024])
def test_psrcc(sig,
               fs,
               num_ceps,
               low_freq,
               high_freq,
               nfilts,
               nfft,
               lifter_exp=0.6,
               fb_type='fcmel',
               dct_type=1,
               use_cmp=True,
               win_len=0.025,
               win_hop=0.01,
               pre_emph=0.97,
               dither=1,
               sumpower=1,
               band_width=1,
               model_order=0,
               broaden=0):

    # compute mfccs and mfes
    psrccs = psrcc(
        sig=sig,
        fs=fs,
        num_ceps=num_ceps,
        low_freq=low_freq,
        high_freq=high_freq,
        nfilts=nfilts,
        nfft=nfft)

    # assert number of returned cepstrum coefficients
    assert psrccs.shape[1] == num_ceps

    # assert number of filters is bigger than number of cepstrums
    with pytest.raises(ParameterError):
        # compute mfccs and mfes
        psrccs = psrcc(
            sig=sig,
            fs=fs,
            num_ceps=num_ceps,
            low_freq=low_freq,
            high_freq=high_freq,
            nfilts=num_ceps - 1,
            nfft=nfft)

    # check lifter Parameter error for low freq
    with pytest.raises(ParameterError):
        psrccs = psrcc(
            sig=sig,
            fs=fs,
            num_ceps=num_ceps,
            low_freq=-5,
            high_freq=high_freq,
            nfilts=nfilts,
            nfft=nfft)

    # check lifter Parameter error for high freq
    with pytest.raises(ParameterError):
        psrccs = psrcc(
            sig=sig,
            fs=fs,
            num_ceps=num_ceps,
            low_freq=low_freq,
            high_freq=16000,
            nfilts=nfilts,
            nfft=nfft)

    if DEBUG_MODE:
        vis.visualize_features(psrccs, 'PSRCC Index', 'Frame Index')
    assert True


if __name__ == "__main__":
    # read wave file  and plot spectogram
    fs, sig = get_data('../test21.wav')
    if DEBUG_MODE:
        vis.spectogram(sig, fs)

    # compute and display STFT
    X, _ = stft(sig=sig, fs=fs, win_type="hann", win_len=0.025, win_hop=0.01)
    if DEBUG_MODE:
        display_stft(X, fs, len(sig), 0, 2000, -10, 0)

    # init input vars
    num_ceps = 13
    low_freq = 0
    high_freq = 2000
    nfilts = 24
    nfft = 512

    # run tests
    test_mfcc(sig, fs, num_ceps, low_freq, high_freq, nfilts, nfft)
    test_lfcc(sig, fs, num_ceps, low_freq, high_freq, nfilts, nfft)
    test_ngcc(sig, fs, num_ceps, low_freq, high_freq, nfilts, nfft)
    test_bfcc(sig, fs, num_ceps, low_freq, high_freq, nfilts, nfft)
    test_pncc(sig, fs, num_ceps, low_freq, high_freq, nfilts, nfft)
    test_msrcc(sig, fs, num_ceps, low_freq, high_freq, nfilts, nfft)
    test_psrcc(sig, fs, num_ceps, low_freq, high_freq, nfilts, nfft)
    test_lpc(sig, fs, num_ceps, low_freq, high_freq, nfilts, nfft)
    test_rplp(sig, fs, num_ceps, low_freq, high_freq, nfilts, nfft)
    test_gfcc(sig, fs, num_ceps, low_freq, high_freq, nfilts, nfft)

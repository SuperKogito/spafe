import pytest
import scipy.io.wavfile
from spafe.utils import vis
from spafe.features.lpc import lpc, lpcc
from spafe.features.lfcc import lfcc
from spafe.features.gfcc import gfcc
from spafe.features.ngcc import ngcc
from spafe.features.bfcc import bfcc
from spafe.features.msrcc import msrcc
from spafe.features.psrcc import psrcc
from spafe.features.pncc import pncc
from spafe.features.rplp import rplp, plp
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
def test_mfcc(sig, fs):
    # compute mfccs and mfes
    mfccs = mfcc(sig, 13)
    imfccs = imfcc(sig, 13)
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
def test_lfcc(sig, fs):
    # compute mfccs and mfes
    lfccs = lfcc(sig, fs, 13)
    if DEBUG_MODE:
        vis.visualize_features(lfccs, 'LFCC Index', 'Frame Index')
    assert True


@pytest.mark.test_id(203)
def test_gfcc(sig, fs):
    # compute gfccs
    gfccs = gfcc(sig, fs, 13)
    if DEBUG_MODE:
        vis.visualize_features(gfccs, 'GFCC Index', 'Frame Index')
    assert True


@pytest.mark.test_id(204)
def test_ngcc(sig, fs):
    # compute gfccs
    ngccs = ngcc(sig, fs, 13)
    if DEBUG_MODE:
        vis.visualize_features(ngccs, 'NGCC Index', 'Frame Index')
    assert True


@pytest.mark.test_id(205)
def test_bfcc(sig, fs):
    # compute bfccs
    bfccs = bfcc(sig, fs, 13)
    if DEBUG_MODE:
        vis.visualize_features(bfccs, 'BFCC Index', 'Frame Index')
    assert True


@pytest.mark.test_id(206)
def test_pncc(sig, fs):
    # compute bfccs
    pnccs = pncc(sig, fs, 13)
    if DEBUG_MODE:
        vis.visualize_features(pnccs, 'PNCC Index', 'Frame Index')
    assert True


@pytest.mark.test_id(207)
def test_rplp(sig, fs):
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
def test_lpc(sig, fs):
    # compute lpcs and lsps
    lpcs = lpc(sig, fs, 13)
    lpccs = lpcc(sig, fs, 13)
    if DEBUG_MODE:
        vis.visualize_features(lpcs, 'LPC Index', 'Frame Index')
    if DEBUG_MODE:
        vis.visualize_features(lpccs, 'LPCC Index', 'Frame Index')
    assert True


@pytest.mark.test_id(209)
def test_msrcc(sig, fs):
    # compute lpcs and lsps
    msrccs = msrcc(sig, fs, 13)
    if DEBUG_MODE:
        vis.visualize_features(msrccs, 'MSRCC Index', 'Frame Index')
    assert True


@pytest.mark.test_id(210)
def test_psrcc(sig, fs):
    # compute lpcs and lsps
    psrccs = psrcc(sig, fs, 13)
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

    # run tests
    test_mfcc(sig, fs)
    test_lfcc(sig, fs)
    test_gfcc(sig, fs)
    test_ngcc(sig, fs)
    test_bfcc(sig, fs)
    test_pncc(sig, fs)
    test_msrcc(sig, fs)
    test_psrcc(sig, fs)
    test_lpc(sig, fs)
    test_rplp(sig, fs)

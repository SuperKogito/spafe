import pytest
import scipy.io.wavfile
from spafe.utils import vis
from spafe.features.plp import plp
from spafe.features.lpc import lpc
from spafe.features.lsp import lsp
from spafe.features.lfcc import lfcc
from spafe.features.gfcc import gfcc
from spafe.features.ngcc import ngcc
from spafe.features.bfcc import bfcc
from spafe.features.cqcc import cqcc
from spafe.features.pncc import pncc
from spafe.features.rplp import rplp
from spafe.features.mfcc import mfcc, imfcc, mfe


# read wave file  and plot spectogram 
data = lambda : scipy.io.wavfile.read('test.wav')

@pytest.mark.test_id(201)
def test_mfcc():
    fs, sig = data()
    # compute mfccs and mfes
    mfccs  = mfcc(sig, 13)
    imfccs = imfcc(sig, 13)
    mfes   = mfe(sig, fs) 
    # visualize the results
    vis.visualize_features(mfccs, 'MFCC Coefficient Index','Frame Index', True)
    vis.visualize_features(imfccs, 'IMFCC Coefficient Index','Frame Index', True)
    vis.plot(mfes,  'MFE Coefficient Index','Frame Index', True)
    print("MFCC features extraction success")
    assert True

@pytest.mark.test_id(202)
def test_lfcc():
    fs, sig = data()
    # compute mfccs and mfes
    lfccs = lfcc(sig, 13)
    vis.visualize_features(lfccs, 'LFCC Coefficient Index','Frame Index', True)
    assert True

@pytest.mark.test_id(203)
def test_gfcc():
    fs, sig = data()
    # compute gfccs
    gfccs = gfcc(sig, 13) 
    vis.visualize_features(gfccs, 'GFCC Coefficient Index','Frame Index', True)
    assert True
    
@pytest.mark.test_id(204)
def test_ngcc():
    fs, sig = data()
    # compute gfccs
    ngccs = ngcc(sig, 13) 
    vis.visualize_features(ngccs, 'NGC Coefficient Index','Frame Index', True)
    assert True

@pytest.mark.test_id(205)
def test_bfcc():
    fs, sig = data()
    # compute bfccs
    bfccs = bfcc(sig, 13)
    vis.visualize_features(bfccs, 'BFCC Coefficient Index','Frame Index', True)
    assert True

@pytest.mark.test_id(206)
def test_pncc(): 
    fs, sig = data()
    # compute bfccs
    pnccs = pncc(sig, 13)
    vis.visualize_features(pnccs, 'pnccs Coefficient Index','Frame Index', True)
    assert True

@pytest.mark.test_id(207)
def test_cqcc():
    fs, sig = data()
    # compute bfccs
    cqccs= cqcc(sig, 13)
    vis.plot(cqccs, 'CQC Coefficient Index','Frame Index', True)

@pytest.mark.test_id(208)
def test_plp():    
    fs, sig = data()
    # compute bfccs
    plps = plp(sig, 13)
    vis.visualize_features(plps, 'PLP Coefficient Index','Frame Index', True)
    assert True
    
@pytest.mark.test_id(209)
def test_rplp():
    fs, sig = data()
    # compute bfccs
    rplps = rplp(sig, 13)
    vis.visualize_features(rplps, 'RPLP Coefficient Index','Frame Index', True)
    assert True

@pytest.mark.test_id(210)
def test_lpc():
    fs, sig = data()
    # compute lpcs and lsps
    lpcs = lpc(sig, 1)
    vis.visualize_features(lpcs, 'LPC Coefficient Index','Frame Index', True)
    assert True

@pytest.mark.test_id(211)
def test_lsp():
    fs, sig = data()
    lpcs = lpc(sig, 1)
    lsps = lsp(lpcs)
    vis.visualize_features(lsps, 'LSP Coefficient Index','Frame Index', True)
    assert True


if __name__ == "__main__":
    # read wave file  and plot spectogram 
    fs, sig = data()    
    vis.spectogram(sig, fs, True)
    # run tests
    test_mfcc()
    test_lfcc()
    test_gfcc()
    test_ngcc()
    test_bfcc()
    test_pncc() 
    test_cqcc()
    test_plp()    
    test_rplp()
    test_lpc()
    test_lsp()




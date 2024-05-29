import pytest
import numpy as np
import scipy.io.wavfile
from spafe.utils import vis
from spafe.features.cqcc import cqcc
from spafe.utils.preprocessing import framing
from spafe.utils.exceptions import ParameterError
from spafe.utils.cepstral import normalize_ceps, lifter_ceps


@pytest.mark.test_id(202)
@pytest.mark.usefixtures("sig")
@pytest.mark.usefixtures("fs")
@pytest.mark.parametrize("num_ceps", [13, 28])
@pytest.mark.parametrize("pre_emph", [False, True])
@pytest.mark.parametrize("nfft", [1024])
@pytest.mark.parametrize("low_freq", [50])
@pytest.mark.parametrize("high_freq", [4000])
@pytest.mark.parametrize("dct_type", [1, 2, 4])
@pytest.mark.parametrize("lifter", [None, 0.7, -7])
@pytest.mark.parametrize("normalize", [None, "mvn", "ms"])
@pytest.mark.parametrize("resampling_ratio", [1.0, 0.9, 0.3])
def test_cqcc(
    sig,
    fs,
    num_ceps,
    pre_emph,
    nfft,
    low_freq,
    high_freq,
    dct_type,
    lifter,
    normalize,
    resampling_ratio,
):
    """
    test cqcc features module for the following:
        - check if ParameterErrors are raised for:
                - nfilts < num_ceps
                - negative low_freq value
                - high_freq > fs / 2
        - check that the returned number of cepstrums is correct.
        - check normalization.
        - check liftering.
    """
    # get number of frames
    frames, frame_length = framing(sig=sig, fs=fs, win_len=0.025, win_hop=0.01)
    num_frames = len(frames)

    # check error for number of filters is smaller than number of cepstrums
    with pytest.raises(ParameterError):
        cqccs = cqcc(
            sig=sig,
            fs=fs,
            num_ceps=num_ceps,
            nfft=nfft,
            low_freq=low_freq,
            high_freq=fs,
            resampling_ratio=resampling_ratio,
        )

    # compute features
    cqccs = cqcc(
        sig=sig,
        fs=fs,
        num_ceps=num_ceps,
        pre_emph=pre_emph,
        nfft=nfft,
        low_freq=low_freq,
        high_freq=high_freq,
        dct_type=dct_type,
        lifter=lifter,
        normalize=normalize,
        resampling_ratio=resampling_ratio,
    )

    # assert number of returned cepstrum coefficients
    if not cqccs.shape[1] == num_ceps:
        raise AssertionError

    # assert number of returned cepstrum coefficients
    if not cqccs.shape[0] == int(num_frames * resampling_ratio):
        raise AssertionError

    # check normalize
    if normalize:
        np.testing.assert_array_almost_equal(
            cqccs,
            normalize_ceps(
                cqcc(
                    sig=sig,
                    fs=fs,
                    num_ceps=num_ceps,
                    pre_emph=pre_emph,
                    nfft=nfft,
                    low_freq=low_freq,
                    high_freq=high_freq,
                    dct_type=dct_type,
                    lifter=lifter,
                    normalize=None,
                    resampling_ratio=resampling_ratio,
                ),
                normalize,
            ),
            3,
        )
    else:
        # check lifter
        if lifter:
            np.testing.assert_array_almost_equal(
                cqccs,
                lifter_ceps(
                    cqcc(
                        sig=sig,
                        fs=fs,
                        num_ceps=num_ceps,
                        pre_emph=pre_emph,
                        nfft=nfft,
                        low_freq=low_freq,
                        high_freq=high_freq,
                        dct_type=dct_type,
                        lifter=None,
                        normalize=normalize,
                        resampling_ratio=resampling_ratio,
                    ),
                    lifter,
                ),
                3,
            )

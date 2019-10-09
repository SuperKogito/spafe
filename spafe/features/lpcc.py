import numpy


def lpcc(seq, order=None):
    '''
    Computes the linear predictive cepstral components.

    Note:
        Returned values are in the frequency domain

    Examples:
        audiofile = AudioFile.open('file.wav',16000)
        frames    = audiofile.frames(512,np.hamming)
        for frame in frames:
            frame.lpcc()

    Note that we already preprocess in the Frame class the lpc conversion!

    Args:
        seq                    : A sequence of lpc components. Need to be preprocessed by lpc()
        err_term               : Error term for lpc sequence. Returned by lpc()[1]
        order (default = None) : Return size of the array. Function returns order+1 length array. 
                                 Default is len(seq)

    Returns:
        List with lpcc components with default length len(seq), otherwise length order +1
    '''
    if order is None:
        order = len(seq) - 1

    lpcc_coeffs = [-seq[0], -seq[1]]

    for n in range(2, order + 1):
        # Use order + 1 as upper bound for the last iteration
        upbound    = (order + 1 if n > order else n)
        lpcc_coef  = -sum(i * lpcc_coeffs[i] * seq[n - i - 1] for i in range(1, upbound)) * 1. / upbound
        lpcc_coef -= seq[n - 1] if n <= len(seq) else 0
        lpcc_coeffs.append(lpcc_coef)
    return lpcc_coeffs

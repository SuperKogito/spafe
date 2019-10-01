import numpy


def lsp(lpcseq, rectify=True):
    """
    Computes Line spectrum pairs (also called  line spectral frequencies [lsf]).
    Does not use any fancy algorithm except numpy.roots to solve for the zeros of
    the given polynom A(z) = 0.5(P(z) + Q(z))

    Examples:
        audiofile = AudioFile.open('file.wav',16000)
        frames    = audiofile.frames(512,numpy.hamming)
        for frame in frames:
            frame.lpcc()

    Args:
        lpcseq (array) : The sequence of lpc coefficients as \sum_k=1^{p} a_k z^{-k}
        rectify (bool) : If true returns only the values >= 0, since the result is symmetric.
                         If all values are wished, specify rectify = False, (default = True)

    Returns:
        (list) A list with same length as lpcseq (if rectify = True),
               otherwise 2*len(lpcseq), which represents the line spectrum pairs
    """
    # We obtain 1 - A(z) +/- z^-(p+1) * (1 - A(z))
    # After multiplying everything through it becomes
    # 1 - \sum_k=1^p a_k z^{-k} +/- z^-(p+1) - \sum_k=1^p a_k z^{k-(p+1)}
    # Thus we have on the LHS the usual lpc polynomial and on the RHS we need to reverse the coefficient order
    # We assume further that the lpc polynomial is a valid one ( first coefficient is 1! )

    # the rhs does not have a constant expression and we reverse the coefficients
    rhs = [0] + lpcseq[::-1] + [1]

    # init the P and the Q polynomials
    P, Q = [], []

    # Assuming constant coefficient is 1, which is required. Moreover z^{-p+1} does not exist on the lhs, thus appending 0
    lpcseq = [1] + lpcseq[:] + [0]
    for l,r in itertools.zip_longest(lpcseq, rhs):
        P.append(l + r)
        Q.append(l - r)

    # Find the roots of the polynomials P,Q ( np assumes we have the form of: p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]
    # mso we need to reverse the order)
    p_roots = numpy.roots(P[::-1])
    q_roots = numpy.roots(Q[::-1])

    # Keep the roots in order
    lsf_p = sorted(numpy.angle(p_roots))
    lsf_q = sorted(numpy.angle(q_roots))

    # print sorted(lsf_p+lsf_q),len([i for  i in lsf_p+lsf_q if i > 0.])
    if rectify:
        # We only return the positive elements, and also remove the final Pi (3.14) value at the end,
        # since it always occurs
        return sorted(i for i in lsf_q + lsf_p if (i > 0))[:-1]
    else:
        # Remove the -Pi and +pi at the beginning and end in the list
        return sorted(i for i in lsf_q + lsf_p)[1:-1]

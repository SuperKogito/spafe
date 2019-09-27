import librosa
import itertools
import numpy as np


class FeaturesExtractor:
    def __init__(self, signal, rate):
        self.signal, self.rate = signal, rate
        
    def get_mfcc_features(self, signal, rate):
        melspectrogram = librosa.feature.melspectrogram(y=self.signal, sr=self.rate, n_mels=80, fmax=8000) 
        # Extract MFCCs, MFCC deltas, MFCC dosuble deltas and MFECs
        mfcc        = librosa.feature.mfcc(signal, n_mfcc=13)
        delta_mfcc  = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        mfec        = librosa.power_to_db(melspectrogram) 
        return [mfec, mfcc, delta_mfcc, delta2_mfcc]

    def get_linear_features(self, signal, rate):
        # Extract LPCs, LPCCs and LSPs
        lpc  = librosa.core.lpc(signal, order=3)
        lpcc = self.lpcc(lpc)
        lsp  = self.lsp(lpc)
        return [lpc, lpcc, lsp]

    def get_gfcc(self, signal, rate):
        import gfcc
        return gfcc.get_gfcc(signal, rate)        

    def lpcc(self, seq, order=None):
        '''
        Function: lpcc
        Summary: Computes the linear predictive cepstral compoents. Note: Returned values are in the frequency domain
        Examples: audiofile = AudioFile.open('file.wav',16000)
                  frames = audiofile.frames(512,np.hamming)
                  for frame in frames:
                    frame.lpcc()
                  Note that we already preprocess in the Frame class the lpc conversion!
        Attributes:
            @param (seq):A sequence of lpc components. Need to be preprocessed by lpc()
            @param (err_term):Error term for lpc sequence. Returned by lpc()[1]
            @param (order) default=None: Return size of the array. Function returns order+1 length array. Default is len(seq)
        Returns: List with lpcc components with default length len(seq), otherwise length order +1
        '''
        if order is None: order = len(seq) - 1
        lpcc_coeffs = [-seq[0], -seq[1]]
        for n in range(2, order + 1):
            # Use order + 1 as upper bound for the last iteration
            upbound    = (order + 1 if n > order else n)
            lpcc_coef  = -sum(i * lpcc_coeffs[i] * seq[n - i - 1] for i in range(1, upbound)) * 1. / upbound
            lpcc_coef -= seq[n - 1] if n <= len(seq) else 0
            lpcc_coeffs.append(lpcc_coef)
        return lpcc_coeffs

    def lsp(self, lpcseq, rectify=True):
        """
        Function: lsp
        Summary: Computes Line spectrum pairs ( also called  line spectral frequencies [lsf]). Does not use any fancy algorithm except np.roots to solve
        for the zeros of the given polynom A(z) = 0.5(P(z) + Q(z))

        Examples:
            audiofile = AudioFile.open('file.wav',16000)
            frames    = audiofile.frames(512,np.hamming)
            for frame in frames:
                frame.lpcc()

        Args:
            lpcseq (array) : The sequence of lpc coefficients as \sum_k=1^{p} a_k z^{-k}
            rectify (bool) : If true returns only the values >= 0, since the result is symmetric. If all values are wished, specify rectify = False, (default = True)

        Returns:
            (list) A list with same length as lpcseq (if rectify = True), otherwise 2*len(lpcseq), which represents the line spectrum pairs
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
        p_roots = np.roots(P[::-1])
        q_roots = np.roots(Q[::-1])
        # Keep the roots in order
        lsf_p = sorted(np.angle(p_roots))
        lsf_q = sorted(np.angle(q_roots))
        # print sorted(lsf_p+lsf_q),len([i for  i in lsf_p+lsf_q if i > 0.])
        if rectify:
            # We only return the positive elements, and also remove the final Pi (3.14) value at the end,
            # since it always occurs
            return sorted(i for i in lsf_q + lsf_p if (i > 0))[:-1]
        else:
            # Remove the -Pi and +pi at the beginning and end in the list
            return sorted(i for i in lsf_q + lsf_p)[1:-1]

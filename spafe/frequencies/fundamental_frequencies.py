# -*- coding: utf-8 -*-
"""
Credits to:
    Patrice Guyot. (2018, April 19).
    Fast Python implementation of the Yin algorithm (Version v1.1.1).
    Zenodo. http://doi.org/10.5281/zenodo.1220947
"""
import time
import scipy
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt


class FundamentalFrequenciesExtractor:
    def __init__(self, debug=False):
        self.debug = debug

    def differenceFunction(self, x, N, tau_max):
        """
        Compute difference function of data x. This corresponds to equation (6) in [1]
        Fastest implementation. Use the same approach than differenceFunction_scipy.
        This solution is implemented directly with np fft.

        Args:
            x       (array) : audio data
            N       (int)   : length of data
            tau_max (int)   : integration window size

        Returns:
            (list) : difference function
        """
        x = np.array(x, np.float64)
        w = x.size
        x_cumsum = np.concatenate((np.array([0]), (x * x).cumsum()))
        conv = scipy.signal.fftconvolve(x, x[::-1])
        tmp = x_cumsum[w:0:-1] + x_cumsum[w] - x_cumsum[:w] - 2 * conv[w - 1:]
        return tmp[:tau_max]

    def cumulativeMeanNormalizedDifferenceFunction(self, df, N):
        """
        Compute cumulative mean normalized difference function (CMND).
        This corresponds to equation (8) in [1].

        Args:
            df      (list) : Difference function
            N       (int)  : length of data
            tau_max (int)  : integration window size

        Returns:
            (list) : cumulative mean normalized difference function
        """
        cmndf = df[1:] * range(1, N) / np.cumsum(df[1:]).astype(
            float)  # scipy method
        return np.insert(cmndf, 0, 1)

    def getPitch(self, cmdf, tau_min, tau_max, harmo_th=0.1):
        """
        Return fundamental period of a frame based on CMND function.
            - cmdf: Cumulative Mean Normalized Difference function

        Args:
            tau_min  (int)   : minimum period for speech
            tau_max  (int)   : maximum period for speech
            harmo_th (float) : harmonicity threshold to determine if it is necessary to compute pitch frequency

        Returns:
            (float) : fundamental period if there is values under threshold, 0 otherwise
        """
        tau = tau_min
        while tau < tau_max:
            if cmdf[tau] < harmo_th:
                while tau + 1 < tau_max and cmdf[tau + 1] < cmdf[tau]:
                    tau += 1
                return tau
            tau += 1

        return 0  # if unvoiced

    def compute_yin(self,
                    sig,
                    fs,
                    dataFileName=None,
                    w_len=512,
                    w_step=256,
                    f0_min=50,
                    f0_max=3000,
                    harmo_thresh=0.1):
        """
        Compute the Yin Algorithm. Return fundamental frequency and harmonic rate.

        Args:
            sig        (list) : Audio signal (list of float)
            fs          (int) : sampling rate (= average number of samples pro 1 second)
            w_len       (int) : size of the analysis window (in #samples)
            w_step      (int) : size of the lag between two consecutives windows (in #samples)
            f0_min      (int) : Minimum fundamental frequency that can be detected (in Hertz)
            f0_max      (int) : Maximum fundamental frequency that can be detected (in Hertz)
            harmo_tresh (int) : Threshold of detection. The yalgorithmù return the first minimum of the CMND fubction below this threshold.

        Returns:
            (tuple) : tuple include the following
                          - pitches       : list of fundamental frequencies,
                          - harmonic_rates: list of harmonic rate values for each fundamental frequency value (= confidence value)
                          - argmins       : minimums of the Cumulative Mean Normalized DifferenceFunction
                          - times         : list of time of each estimation
        """
        if self.debug:
            print('Yin: compute yin algorithm')
        tau_min, tau_max = int(fs / f0_max), int(fs / f0_min)

        timeScale = range(0,
                          len(sig) - w_len,
                          w_step)  # time values for each analysis window
        times = [t / float(fs) for t in timeScale]
        frames = [sig[t:t + w_len] for t in timeScale]

        pitches = [0.0] * len(timeScale)
        harmonic_rates = [0.0] * len(timeScale)
        argmins = [0.0] * len(timeScale)

        for i, frame in enumerate(frames):
            # Compute YIN
            df = self.differenceFunction(frame, w_len, tau_max)
            cmdf = self.cumulativeMeanNormalizedDifferenceFunction(df, tau_max)
            p = self.getPitch(cmdf, tau_min, tau_max, harmo_thresh)

            # Get results
            if np.argmin(cmdf) > tau_min:
                argmins[i] = float(fs / np.argmin(cmdf))
            #  A pitch was found
            if p != 0:
                pitches[i] = float(fs / p)
                harmonic_rates[i] = cmdf[p]
            #  No pitch, but we compute a value of the harmonic rate
            else:
                harmonic_rates[i] = min(cmdf)

        return pitches, harmonic_rates, argmins, times

    def main(self,
             sig,
             fs,
             w_len=1024,
             w_step=256,
             f0_min=70,
             f0_max=200,
             harmo_thresh=0.85,
             audioDir="./",
             dataFileName=None):
        """
        Run the computation of the Yin algorithm on a example file.

        Args:
            sig        (list) : Audio signal (list of float)
            fs          (int) : sampling rate (= average number of samples pro 1 second)
            w_len       (int) : size of the analysis window (in #samples)
            w_step      (int) : size of the lag between two consecutives windows (in #samples)
            f0_min      (int) : Minimum fundamental frequency that can be detected (in Hertz)
            f0_max      (int) : Maximum fundamental frequency that can be detected (in Hertz)
            harmo_tresh (int) : Threshold of detection. The yalgorithmù return the first minimum of the CMND fubction below this threshold.

        Returns:
            (tuple) : tuple include the following
                          - pitches       : list of fundamental frequencies,
                          - harmonic_rates: list of harmonic rate values for each fundamental frequency value (= confidence value)
                          - argmins       : minimums of the Cumulative Mean Normalized DifferenceFunction
                          - times         : list of time of each estimation
        """
        start = time.time()
        print(sig, fs)
        duration = len(sig) / float(fs)
        pitches, harmonic_rates, argmins, times = self.compute_yin(
            sig, fs, dataFileName, w_len, w_step, f0_min, f0_max, harmo_thresh)

        if self.debug:
            print("Yin computed in: ", time.time() - start)
            plt.figure(figsize=(20, 10))
            plt.subplots_adjust(left=0.125,
                                right=0.9,
                                bottom=0.1,
                                top=0.9,
                                wspace=0.2,
                                hspace=0.99)
            # plot audio data
            ax1 = plt.subplot(4, 1, 1)
            ax1.plot(
                [float(x) * duration / len(sig) for x in range(0, len(sig))],
                sig)
            ax1.set_title('Audio data')
            ax1.set_ylabel('Amplitude')

            # plot F0
            ax2 = plt.subplot(4, 1, 2)
            ax2.plot([
                float(x) * duration / len(pitches)
                for x in range(0, len(pitches))
            ], pitches)
            ax2.set_title('F0')
            ax2.set_ylabel('Frequency (Hz)')

            # plot Harmonic rate
            ax3 = plt.subplot(4, 1, 3, sharex=ax2)
            ax3.plot([
                float(x) * duration / len(harmonic_rates)
                for x in range(0, len(harmonic_rates))
            ], harmonic_rates, "-x")
            ax3.plot([
                float(x) * duration / len(harmonic_rates)
                for x in range(0, len(harmonic_rates))
            ], [harmo_thresh] * len(harmonic_rates), 'r', "--")
            ax3.set_title('Harmonic rate')
            ax3.set_ylabel('Rate')

            # plot Index of minimums of CMND
            ax4 = plt.subplot(4, 1, 4, sharex=ax2)
            ax4.plot([
                float(x) * duration / len(argmins)
                for x in range(0, len(argmins))
            ], argmins, "-x")
            ax4.set_title('Index of minimums of CMND')
            ax4.set_ylabel('Frequency (Hz)')
            ax4.set_xlabel('Time (seconds)')
            plt.show()

        return np.array(pitches), harmonic_rates, argmins, times

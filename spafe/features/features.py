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

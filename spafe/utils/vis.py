# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt 


def visualize(mat, ylabel, xlabel):
    plt.imshow(mat, origin='lower', aspect='auto', interpolation='nearest')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()

def plot(y, ylabel, xlabel):
    plt.plot(y)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()
    
def spectogram(sig, fs):
    plt.specgram(sig, NFFT=512, Fs=fs)
    plt.ylabel("Frequency (kHz)")
    plt.xlabel("Time (s)")
    plt.show()

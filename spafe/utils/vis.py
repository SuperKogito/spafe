# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt 

def visualize_fbanks(fbanks, ylabel, xlabel, test_mode=False):
    for i in range(len(fbanks)):
        plt.plot(fbanks[i])
        plt.ylim(0, 1.1)
        plt.grid(True)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
    plt.show(block=False)
    if test_mode: plt.close()
    
def visualize_features(feats, ylabel, xlabel, test_mode=False):
    plt.imshow(feats, origin='lower', aspect='auto', interpolation='nearest')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show(block=False)
    if test_mode: plt.close()

def plot(y, ylabel, xlabel, test_mode=False):
    plt.plot(y)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show(block=False)
    if test_mode: plt.close()

def spectogram(sig, fs, test_mode=False):
    plt.specgram(sig, NFFT=512, Fs=fs)
    plt.ylabel("Frequency (kHz)")
    plt.xlabel("Time (s)")
    plt.show(block=False)
    if test_mode: plt.close()

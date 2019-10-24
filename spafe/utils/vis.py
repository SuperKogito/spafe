# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

def visualize_fbanks(fbanks, ylabel, xlabel):
    for i in range(len(fbanks)):
        plt.plot(fbanks[i])
        plt.ylim(0, 1.1)
        plt.grid(True)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
    plt.show(block=False)
    plt.close()

def visualize_features(feats, ylabel, xlabel, cmap='viridis'):
    plt.imshow(feats.T, origin='lower', aspect='auto',
               cmap=cmap,  interpolation='nearest')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show(block=False)
    plt.close()

def plot(y, ylabel, xlabel):
    plt.plot(y)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show(block=False)
    plt.close()

def spectogram(sig, fs):
    plt.specgram(sig, NFFT=1024, Fs=fs)
    plt.ylabel("Frequency (kHz)")
    plt.xlabel("Time (s)")
    plt.show(block=False)
    plt.close()

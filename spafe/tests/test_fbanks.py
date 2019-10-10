import matplotlib.pyplot as plt 
from spafe.fbanks import mel_fbanks
from spafe.fbanks import bark_fbanks
from spafe.fbanks import linear_fbanks
from spafe.fbanks import gammatone_fbanks


def visualize(fbanks, ylabel, xlabel):
    for i in range(len(fbanks)):
        plt.plot(fbanks[i])
        plt.ylim(0, 1.1)
        plt.grid(True)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
    plt.show()

if __name__ == "__main__":
    # compute the Mel, Bark and Gammaton filterbanks
    mel_fbanks   = mel_fbanks.mel_filter_banks(nfilts=24, nfft=512, fs=16000)
    bark_fbanks  = bark_fbanks.bark_filter_banks(nfilts=24, nfft=512, fs=16000)
    lin_fbanks   = linear_fbanks.linear_filter_banks(nfilts=24, nfft=512, fs=16000)
    gamma_fbanks = gammatone_fbanks.gammatone_filter_banks(nfilts=24, nfft=512, fs=16000)

    # plot the Mel filter banks 
    visualize(mel_fbanks, "Amplitude", "Frequency (Hz)")

    # plot the Linear filter banks 
    visualize(lin_fbanks, "Amplitude", "Frequency (Hz)")

    # plot the Bark filter banks 
    visualize(bark_fbanks, "Amplitude", "Frequency (Hz)")

    # plot the Gammatone filter banks 
    visualize(gamma_fbanks, "Amplitude", "Frequency (Hz)")

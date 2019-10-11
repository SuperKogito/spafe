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
    mel_filbanks   = mel_fbanks.mel_filter_banks(nfilts=14, nfft=512, fs=16000)
    bark_filbanks  = bark_fbanks.bark_filter_banks(nfilts=24, nfft=512, fs=16000)
    lin_filbanks   = linear_fbanks.linear_filter_banks(nfilts=24, nfft=512, fs=16000)
    gamma_filbanks = gammatone_fbanks.gammatone_filter_banks(nfilts=24, nfft=512, fs=16000)
    imel_filbanks  = mel_fbanks.inverse_mel_filter_banks(nfilts=24, nfft=512, fs=16000)
  
    # plot the Mel filter banks 
    visualize(mel_filbanks, "Amplitude", "Frequency (Hz)")

    # plot the inverse Mel filter banks 
    visualize(imel_filbanks, "Amplitude", "Frequency (Hz)")

    # plot the Linear filter banks 
    visualize(lin_filbanks, "Amplitude", "Frequency (Hz)")

    # plot the Bark filter banks 
    visualize(bark_filbanks, "Amplitude", "Frequency (Hz)")

    # plot the Gammatone filter banks 
    visualize(gamma_filbanks, "Amplitude", "Frequency (Hz)")

import matplotlib.pyplot as plt 
from spafe.fbanks import mel_fbanks
from spafe.fbanks import bark_fbanks
from spafe.fbanks import gammatone_fbanks


if __name__ == "__main__":
    # compute the Mel, Bark and Gammaton filterbanks
    mel_fbanks   = mel_fbanks.mel_filter_banks(nfilts=16, nfft=512, fs=16000)
    bark_fbanks  = bark_fbanks.bark_filter_banks(nfilts=16, nfft=512, fs=16000)
    gamma_fbanks = gammatone_fbanks.gammatone_filter_banks(nfilts=16, nfft=512, fs=16000)

    # plot the Mel filter banks 
    for i in range(len(mel_fbanks)):
        plt.plot(mel_fbanks[i])
        plt.ylim(0, 1.1)
        plt.grid(True)
    plt.show()
    
    # plot the Bark filter banks 
    for i in range(len(bark_fbanks)):
        plt.plot(bark_fbanks[i])
        plt.ylim(0, 1.1)
        plt.grid(True)
    plt.show()

    # plot the Gammatone filter banks 
    for i in range(len(gamma_fbanks)):
        plt.plot(gamma_fbanks[i])
        plt.ylim(0, 1.1)
        plt.grid(True)
    plt.show()
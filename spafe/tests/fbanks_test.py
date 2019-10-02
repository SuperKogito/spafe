import matplotlib.pyplot as plt 
from spafe.fbanks import mel_fbanks
from spafe.fbanks import bark_fbanks


if __name__ == "__main__":
    # compute the Bark and Mel filterbanks 
    bark_fbanks = bark_fbanks.bark_filter_banks(nfilt=26, nfft=512, fs=16000)
    mel_fbanks  = mel_fbanks.mel_filter_banks(nfilt=26,   nfft=512, fs=16000)

    # plot the Bark filter banks 
    for i in range(len(mel_fbanks)):
        plt.plot(mel_fbanks[i])
        plt.ylim(0, 1.1)
        plt.grid(True)
    plt.show()
    
    # plot the Mel filter banks 
    for i in range(len(bark_fbanks)):
        plt.plot(bark_fbanks[i])
        plt.ylim(0, 1.1)
        plt.grid(True)
    plt.show()

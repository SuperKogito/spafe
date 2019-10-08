import scipy.io.wavfile
import matplotlib.pyplot as plt

#read wave file 
fs, sig = scipy.io.wavfile.read('../sample.wav')


from spafe.features.mfcc import mfcc, mfe
# compute mfccs and mfes
mfccs = mfcc(sig, 13)
mfes  = mfe(sig, fs) 

plt.imshow(mfccs, origin='lower', aspect='auto', interpolation='nearest')
plt.ylabel('MFCC Coefficient Index')
plt.xlabel('Frame Index')
plt.show()

plt.plot(mfes)
plt.ylabel('MFE Coefficient Index')
plt.xlabel('Frame Index')
plt.show()


from spafe.features.gfcc import gfcc
# compute gfccs
gfccs = gfcc(sig, 13) 

plt.imshow(gfccs, origin='lower', aspect='auto', interpolation='nearest')
plt.ylabel('GFC Coefficient Index')
plt.xlabel('Frame Index')
plt.show()

from spafe.features.bfcc import bfcc
# compute bfccs
bfccs = bfcc(sig, 13)

plt.imshow(bfccs, origin='lower', aspect='auto', interpolation='nearest')
plt.ylabel('BFC Coefficient Index')
plt.xlabel('Frame Index')
plt.show()



from spafe.features.lpc import lpc
from spafe.features.lsp import lsp
# compute lpcs and lsps
lpcs = lpc(sig, 1)
lsps = lsp(lpcs)

plt.imshow(lpcs, origin='lower', aspect='auto', interpolation='nearest')
plt.ylabel('LP Coefficient Index')
plt.xlabel('Frame Index')
plt.show()



plt.imshow(lsps, origin='lower', aspect='auto', interpolation='nearest')
plt.ylabel('LSP Coefficient Index')
plt.xlabel('Frame Index')
plt.show()



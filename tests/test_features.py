import scipy.io.wavfile
from ..utils import vis


#read wave file 
fs, sig = scipy.io.wavfile.read('../test.wav')

# plot spectogram 
vis.spectogram(sig, fs)

from ..features.mfcc import mfcc, imfcc, mfe
# compute mfccs and mfes
mfccs  = mfcc(sig, 13)
imfccs = imfcc(sig, 13)
mfes   = mfe(sig, fs) 

vis.visualize(mfccs, 'MFCC Coefficient Index','Frame Index')
vis.visualize(imfccs, 'IMFCC Coefficient Index','Frame Index')
#vis.visualize((np.append(mfes, 0)).reshape(277,13),  'MFE Coefficient Index','Frame Index')
vis.plot(mfes,  'MFE Coefficient Index','Frame Index')


from ..features.lfcc import lfcc
# compute mfccs and mfes
lfccs = lfcc(sig, 13)
vis.visualize(lfccs, 'LFCC Coefficient Index','Frame Index')


from ..features.gfcc import gfcc
# compute gfccs
gfccs = gfcc(sig, 13) 
vis.visualize(gfccs, 'GFCC Coefficient Index','Frame Index')

from ..features.ngcc import ngcc
# compute gfccs
ngccs = ngcc(sig, 13) 
vis.visualize(ngccs, 'NGC Coefficient Index','Frame Index')


from ..features.bfcc import bfcc
# compute bfccs
bfccs = bfcc(sig, 13)
vis.visualize(bfccs, 'BFCC Coefficient Index','Frame Index')

from ..features.pncc import pncc
# compute bfccs
pnccs = pncc(sig, 13)
vis.visualize(pnccs, 'pnccs Coefficient Index','Frame Index')



from ..features.cqcc import cqcc
# compute bfccs
cqccs= cqcc(sig, 13)
vis.plot(cqccs, 'CQC Coefficient Index','Frame Index')




from ..features.plp import plp
# compute bfccs
plps = plp(sig, 13)
vis.visualize(plps, 'PLP Coefficient Index','Frame Index')

from ..features.rplp import rplp
# compute bfccs
rplps = rplp(sig, 13)
vis.visualize(rplps, 'RPLP Coefficient Index','Frame Index')



from ..features.lpc import lpc
from ..features.lsp import lsp
# compute lpcs and lsps
lpcs = lpc(sig, 1)

vis.visualize(lpcs, 'LPC Coefficient Index','Frame Index')

lsps = lsp(lpcs)
vis.visualize(lsps, 'LSP Coefficient Index','Frame Index')
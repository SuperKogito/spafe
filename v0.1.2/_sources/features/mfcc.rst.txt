spafe.features.mfcc
===================


.. automodule:: spafe.features.mfcc
    :members:
    :undoc-members:
    :show-inheritance:

Example:

.. code-block:: python

  import scipy.io.wavfile
  import spafe.utils.vis as vis
  from spafe.features.mfcc import mfcc, imfcc, mfe


  # read wave file
  fs, sig = scipy.io.wavfile.read('../test.wav')

  # compute mfccs and mfes
  mfccs  = mfcc(sig, 13)
  imfccs = imfcc(sig, 13)
  mfes   = mfe(sig, fs)

  # visualize features
  vis.visualize(mfccs, 'MFCC Coefficient Index','Frame Index')
  vis.visualize(imfccs, 'IMFCC Coefficient Index','Frame Index')
  vis.plot(mfes,  'MFE Coefficient Index','Frame Index')


.. image:: images/mfcc.png
 :scale: 100 %
 :align: center


.. image:: images/imfcc.png
  :scale: 100 %
  :align: center


.. image:: images/mfe.png
 :scale: 100 %
 :align: center

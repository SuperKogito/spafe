spafe.features.bfcc
===================


.. automodule:: spafe.features.bfcc
  :members:
  :undoc-members:
  :show-inheritance:

Example:

.. code-block:: python

  import scipy.io.wavfile
  import spafe.utils.vis as vis
  from spafe.features.mfcc import bfcc


  #read wave file
  fs, sig = scipy.io.wavfile.read('../test.wav')

  # compute bfccs
  bfccs  = bfcc(sig, 13)

  # visualize features
  vis.visualize(bfccs, LMFCC Coefficient Index','Frame Index')


.. image:: images/bfcc.png
   :scale: 100 %
   :alt: alternate text
   :align: center

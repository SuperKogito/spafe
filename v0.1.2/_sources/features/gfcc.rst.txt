spafe.features.gfcc
===================


.. automodule:: spafe.features.gfcc
    :members:
    :undoc-members:
    :show-inheritance:

Example:

.. code-block:: python

  import scipy.io.wavfile
  import spafe.utils.vis as vis
  from spafe.features.mfcc import gfcc


  #read wave file
  fs, sig = scipy.io.wavfile.read('../test.wav')

  # compute gfccs
  gfccs  = gfcc(sig, 13)

  # visualize features
  vis.visualize(gfccs, LMFCC Coefficient Index','Frame Index')


.. image:: images/gfcc.png
   :scale: 100 %
   :alt: alternate text
   :align: center

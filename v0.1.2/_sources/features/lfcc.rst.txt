spafe.features.lfcc
===================


.. automodule:: spafe.features.lfcc
    :members:
    :undoc-members:
    :show-inheritance:

Example:

.. code-block:: python

  import scipy.io.wavfile
  import spafe.utils.vis as vis
  from spafe.features.mfcc import lfcc


  #read wave file
  fs, sig = scipy.io.wavfile.read('../test.wav')

  # compute lfccs
  lfccs  = lfcc(sig, 13)

  # visualize features
  vis.visualize(lfccs, LMFCC Coefficient Index','Frame Index')


.. image:: images/lfcc.png
   :scale: 100 %
   :alt: alternate text
   :align: center

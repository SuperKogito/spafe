spafe.features.ngcc
===================


.. automodule:: spafe.features.ngcc
    :members:
    :undoc-members:
    :show-inheritance:

Example:

.. code-block:: python

  import scipy.io.wavfile
  import spafe.utils.vis as vis
  from spafe.features.mfcc import ngcc


  #read wave file
  fs, sig = scipy.io.wavfile.read('../test.wav')

  # compute ngccs
  ngccs  = ngcc(sig, 13)

  # visualize features
  vis.visualize(ngccs, LMFCC Coefficient Index','Frame Index')


.. image:: images/ngcc.png
   :scale: 100 %
   :alt: alternate text
   :align: center

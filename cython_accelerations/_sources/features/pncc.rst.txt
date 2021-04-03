spafe.features.pncc
===================


.. automodule:: spafe.features.pncc
    :members:
    :undoc-members:
    :show-inheritance:

Example:

.. code-block:: python

  import scipy.io.wavfile
  import spafe.utils.vis as vis
  from spafe.features.mfcc import pncc


  #read wave file
  fs, sig = scipy.io.wavfile.read('../test.wav')

  # compute pnccs
  pnccs  = pncc(sig, 13)

  # visualize features
  vis.visualize(pnccs, 'PNCC Index','Frame Index')


.. image:: images/pncc.png
   :scale: 100 %
   :alt: alternate text
   :align: center

spafe.features.rplp
===================


.. automodule:: spafe.features.rplp
    :members:
    :undoc-members:
    :show-inheritance:

Example:

.. code-block:: python

  import scipy.io.wavfile
  import spafe.utils.vis as vis
  from spafe.features.mfcc import plp


  #read wave file
  fs, sig = scipy.io.wavfile.read('../test.wav')

  # compute plps
  plps  = plp(sig, 13)

  # visualize features
  vis.visualize(plps, 'PLP Coefficient Index','Frame Index')


.. image:: images/plp.png
   :scale: 100 %
   :alt: alternate text
   :align: center

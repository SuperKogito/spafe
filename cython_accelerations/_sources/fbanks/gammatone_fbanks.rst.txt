spafe.fbanks.gammatone_fbanks
=============================


.. automodule:: spafe.fbanks.gammatone_fbanks
    :members:
    :undoc-members:
    :show-inheritance:

Example:

.. code-block:: python

  import matplotlib.pyplot as plt
  from spafe.fbanks import gammatone_fbanks

  # compute fbanks
  fbanks = gammatone_fbanks.gammatone_filter_banks(nfilts=24, nfft=512, fs=16000)

  # plot fbanks
  for i in range(len(fbanks)):
      plt.plot(fbanks[i])
      plt.ylim(0, 1.1)
      plt.grid(True)
      plt.ylabel(ylabel)
      plt.xlabel(xlabel)
      plt.show()

.. image:: images/gammatone_fbanks.png
   :scale: 100 %
   :alt: alternate text
   :align: center

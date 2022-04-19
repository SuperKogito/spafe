Welcome to **Spafe**'s  documentation!
========================================

.. image:: _static/logo2.png
   :scale: 15 %
   :align: center


**spafe** (Simplified Python Audio Features Extraction) is a light weight library that aims to unite audio feature extraction algorithms in one simple code.


Dependencies
------------

pydiogment is built using Python3_  and it requires the following:

-	Python packages:
  -	NumPy_ (>= 1.17.2) :  ``pip install numpy``
  -	SciPy_  (>= 1.3.1) :  ``pip install scipy``

Installation
------------

If you already have a working installation of numpy and scipy, you can simply install pydiogment using pip:

  ``pip install spafe``

To update an exisiting pydiogment version use:

  ``pip install -U spafe``

.. _Python3 : https://www.python.org/download/releases/3.0/
..	_NumPy : https://numpy.org/
..	_SciPy : https://scipy.org/
.. _FFmpeg : https://www.ffmpeg.org/


Documentation
-------------

For the documentation of the modules please refer to:


.. toctree::
   :maxdepth: 2
   :caption: Spafe

   api
   exp


Contributors
------------

.. contributors:: superkogito/spafe
   :contributions:
   :limit: 10
   :order: DESC

Citation
--------

  @software{ayoubmalek2020,
    author = {Ayoub Malek},
    title = {spafe/spafe: 0.1.2},
    month = Apr,
    year = 2020,
    version = {0.1.2},
    url = {https://github.com/SuperKogito/spafe}
  }

Welcome to **Spafe**'s  documentation!
========================================

.. image:: _static/logo2.png
   :scale: 10 %
   :align: center

**spafe** (Simplified Python Audio Features Extraction) is a light weight library that aims to unite audio feature extraction algorithms in one simple code.

Dependencies
------------

spafe is built using `Python3 <https://www.python.org/download/releases/3.0/>`__  and it requires the following Python packages:

-  `NumPy <https://numpy.org/>`__ (>= 1.22.0)
-  `SciPy <https://scipy.org/>`__ (>= 1.8.0)

if you want to use the visualization module/ functions of spafe, you
will need to install:

-  `Matplotlib <https://matplotlib.org/>`__ (>= 3.5.2)


Installation
------------

Install from PyPI
~~~~~~~~~~~~~~~~~

- You can simply install spafe using pip: ``pip install spafe``
- To update an existing spafe version use: ``pip install -U spafe``

Install from Anaconda
~~~~~~~~~~~~~~~~~~~~~

-  Spafe is also available on anaconda: ``conda install spafe``


Documentation
-------------

For the documentation of the modules please refer to:

.. toctree::
   :maxdepth: 2

   api_documentation


Citation
--------

-  If you want to cite spafe as a software, please cite the version used as indexed at `Zenodo <https://zenodo.org/>`__: |DOI|

   * *Ayoub Malek, Hadrien Titeux, Stefano Borzì, Christian Heider Nielsen, Fabian-Robert Stöter, Hervé BREDIN, & Kevin Mattheus Moerman. (2023). SuperKogito/spafe: v0.3.2 (v0.3.2). Zenodo.* https://doi.org/10.5281/zenodo.7686438

|

- You can also site spafe's paper on `JOSS <https://https://joss.theoj.org/>`__: |DOI2|

  * *Malek, A., (2023). Spafe: Simplified python audio features extraction. Journal of Open Source Software, 8(81), 4739,* https://doi.org/10.21105/joss.04739




.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7686438.svg
   :target: https://doi.org/10.5281/zenodo.7686438

.. |DOI2| image:: https://joss.theoj.org/papers/10.21105/joss.04739/status.svg
  :target: https://doi.org/10.21105/joss.04739

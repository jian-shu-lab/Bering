Bering - Spatial Segmentation and Cell Annotation in Python
================================================

**Bering** is a tool for the molecular annotation and cell annotation in single-cell spatial transcriptomics data. It is based on the `_torch_geometric`_ package and uses the `scanpy`_ package for the analysis of single-cell transcriptomics data.
It builds on top of `torch_geometric`_ and `scanpy`_, from which it inherits modularity and scalability.
It provides versatile models that leverages the spatial coordinates of the data, as well as pre-trained models across spatial technologies and tissues.

Visit our `documentation`_ for installation, tutorials, examples and more.

Manuscript
----------
We will release our manuscript soon. 

Bering's key applications
--------------------------
- Identify background and real signals in noisy spatial data.
- Identify cell annotations for transcripts on single-cell spatial data.
- Efficiently cell segmentation with cell annotations.
- Build and fine-tune pre-trained model.

Installation
------------
Install Bering via PyPI by running::

    pip install Bering

or via Conda as::

    conda install -c conda-forge Bering

Contributing to Bering
-----------------------
We are happy about any contributions! Before you start, check out our `contributing guide <CONTRIBUTING.rst>`_.


.. _Palla, Spitzer et al. (2022): https://doi.org/10.1038/s41592-021-01358-2
.. _scanpy: https://scanpy.readthedocs.io/en/stable/
.. _torch_geometric: https://pytorch-geometric.readthedocs.io/en/latest/
.. _documentation: https://celldrift.readthedocs.io/en/latest/index.html

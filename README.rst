|PyPI| |Docs|

Bering - Spatial Segmentation and Cell Annotation in Python
================================================

**Bering** is a deep learning algorithm for simultaneous molecular annotation and cell segmentation in single-cell spatial transcriptomics data. 
It builds on top of `torch_geometric`_ and `scanpy`_, from which it inherits modularity and scalability.
It provides versatile models that leverages the spatial coordinates of the data, as well as pre-trained models across spatial technologies and tissues.

Visit our `documentation`_ for installation, tutorials, examples and more.

Manuscript
----------
The manuscript has been submittet for peer review. A preprint will be released soon.

Bering's key applications
--------------------------
- Identify background and real signals in noisy spatial transcriptomics data.
- Identify cell annotations for transcripts on single-cell spatial data.
- Efficiently cell segmentation with cell annotations.
- Build and fine-tune pre-trained model on new data using transfer learning.

Installation
------------
Install Bering via PyPI by running::

    pip install Bering

or via Conda as::

    conda install -c conda-forge Bering

Contact
-----------------------
We are happy about any feedback! If you have any questions, please feel free to contact Kang.Jin@cchmc, zuobai.zhang@mila.quebec.
Find more research in `Shu_Jian_Lab`_.

.. |PyPI| image:: https://img.shields.io/pypi/v/squidpy.svg
    :target: https://pypi.org/project/Bering/
    :alt: PyPI

.. |Docs| image:: https://img.shields.io/readthedocs/squidpy
    :target: https://bering.readthedocs.io/en/latest/
    :alt: Documentation

.. _Palla, Spitzer et al. (2022): https://doi.org/10.1038/s41592-021-01358-2
.. _scanpy: https://scanpy.readthedocs.io/en/stable/
.. _torch_geometric: https://pytorch-geometric.readthedocs.io/en/latest/
.. _documentation: https://bering.readthedocs.io/en/latest/
.. _Shu_Jian_Lab: https://www.jianshulab.org/team
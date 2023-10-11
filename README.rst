|PyPI| |Docs| |downloads|

Bering - Spatial Segmentation and Cell Annotation in Python
================================================

**Bering** is a deep learning algorithm for simultaneous molecular annotation and cell segmentation in single-cell spatial transcriptomics data. 
It builds on top of `torch_geometric`_ and `scanpy`_, from which it inherits modularity and scalability.
It provides versatile models that leverages the spatial coordinates of the data, as well as pre-trained models across spatial technologies and tissues.

Visit our `documentation`_ for installation, tutorials, examples and more.

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

Manuscript
----------
Please refer to our manuscript `Jin, Zhang et al. (2023)`_ for more details.

Contact
-----------------------
We are happy about any feedback! If you have any questions, please feel free to contact Kang.Jin@cchmc, zuobai.zhang@mila.quebec.
Find more research in `Shu_Jian_Lab`_.

.. |PyPI| image:: https://img.shields.io/pypi/v/Bering
    :target: https://pypi.org/project/Bering/
    :alt: PyPI

.. |Docs| image:: https://img.shields.io/readthedocs/bering
    :target: https://bering.readthedocs.io/en/latest/
    :alt: Documentation

.. |downloads| image:: https://img.shields.io/pepy/dt/Bering
    :target: https://www.pepy.tech/projects/Bering
    :alt: Downloads

.. _Jin, Zhang et al. (2023): https://www.biorxiv.org/content/10.1101/2023.09.19.558548v1
.. _scanpy: https://scanpy.readthedocs.io/en/stable/
.. _torch_geometric: https://pytorch-geometric.readthedocs.io/en/latest/
.. _documentation: https://bering.readthedocs.io/en/latest/
.. _Shu_Jian_Lab: https://www.jianshulab.org/team
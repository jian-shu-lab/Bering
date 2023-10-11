Bering - Segmentaion of Spatial Single Cell in Python
================================================

**Bering** is a tool for the segmentation on single-cell spatial data.
It builds on top of `torch_geometric`_ and `scanpy`_, from which it inherits modularity and scalability.
It provides analysis tools that leverages the spatial coordinates of the data, as well as
tissue images if available.

.. image:: ./_static/img/figure1.png
    :alt: Bering title figure
    :width: 1000px
    :align: center
    :target: https://github.com/KANG-BIOINFO

Manuscript
----------
Please refer to our manuscript ([Jin, Zhang et al., 2023, bioRxiv](https://www.biorxiv.org/content/10.1101/2023.09.19.558548v1)) for more details.

Bering's key applications
--------------------------
- Identify background and real signals in noisy spatial data.
- Identify cell annotations for transcripts on single-cell spatial data.
- Efficiently cell segmentation with cell annotations.
- Build and fine-tune pre-trained model on new data.

Getting started with Bering
----------------------------
- Browse :doc:`notebooks/tutorials/index` and :doc:`notebooks/examples/index`.
- Discuss usage on `discourse`_ and development on `github`_.

Contributing to Bering
-----------------------
We are happy about any contributions! Before you start, check out our `contributing guide`_.

.. toctree::
    :caption: General
    :maxdepth: 2
    :hidden:

    installation
    api
    classes
    release_notes
    references

.. toctree::
    :caption: Gallery
    :maxdepth: 2
    :hidden:

    notebooks/tutorials/index
    notebooks/examples/index

.. _scanpy: https://scanpy.readthedocs.io/en/stable/
.. _torch_geometric: https://pytorch-geometric.readthedocs.io/en/latest/
.. _contributing guide: https://github.com/KANG-BIOINFO
.. _discourse: https://github.com/jian-shu-lab/Bering/issues
.. _github: https://github.com/jian-shu-lab/Bering

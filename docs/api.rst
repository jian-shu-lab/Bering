API
===
Import Bering as::

    import Bering as br

Objects
~~~~~

.. module:: Bering.objects
.. currentmodule:: Bering

.. autosummary::
    :toctree: api

    objects.Bering_Graph


Preprocess
~~~~~

.. module:: Bering.preprocess
.. currentmodule:: Bering

.. autosummary::
    :toctree: api
    
    preprocess.add_image_features

Graph
~~~~~

.. module:: Bering.graphs
.. currentmodule:: Bering

.. autosummary::
    :toctree: api

    graphs.BuildGraph
    graphs.BuildGraph_fromRaw
    graphs.BuildWindowGraphs
    graphs.CreateData

Models
~~~~~

.. module:: Bering.models
.. currentmodule:: Bering

.. autosummary::
    :toctree: api

    models.GCN
    models.BaselineMLP
    models.GaussianSmearing
    models.ImageEncoder
    models.EdgeClf


Training
~~~~~

.. module:: Bering.training
.. currentmodule:: Bering

.. autosummary::
    :toctree: api

    training.TrainerNode
    training.TrainerEdge
    training.Training
    training.EarlyStopper

Datasets
~~~~~

.. module:: Bering.datasets
.. currentmodule:: Bering

.. autosummary::
    :toctree: api

    datasets.merfish_cortex_zhang
    datasets.cosmx_nsclc_he
    datasets.iss_ca1_qian
    datasets.merfish_ileum_petukhov
    datasets.stereoseq_embryobrain_chen
    datasets.xenium_dcis_janesick

Pretrained Models
~~~~~

.. module:: Bering.datasets
.. currentmodule:: Bering

.. autosummary::
    :toctree: api

    datasets.model_merfish_cortex_zhang
    datasets.model_cosmx_nsclc_he
    datasets.model_iss_ca1_qian
    datasets.model_xenium_dcis_janesick

Plotting
~~~~~

.. module:: Bering.plotting
.. currentmodule:: Bering

.. autosummary::
    :toctree: api

    plotting.Plot_SliceImages
    plotting.Plot_Classification
    plotting.Plot_Segmentation
    plotting.Plot_Segmentation_Post
    plotting.Plot_Classification_Post


Utilities
~~~~~

.. module:: Bering.tools
.. currentmodule:: Bering

.. autosummary::
    :toctree: api

    tools.node_classification
    tools.cell_segmentation
    tools.cell_annotation
    tools.cell_analyze
from copy import copy
from typing import NamedTuple
import pandas 

from ._utils import AMetadata

model_merfish_cortex_zhang = AMetadata(
    name="model_merfish_cortex_zhang",
    doc_header="Pre-trained model of MERFISH Cortex dataset from `Zhang et al <https://doi.org/10.1126/science.aar7042>`__.",
    url = "https://figshare.com/ndownloader/files/42109140",
    category = 'models',
)

model_cosmx_nsclc_he = AMetadata(
    name="model_cosmx_nsclc_he",
    doc_header="Pre-trained model of Nanostring ung cancer dataset from `He et al <https://www.nature.com/articles/s41587-022-01483-z>`__.",
    url = "https://figshare.com/ndownloader/files/42109134",
    category = 'models',
)

model_iss_ca1_qian = AMetadata(
    name="model_iss_ca1_qian",
    doc_header="Pre-trained model of ISS CA1 hippocampus data (sample 3-1 left) from `Qian et al <https://www.nature.com/articles/s41592-019-0631-4>`__.",
    url = "https://figshare.com/ndownloader/files/42109137",
    category = 'models',
)

# model_merfish_ileum_petukhov = AMetadata(
#     name="merfish_ileum_petukhov",
#     doc_header="Pre-trained model of MERFISH ileum dataset from `Petukhov et al <https://www.nature.com/articles/s41587-021-01044-w>`__.",
#     url = "https://figshare.com/ndownloader/files/41409102",
#     category = 'models',
# )

# model_stereoseq_embryobrain_chen = AMetadata(
#     name="stereoseq_embryobrain_chen",
#     doc_header="Pre-trained model of Stereo-seq embryo brain dataset from `Chen et al <https://www.sciencedirect.com/science/article/pii/S0092867422003993?via%3Dihub>`__.",
#     url = "https://figshare.com/ndownloader/files/41409105",
# )

model_xenium_dcis_janesick = AMetadata(
    name="model_xenium_dcis_janesick",
    doc_header="Pre-trained model of Xenium breast cancer dataset from `Janesick et al <https://www.biorxiv.org/content/10.1101/2022.10.06.510405v2>`__.",
    url = "https://figshare.com/ndownloader/files/42109143",
    category = 'models',
)

for name, var in copy(locals()).items():
    if isinstance(var, AMetadata):
        var._create_function(name, globals())

__all__ = [  # noqa: F822
    "model_merfish_cortex_zhang",
    "model_cosmx_nsclc_he",
    "model_iss_ca1_qian",
    # "model_merfish_ileum_petukhov",
    # "model_stereoseq_embryobrain_chen",
    "model_xenium_dcis_janesick",
]
from copy import copy
import pandas 

from ._utils import AMetadata

merfish_cortex_zhang = AMetadata(
    name="merfish_cortex_zhang",
    doc_header="Pre-processed MERFISH dataset from `Zhang et al <https://doi.org/10.1126/science.aar7042>`__.",
    url = "https://figshare.com/ndownloader/files/41409090",
)

cosmx_nsclc_he = AMetadata(
    name="cosmx_nsclc_he",
    doc_header="Pre-processed Nanostring CoxMx dataset from `He et al <https://www.nature.com/articles/s41587-022-01483-z>`__.",
    url = "https://figshare.com/ndownloader/files/41409093",
)

iss_ca1_qian = AMetadata(
    name="iss_ca1_qian",
    doc_header="Pre-processed ISS CA1 (sample 3-1 left) from `Qian et al <https://www.nature.com/articles/s41592-019-0631-4>`__.",
    url = "https://figshare.com/ndownloader/files/41409096",
)

merfish_ileum_petukhov = AMetadata(
    name="merfish_ileum_petukhov",
    doc_header="Pre-processed MERFISH dataset from `Petukhov et al <https://www.nature.com/articles/s41587-021-01044-w>`__.",
    url = "https://figshare.com/ndownloader/files/41409102",
)

stereoseq_embryobrain_chen = AMetadata(
    name="stereoseq_embryobrain_chen",
    doc_header="Pre-processed Stereo-seq dataset from `Chen et al <https://www.sciencedirect.com/science/article/pii/S0092867422003993?via%3Dihub>`__.",
    url = "https://figshare.com/ndownloader/files/41409105",
)

xenium_dcis_janesick = AMetadata(
    name="xenium_dcis_janesick",
    doc_header="Pre-processed Xenium dataset from `Janesick et al <https://www.biorxiv.org/content/10.1101/2022.10.06.510405v2>`__.",
    url = "https://figshare.com/ndownloader/files/41409108",
)

for name, var in copy(locals()).items():
    if isinstance(var, AMetadata):
        var._create_function(name, globals())

__all__ = [  # noqa: F822
    "merfish_cortex_zhang",
    "cosmx_nsclc_he",
    "iss_ca1_qian",
    "merfish_ileum_petukhov",
    "stereoseq_embryobrain_chen",
    "xenium_dcis_janesick",
]

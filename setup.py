import os
from setuptools import setup
from setuptools import find_packages

# build long description
base_dir = os.path.dirname(os.path.abspath(__file__))
long_description = '\n\n'.join([open(os.path.join(base_dir,'README.rst'),'r').read()])

setup(
    name = 'Bering',
    version = '0.1.2',
    description = 'Bering: Transfer Learning of Cell Segmentation and Annotation for Spatial Omics',
    author = 'Kang Jin',
    author_email = 'kangjin@fas.harvard.edu',
    maintainer = 'Kang Jin',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/KANG-BIOINFO/Bering',
    project_urls={
        'Ducumentation':'https://celldrift.readthedocs.io/en/latest/index.html#',
    },
    packages = find_packages(),
    install_requires = [
        'numpy',
        'pandas',
        'anndata>=0.8.0',
        'scanpy>=1.6.0',
        'matplotlib',
        'scipy',
        'scikit-learn',
        'igraph',
        'leidenalg',
        'torch>=1.12.1',
        'torch_geometric>=2.1.0',
    ],
)
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from inspect import Parameter, Signature, signature
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from pandas import read_csv, DataFrame

PathLike = Union[os.PathLike, str]


@dataclass(frozen=True)
class Metadata(ABC):
    """Base class handling metadata."""

    name: str
    url: str

    category: Optional[str] = 'datasets'
    doc_header: Optional[str] = field(default=None, repr=False)
    path: Optional[PathLike] = field(default=None, repr=False)
    shape: Optional[Tuple[int, int]] = field(default=None, repr=False)
    library_id: Optional[Union[str, Sequence[str]]] = field(default=None, repr=False)

    _DOC_FMT = ""

    def __post_init__(self) -> None:
        if self.doc_header is None:
            object.__setattr__(self, "doc_header", f"Download `{self.name.title().replace('_', ' ')}` data.")
        # if self.path is None:
        #     object.__setattr__(self, "path", os.path.expanduser(f"~/.cache/squidpy/{self.name}"))

    @property
    @abstractmethod
    def _extension(self) -> str:
        pass

    @abstractmethod
    def _download(
        self, 
        fpath: PathLike, 
        backup_url: str, 
        category: str,
        **kwargs: Any
    ) -> Any:
        pass

    @abstractmethod
    def _create_signature(self) -> Signature:
        pass

    def _create_function(self, name: str, glob_ns: Dict[str, Any]) -> None:
        if name in globals():
            raise KeyError(f"Function name `{name}` is already present in `{sorted(globals().keys())}`.")

        sig = self._create_signature()
        globals()["NoneType"] = type(None)  # __post_init__ return annotation
        globals()[name] = self

        exec(
            f"def {self.name}{sig}:\n"
            f'    """'
            f"    {self._DOC_FMT.format(doc_header=self.doc_header, shape=self.shape)}"
            f'    """\n'
            f"    return {name}.download(path, **kwargs)".replace(" /,", ""),
            globals(),
            glob_ns,
        )

    def download(
        self, 
        fpath: Optional[PathLike] = None, 
        category: str = 'datasets',
        **kwargs: Any
    ) -> Any:
        """Download the dataset into ``fpath``."""
        fpath = str(self.path if fpath is None else fpath)
        if not fpath.endswith(self._extension):
            fpath += self._extension

        if os.path.isfile(fpath):
            print(f"Loading dataset `{self.name}` from `{fpath}`")
        else:
            print(f"Downloading dataset `{self.name}` from `{self.url}` as `{fpath}`")

        dirname = Path(fpath).parent
        try:
            if not dirname.is_dir():
                print(f"Creating directory `{dirname}`")
                dirname.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Unable to create directory `{dirname}`. Reason `{e}`")

        spots = self._download(fpath=fpath, backup_url=self.url, category = category, **kwargs)

        # if self.shape is not None and data.shape != self.shape:
        #     raise ValueError(f"Expected the data to have number of spots (seg, unseg) `{self.shape}`, found `{data.shape}`.")

        # return data
        return spots


class AMetadata(Metadata):
    """Metadata class for :class:`anndata.AnnData`."""

    _DOC_FMT = """
    {doc_header}

    The shape of this :class:`anndata.AnnData` object ``{shape}``.

    Parameters
    ----------
    path
        Path where to save the dataset.
    kwargs
        Keyword arguments for :func:`scanpy.read`.

    Returns
    -------
    The dataset."""

    def _create_signature(self) -> Signature:
        return signature(lambda _: _).replace(
            parameters=[
                Parameter("path", kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=PathLike, default=None),
                Parameter("kwargs", kind=Parameter.VAR_KEYWORD, annotation=Any),
            ],
            # return_annotation=anndata.AnnData,
            return_annotation = DataFrame,
        )

    def _download(self, fpath: PathLike, backup_url: str, category: str = 'datasets', **kwargs: Any) -> Tuple[DataFrame, DataFrame]:
        kwargs.setdefault("keep_default_na", False)
        # kwargs.setdefault("cache", True)
        # kwargs.setdefault("sparse", True)
        # kwargs.setdefault("cache", True)

        if category == 'datasets':
            if fpath != 'None.tsv':
                results = read_csv(fpath, sep = '\t', header = 0, index_col = 0, **kwargs)
            else:
                results = read_csv(backup_url, sep = '\t', header = 0, index_col = 0, **kwargs)
            # spots = read_csv(backup_url, sep = '\t', header = 0, index_col = 0, **kwargs)
        else:
            with open(backup_url, 'rb') as f:
                results = pickle.load(f)
        return results

    @property
    def _extension(self) -> str:
        # return ".h5ad"
        return ".tsv"


# class ImgMetadata(Metadata):
#     """Metadata class for :class:`squidpy.im.ImageContainer`."""

#     _DOC_FMT = """
#     {doc_header}

#     The shape of this image is ``{shape}``.

#     Parameters
#     ----------
#     path
#         Path where to save the .tiff image.
#     kwargs
#         Keyword arguments for :meth:`squidpy.im.ImageContainer.add_img`.

#     Returns
#     -------
#     :class:`squidpy.im.ImageContainer` The image data."""
#     # not the perfect annotation, but better than nothing
#     _EXT = ".tiff"

#     def _create_signature(self) -> Signature:
#         return signature(lambda _: _).replace(
#             parameters=[
#                 Parameter("path", kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=PathLike, default=None),
#                 Parameter("kwargs", kind=Parameter.VAR_KEYWORD, annotation=Any),
#             ],
#         )

#     def _download(self, fpath: PathLike, backup_url: str, **kwargs: Any) -> Any:
#         from squidpy.im import ImageContainer  # type: ignore[attr-defined]

#         # check_presence_download(Path(fpath), backup_url)

#         img = ImageContainer()
#         img.add_img(fpath, layer="image", library_id=self.library_id, **kwargs)

#         return img

#     @property
#     def _extension(self) -> str:
#         return ".tiff"

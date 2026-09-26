"""PDB file parsing using pandas DataFrames"""

# Add imports here
from .Scene import Scene
from .transformation import Transformation
from .matching import (
    Matching,
    OrderMatching,
    ColumnMatching,
    SequenceMatching,
    as_matching,
)
from . import contacts  # distance-map / virtual-Cβ geometry kernels
from . import geometry, sparse
from .sparse import SparseMatrix
from . import parsers, backends  # register file-format parsers and object backends

from ._version import __version__

__all__ = [
    "Scene",
    "Transformation",
    "Matching",
    "OrderMatching",
    "ColumnMatching",
    "SequenceMatching",
    "as_matching",
    "contacts",
    "geometry",
    "sparse",
    "SparseMatrix",
    "parsers",
    "backends",
    "__version__",
]

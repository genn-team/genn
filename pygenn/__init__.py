# python imports
import sys

# pygenn interface
from .genn import (PlogSeverity, SynapseMatrixType, VarAccess,
                   VarAccessMode, VarLocation, ScalarPrecision, 
                   SpanType, TimePrecision)
from .genn_model import (GeNNModel, init_sparse_connectivity, 
                         init_toeplitz_connectivity, init_var)

if sys.version_info >= (3, 8):
    from importlib import metadata
else:
    import importlib_metadata as metadata

__version__ = metadata.version("pygenn")

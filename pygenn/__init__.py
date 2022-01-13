# python imports
import sys

# pygenn interface
from .genn import (create_var_ref, create_psm_var_ref, create_wu_pre_var_ref,
                   create_wu_post_var_ref, create_wu_var_ref, PlogSeverity, 
                   ScalarPrecision, SpanType, SynapseMatrixType, TimePrecision,
                   VarAccess, VarAccessMode, VarLocation)
from .genn_model import (GeNNModel, init_sparse_connectivity, 
                         init_toeplitz_connectivity, init_var)

if sys.version_info >= (3, 8):
    from importlib import metadata
else:
    import importlib_metadata as metadata

__version__ = metadata.version("pygenn")

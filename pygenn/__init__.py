# python imports
import sys

# pygenn interface
from .genn import (create_var_ref, create_psm_var_ref, create_wu_pre_var_ref,
                   create_wu_post_var_ref, create_wu_var_ref, create_egp_ref,
                   create_psm_egp_ref, create_wu_egp_ref, 
                   CustomUpdateVarAccess, PlogSeverity, SpanType,
                   SynapseMatrixType, VarAccess, VarAccessMode, VarLocation)
from .genn_model import (GeNNModel, create_neuron_model,
                         create_postsynaptic_model,
                         create_weight_update_model,
                         create_current_source_model,
                         create_custom_update_model,
                         create_custom_connectivity_update_model,
                         create_var_init_snippet,
                         create_sparse_connect_init_snippet,
                         create_toeplitz_connect_init_snippet,
                         init_sparse_connectivity, 
                         init_toeplitz_connectivity, init_var)

if sys.version_info >= (3, 8):
    from importlib import metadata
else:
    import importlib_metadata as metadata

__version__ = metadata.version("pygenn")

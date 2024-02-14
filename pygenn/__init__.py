# python imports
import sys

# pygenn interface
from ._genn import (create_var_ref, create_psm_var_ref, create_wu_pre_var_ref,
                    create_wu_post_var_ref, create_wu_var_ref, create_egp_ref,
                    create_psm_egp_ref, create_wu_egp_ref, get_var_access_dim,
                    CurrentSource, CustomConnectivityUpdate, CustomUpdate,
                    CustomUpdateWU, CustomUpdateVarAccess, NeuronGroup,
                    ParallelismHint,PlogSeverity, SynapseGroup, SynapseMatrixType, 
                    SynapseMatrixConnectivity, SynapseMatrixWeight, VarAccess, 
                    VarAccessDim, VarAccessMode, VarAccessModeAttribute, VarLocation)
from ._genn_model import (GeNNModel, create_neuron_model,
                          create_postsynaptic_model,
                          create_weight_update_model,
                          create_current_source_model,
                          create_custom_update_model,
                          create_custom_connectivity_update_model,
                          create_var_init_snippet,
                          create_sparse_connect_init_snippet,
                          create_toeplitz_connect_init_snippet,
                          init_postsynaptic, init_sparse_connectivity,
                          init_toeplitz_connectivity, init_var,
                          init_weight_update)

__all__ = ["create_current_source_model", "create_custom_update_model",
           "create_custom_connectivity_update_model", "create_neuron_model",
           "create_postsynaptic_model", "create_psm_var_ref", 
           "create_sparse_connect_init_snippet", "create_toeplitz_connect_init_snippet",
           "create_var_ref", "create_weight_update_model", "create_wu_pre_var_ref",
           "create_wu_post_var_ref", "create_wu_var_ref", "create_egp_ref",
           "create_psm_egp_ref", "create_var_init_snippet", "create_wu_egp_ref", 
           "get_var_access_dim", "init_postsynaptic", "init_sparse_connectivity",
           "init_toeplitz_connectivity", "init_var", "init_weight_update",
           "CurrentSource", "CustomConnectivityUpdate", "CustomUpdate",
           "CustomUpdateWU", "CustomUpdateVarAccess", "GeNNModel", 
           "NeuronGroup", "ParallelismHint", "PlogSeverity", "SynapseGroup",
           "SynapseMatrixType", "SynapseMatrixConnectivity",
           "SynapseMatrixWeight", "VarAccess", "VarAccessDim",
           "VarAccessMode", "VarAccessModeAttribute", "VarLocation"]

if sys.version_info >= (3, 8):
    from importlib import metadata
else:
    import importlib_metadata as metadata

__version__ = metadata.version("pygenn")

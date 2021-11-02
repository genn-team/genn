# python imports
import sys

# pygenn interface
from .genn_groups import SynapseGroup, NeuronGroup, CurrentSource, CustomUpdate
from .genn_model import GeNNModel

if sys.version_info >= (3, 8):
    from importlib import metadata
else:
    import importlib_metadata as metadata

__version__ = metadata.version("pygenn")
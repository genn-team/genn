import re
import sys
import numpy as np
from . import (current_source_models, custom_connectivity_update_models,
               custom_update_models, init_sparse_connectivity_snippets, 
               init_toeplitz_connectivity_snippets, init_var_snippets,
               neuron_models, postsynaptic_models, types, weight_update_models)


from collections import OrderedDict
from typing import Callable, Dict, Optional, List, Sequence, Tuple, Union
from ._genn import (CurrentSource, CurrentSourceModelBase,
                    CustomConnectivityUpdate, CustomConnectivityUpdateModelBase,
                    CustomUpdate, CustomUpdateModelBase, CustomUpdateVar,
                    CustomUpdateVarAccess, CustomUpdateWU, DerivedParam,
                    EGP, EGPRef, EGPReference, InitSparseConnectivitySnippetBase,
                    InitToeplitzConnectivitySnippetBase, InitVarSnippetBase,
                    ModelSpec, NeuronGroup, NeuronModelBase,
                    NumericValue, Param, ParamVal, PlogSeverity,
                    PostsynapticInit, PostsynapticModelBase, ResolvedType,
                    SparseConnectivityInit, SynapseGroup, SynapseMatrixType,
                    ToeplitzConnectivityInit, UnresolvedType, Var, VarAccess,
                    VarAccessMode, VarInit, VarLocation, VarRef, VarReference,
                    WeightUpdateInit, WeightUpdateModelBase, WUVarReference)
from ._runtime import Runtime
from .genn_groups import (CurrentSourceMixin, CustomConnectivityUpdateMixin,
                          CustomUpdateMixin, CustomUpdateWUMixin,
                          NeuronGroupMixin, SynapseGroupMixin)

from importlib import import_module
from os import path, environ
from platform import system
from psutil import cpu_count
from shutil import which
from subprocess import check_call  # to call make
from textwrap import dedent
from warnings import warn
from weakref import proxy
from ._genn import generate_code, init_logging
from ._deprecated import deprecated
from .model_preprocessor import _get_snippet, _get_var_init, _prepare_param_vals

# Type aliases used in model creation functions
TypeType = Union[str, ResolvedType]
ModelParamsType = Optional[Sequence[Union[str, Tuple[str, TypeType]]]]
ModelDerivedParamsType = Optional[Sequence[Tuple[str, Callable, TypeType]]]
ModelVarsType = Optional[Sequence[Union[Tuple[str, TypeType],
                                        Tuple[str, TypeType, VarAccess]]]]
CUModelVarsType = Optional[Sequence[Union[Tuple[str, TypeType],
                                          Tuple[str, TypeType, CustomUpdateVarAccess]]]]
ModelVarRefsType = Optional[Sequence[Union[Tuple[str, TypeType],
                                          Tuple[str, TypeType, VarAccessMode]]]]
ModelEGPType = Optional[Sequence[Tuple[str, TypeType]]]

# Type aliases used in add_XXX methods
PopParamVals = Dict[str, Union[int, float]]
PopVarVals = Dict[str, Union[VarInit, int, float, np.ndarray, Sequence]]
PopVarRefs = Dict[str, VarReference] 
PopLocalVarRefs = Dict[str, Union[VarReference, str]] 
PopWUVarRefs = Dict[str, WUVarReference]
PopEGPRefs = Dict[str, EGPReference]

# Dynamically add Python mixin to wrapped class
CurrentSource.__bases__ += (CurrentSourceMixin,)
CustomConnectivityUpdate.__bases__ += (CustomConnectivityUpdateMixin,)
CustomUpdate.__bases__ += (CustomUpdateMixin,)
CustomUpdateWU.__bases__ += (CustomUpdateWUMixin,)
NeuronGroup.__bases__ += (NeuronGroupMixin,)
SynapseGroup.__bases__ += (SynapseGroupMixin,)

# If we're on windows
if system() == "Windows":
    # Try import the helper to get Visual C++ environment from setuptools
    try:
        from setuptools.msvc import msvc14_get_vc_env as _get_vc_env
    # Setuptools 0.74.0 removed this function 
    except ImportError:
        try:
            from distutils._msvccompiler import _get_vc_env
        # Things keep moving around in distutils/setuptools 
        except ImportError:
            from distutils.compilers.C.msvc import _get_vc_env
    
    # Get environment and cache in class, convertings
    # all keys to upper-case for consistency
    _msvc_env = _get_vc_env("x86_amd64")
    _msvc_env = {k.upper(): v for k, v in _msvc_env.items()}
    
    # Update process's environment with this
    # **NOTE** this handles both child processes (manually launching msbuild)
    # and stuff within this process (running the code generator)
    environ.update(_msvc_env)
    
    # Find MSBuild in path
    # **NOTE** we need to do this because setting the path via 
    # check_call's env kwarg does not effect finding the executable
    _msbuild = which("msbuild",  path=_msvc_env["PATH"])

    # If Python version is newer than 3.8 and CUDA path is in environment
    if sys.version_info >= (3, 8) and "CUDA_PATH" in environ:
        # Add CUDA bin directory to DLL search directories
        from os import add_dll_directory
        add_dll_directory(path.join(environ["CUDA_PATH"], "bin"))


# Loop through backends in preferential order
backend_modules = OrderedDict()
for b in ["cuda", "hip", "single_threaded_cpu"]:
    # Try and import
    try:
        m = import_module("." + b + "_backend", "pygenn")
    # Ignore failed imports - likely due to non-supported backends
    except ImportError as ex:
        pass
    # Raise any other errors
    except:
        raise
    # Otherwise add to (ordered) dictionary
    else:
        backend_modules[b] = m

# Regular expressions used for upgrading function calls and variables in code strings
_code_upgrades = [
    (re.compile(r"\$\(gennrand_uniform\)"), r"gennrand_uniform()"),
    (re.compile(r"\$\(gennrand_normal\)"), r"gennrand_normal()"),
    (re.compile(r"\$\(gennrand_exponential\)"), r"gennrand_exponential()"),
    (re.compile(r"\$\(gennrand_log_normal,(.*)\)"), r"gennrand_log_normal(\1)"),
    (re.compile(r"\$\(gennrand_gamma,(.*)\)"), r"gennrand_gamma(\1)"),
    (re.compile(r"\$\(gennrand_binomial,(.*)\)"), r"gennrand_binomial(\1)"),
    (re.compile(r"\$\(addToPre,(.*)\)"), r"addToPre(\1)"),
    (re.compile(r"\$\(addToInSyn,(.*)\)"), r"addToPost(\1)"),
    (re.compile(r"\$\(addToInSynDelay,(.*),(.*)\)"), r"addToPostDelay(\1, \2)"),
    (re.compile(r"\$\(addSynapse,(.*)\)"), r"addSynapse(\1)"),
    (re.compile(r"\$\(sT_pre\)"), r"st_pre"),
    (re.compile(r"\$\(sT_post\)"), r"st_post"),
    (re.compile(r"\$\(seT_pre\)"), r"set_pre"),
    (re.compile(r"\$\(seT_post\)"), r"set_post"),
    (re.compile(r"\$\(prev_sT_pre\)"), r"prev_st_pre"),
    (re.compile(r"\$\(prev_sT_post\)"), r"prev_st_post"),
    (re.compile(r"\$\(prev_seT_pre\)"), r"prev_set_pre"),
    (re.compile(r"\$\(prev_seT_post\)"), r"prev_set_post"),
    (re.compile(r"\$\(endRow\)"), None),
    (re.compile(r"\$\(endCol\)"), None)]

# Helpers to wrap lambda function so as to extract
# underlying value from NumericValue parameters!
# **NOTE** seperate function is required to ensure f is bound correctly
def _wrap_max_length_lambda(f):
    return lambda num_pre, num_post, pars: f(num_pre, num_post,
                                             {n: p.value 
                                             for n, p in pars.items()})

def _wrap_kernel_size_lambda(f):
    return lambda pars: f({n: p.value for n, p in pars.items()})

# Regular expression used for upgrading remaining variable references in code strings
_var_upgrade = re.compile(r"\$\(([_a-zA-Z][_a-zA-Z0-9]*)\)")

class GeNNModel(ModelSpec):
    """This class provides an interface for 
    defining, building and running models
    
    Args:
        precision:              Data type to use for ``scalar`` variables
        model_name:             Name of the model
        backend:                Name of backend module to use. Currently 
                                supported "single_threaded_cpu", "cuda". 
                                Defaults to automatically picking the 'best'
                                backend for your system
        time_precision:         data type to use for representing time
        genn_log_level:         Log level for GeNN
        code_gen_log_level:     Log level for GeNN code-generator
        transpiler_log_level:   Log level for GeNN transpiler
        runtime_log_level:      Log level for GeNN runtime
        backend_log_level:      Log level for backend
        preference_kwargs:      Additional keyword arguments to set in backend preferences structure
    """

    def __init__(self, precision: TypeType = "float",
                 model_name: str = "GeNNModel",
                 backend: Optional[str] = None, 
                 time_precision: Optional[TypeType] = None,
                 genn_log_level: PlogSeverity = PlogSeverity.WARNING,
                 code_gen_log_level: PlogSeverity = PlogSeverity.WARNING,
                 transpiler_log_level: PlogSeverity = PlogSeverity.WARNING,
                 runtime_log_level: PlogSeverity = PlogSeverity.WARNING,
                 backend_log_level: PlogSeverity = PlogSeverity.WARNING,
                 **preference_kwargs):
        # Superclass
        super(GeNNModel, self).__init__()

        # Set precision
        self.precision = UnresolvedType(precision)
        
        # Based on time precision, create correct type 
        # of SLM class and determine GeNN time type 
        # **NOTE** all SLM uses its template parameter for is time variable
        self.time_precision = UnresolvedType(self.precision 
                                             if time_precision is None
                                             else time_precision)

        # Initialise GeNN logging
        init_logging(genn_log_level, code_gen_log_level, 
                     transpiler_log_level, runtime_log_level)
        
        self._built = False
        self._loaded = False
        self._runtime = None
        self._preferences = None
        self._model_merged = None
        self._backend = None
        self.backend_name = backend
        self._preference_kwargs = preference_kwargs
        self.backend_log_level = backend_log_level

        # Set model properties
        self.name = model_name

        # Python-side dictionaries of populations
        self.neuron_populations = {}
        self.synapse_populations = {}
        self.current_sources = {}
        self.custom_connectivity_updates = {}
        self.custom_updates = {}

        # Build dictionary containing conversions 
        # between GeNN C++ types and numpy types
        self.genn_types = {
            types.Float:    np.float32,
            types.Double:   np.float64,
            types.Int64:    np.int64,
            types.Uint64:   np.uint64,
            types.Int32:    np.int32,
            types.Uint32:   np.uint32,
            types.Int16:    np.int16,
            types.Uint16:   np.uint16,
            types.Int8:     np.int8,
            types.Uint8:    np.uint8,
            types.Bool:     np.bool_}

    @property
    def backend_name(self) -> str:
        """Name of the currently selected backend"""
        return self._backend_name

    @backend_name.setter
    def backend_name(self, backend_name: str):
        if self._built:
            raise Exception("GeNN model already built")

        # If no backend is specified
        if backend_name is None:
            # Check we have managed to import any bagenn_wrapperckends
            assert len(backend_modules) > 0

            # Set name to first (i.e. best) backend and lookup module from dictionary
            self._backend_name = next(iter(backend_modules))
            self._backend_module = backend_modules[self._backend_name]
        else:
            self._backend_name = backend_name
            self._backend_module = backend_modules[backend_name]
    
    @property
    @deprecated("The name of this property was inconsistent, use dt instead")
    def dT(self):
        return self.dt
    
    @dT.setter
    @deprecated("The name of this property was inconsistent, use dt instead")
    def dT(self, dt):
        self.dt = dt
    
    @property
    def t(self) -> float:
        """Simulation time in ms"""
        return self._runtime.time

    @property
    def timestep(self) -> int:
        """Simulation time step"""
        return self._runtime.timestep

    @timestep.setter
    def timestep(self, timestep: int):
        """Simulation time in timesteps"""
        self._runtime.timestep = timestep

    #@property
    #def free_device_mem_bytes(self):
    #    return self._runtime.free_device_mem_bytes;

    @property
    def neuron_update_time(self) -> float:
        """Time in seconds spent in neuron update kernel.
        Only available if :attr:`.ModelSpec.timing_enabled` is set """
        return self._runtime.neuron_update_time

    @property
    def init_time(self) -> float:
        """Time in seconds spent initialisation kernel.
        Only available if :attr:`.ModelSpec.timing_enabled` is set """
        return self._runtime.init_time

    @property
    def presynaptic_update_time(self) -> float:
        """Time in seconds spent in presynaptic update kernel.
        Only available if :attr:`.ModelSpec.timing_enabled` is set """
        return self._runtime.presynaptic_update_time

    @property
    def postsynaptic_update_time(self) -> float:
        """Time in seconds spent in postsynaptic update kernel.
        Only available if :attr:`.ModelSpec.timing_enabled` is set """
        return self._runtime.postsynaptic_update_time

    @property
    def synapse_dynamics_time(self) -> float:
        """Time in seconds spent in synapse dynamics kernel.
        Only available if :attr:`.ModelSpec.timing_enabled` is set """
        return self._runtime.synapse_dynamics_time

    @property
    def init_sparse_time(self) -> float:
        """Time in seconds spent in sparse initialisation kernel.
        Only available if :attr:`.ModelSpec.timing_enabled` is set """
        return self._runtime.init_sparse_time

    def get_custom_update_time(self, name: str) -> float:
        """Get time in seconds spent in custom update.
        Only available if :attr:`.ModelSpec.timing_enabled` is set.
    
        Args:
            name:   Name of custom update
        """
        return self._runtime.get_custom_update_time(name)

    def get_custom_update_transpose_time(self, name: str) -> float:
        """Get time in seconds spent in transpose custom update.
        Only available if :attr:`.ModelSpec.timing_enabled` is set.
    
        Args:
            name:   Name of custom update
        """
        return self._runtime.get_custom_update_transpose_time(name)
    
    def get_custom_update_remap_time(self, name: str) -> float:
        """Get time in seconds spent in remap custom update.
        Only available if :attr:`.ModelSpec.timing_enabled` is set.
    
        Args:
            name:   Name of custom update
        """
        return self._runtime.get_custom_update_remap_time(name)

    def add_neuron_population(self, pop_name: str, num_neurons: int, 
                              neuron: Union[NeuronModelBase, str],
                              params: PopParamVals = {}, 
                              vars: PopVarVals = {}) -> NeuronGroup:
        """Add a neuron population to the GeNN model

        Args:
            pop_name:       unique name
            num_neurons:    number of neurons
            neuron:         neuron model either as a string referencing a built-in model 
                            (see :mod:`.neuron_models`) or an instance of :class:`.NeuronModelBase`
                            (for example returned by :func:`.create_neuron_model`)
            params:         parameter values for the neuron model (see :ref:`section-parameters`)
            vars:           initial variable values or initialisers 
                            for the neuron model (see :ref:`section-variables`)

        For example, a population of 10 neurons using the built-in Izhikevich model and 
        the standard set of 'tonic spiking' parameters could be added to a model as follows:

        ..  code-block:: python

            pop = model.add_neuron_population("pop", 10, "Izhikevich",
                                              {"a": 0.02, "b": 0.2, "c": -65.0, "d": 6.0},
                                              {"V": -65.0, "U": -20.0})
        """
        if self._built:
            raise Exception("GeNN model already built")

        # Resolve neuron model
        neuron = _get_snippet(neuron, NeuronModelBase, neuron_models)
        
        # Extract parts of vars which should be initialised by GeNN
        var_init = _get_var_init(vars)
        
        # Use superclass to add population
        n_group = self._add_neuron_population(pop_name,
                                              int(num_neurons), neuron,
                                              _prepare_param_vals(params),
                                              var_init)
        
        # Initialise group, store group in dictionary and return
        n_group._init_group(self, vars)
        self.neuron_populations[pop_name] = n_group
        return n_group

    def add_synapse_population(self, pop_name: str, matrix_type: Union[SynapseMatrixType, str],
                               source: NeuronGroup, target: NeuronGroup, 
                               weight_update_init, postsynaptic_init, 
                               connectivity_init: Union[None, SparseConnectivityInit, 
                                                        ToeplitzConnectivityInit] = None) -> SynapseGroup:
        """Add a synapse population to the GeNN model

        Args:
            pop_name:           unique name
            matrix_type:        type of connectivity to use
            source:             source neuron group
            target:             target neuron group
            weight_update_init: initialiser for weight update model, typically
                                created using :func:`init_weight_update`
            postsynaptic_init:  initialiser for postsynaptic model, typically
                                created using :func:`init_postsynaptic`
            connectivity_init:  initialiser for connectivity, typically created
                                using :func:`init_sparse_connectivity` when 
                                ``matrix_type`` is :attr:`SynapseMatrixType.BITMASK`,
                                :attr:`SynapseMatrixType.SPARSE`, 
                                :attr:`SynapseMatrixType.PROCEDURAL` or
                                :attr:`SynapseMatrixType.PROCEDURAL_KERNELG` and with
                                :func:`init_toeplitz_connectivity_connectivity` if
                                it's :attr:`SynapseMatrixType.TOEPLITZ`

        For example, a neuron population ``src_pop`` could be connected to another called
        ``target_pop`` using sparse connectivity, static synapses and 
        exponential shaped current inputs as follows:

        ..  code-block:: python

            pop = model.add_synapse_population("Syn", "SPARSE",
                                               src_pop, target_pop,
                                               init_weight_update("StaticPulseConstantWeight", {"g": 1.0}), 
                                               init_postsynaptic("ExpCurr", {"tau": 5.0}),
                                               init_sparse_connectivity("FixedProbability", {"prob": 0.1}))
        """
        if self._built:
            raise Exception("GeNN model already built")

        # If matrix type is a string, loop up enumeration value
        if isinstance(matrix_type, str):
            matrix_type = getattr(SynapseMatrixType, matrix_type)

        # If no connectivity initialiser is passed, 
        # use unitialised sparse connectivity
        if connectivity_init is None:
            connectivity_init = init_sparse_connectivity(
                init_sparse_connectivity_snippets.Uninitialised(), {})

        # Use superclass to add population
        s_group = self._add_synapse_population(pop_name, matrix_type,
                                               source, target, 
                                               weight_update_init[0],
                                               postsynaptic_init[0],
                                               connectivity_init)

        # Initialise group, store group in dictionary and return
        s_group._init_group(self, postsynaptic_init[1], weight_update_init[1],
                            weight_update_init[2], weight_update_init[3],
                            source, target)
        self.synapse_populations[pop_name] = s_group
        return s_group

    def add_current_source(self, cs_name: str, current_source_model: Union[CurrentSourceModelBase, str], 
                           pop: NeuronGroup, params: PopParamVals = {}, vars: PopVarVals = {}, 
                           var_refs: PopLocalVarRefs = {}) -> CurrentSource:
        """Add a current source to the GeNN model

        Args:
            cs_name:                unique name
            current_source_model:   current source model either as a string referencing a built-in model 
                                    (see :mod:`.current_source_models`) or an instance of :class:`.CurrentSourceModelBase`
                                    (for example returned by :func:`.create_current_source_model`)
            pop:                    neuron population to inject current into
            params:                 parameter values for the current source model (see :ref:`section-parameters`)
            vars:                   initial variable values or initialisers 
                                    for the current source model (see :ref:`section-variables`)
            var_refs:               variables references to neuron variables in ``pop``,
                                    either specified by name or created using :func:`.create_var_ref`
                                    (see :ref:`section-variables-references`)

        For example, a current source to inject a Gaussian noise current can be added to a model as follows:

        ..  code-block:: python

            cs = model.add_current_source("noise", "GaussianNoise", pop,
                                          {"mean": 0.0, "sd": 1.0})

        where ``pop`` is a reference to a neuron population 
        (as returned by :meth:`.GeNNModel.add_neuron_population`)
        """
        if self._built:
            raise Exception("GeNN model already built")

        # Resolve current source model
        current_source_model = _get_snippet(current_source_model, CurrentSourceModelBase,
                                            current_source_models)
        
        # Extract parts of vars which should be initialised by GeNN
        var_init = _get_var_init(vars)
        
        # Use superclass to add population
        c_source = self._add_current_source(cs_name,
                                            current_source_model, pop,
                                            _prepare_param_vals(params),
                                            var_init, var_refs)
        
        # Initialise group, store group in dictionary and return
        c_source._init_group(self, vars, pop)
        self.current_sources[cs_name] = c_source
        return c_source
    
    def add_custom_update(self, cu_name: str, group_name: str, 
                          custom_update_model: Union[CustomUpdateModelBase, str],
                          params: PopParamVals = {}, vars: PopVarVals = {}, 
                          var_refs: Union[PopVarRefs, PopWUVarRefs] = {},
                          egp_refs: PopEGPRefs = {}):
        """Add a custom update to the GeNN model
        
        Args:
            cu_name:                unique name
            group_name:             name of the 'custom update group' to include this update in. 
                                    All custom updates in the same group are executed simultaneously.
            custom_update_model:    custom update model either as a string referencing a built-in model 
                                    (see :mod:`.custom_update_models`) or an instance of 
                                    :class:`.CustomUpdateModelBase`
                                    (for example returned by :func:`.create_custom_update_model`)
            params:                 parameter values for the custom update model (see :ref:`section-parameters`)
            vars:                   initial variable values or initialisers 
                                    for the custom update model (see :ref:`section-variables`)
            var_refs:               references to variables in other populations to 
                                    access from this update, typically created using either
                                    :func:`.create_var_ref` or :func:`.create_wu_var_ref`
                                    (see :ref:`section-variables-references`).
            egp_refs:               references to extra global parameters in other populations
                                    to access from this update, typically created using
                                    :func:`.create_egp_ref` (see :ref:`section-extra-global-parameter-references`).
        
        For example, a custom update to calculate transpose weights could be added to a model as follows:

        ..  code-block:: python

            cu = model.add_custom_update("tranpose_pop", "transpose", "Transpose",
                                         var_refs={"variable": create_wu_var_ref(fwd_sg, "g",
                                                                                 back_sg, "g")})

        where ``fwd_sg`` and ``back_sg`` are references to synapse populations
        (as returned by :meth:`.GeNNModel.add_synapse_population`). This update
        could then subsequently be triggered using the name of it's update group with:
        
        ..  code-block:: python

            model.custom_update("transpose")

        """
        if self._built:
            raise Exception("GeNN model already built")
        
        # Resolve custom update model
        custom_update_model = _get_snippet(custom_update_model, CustomUpdateModelBase,
                                           custom_update_models)
        
        # Extract parts of vars which should be initialised by GeNN
        var_init = _get_var_init(vars)

        # Use superclass to add population
        c_update = self._add_custom_update(cu_name, group_name,
                                           custom_update_model,
                                           _prepare_param_vals(params),
                                           var_init, var_refs, egp_refs)

        # Setup back-reference, store group in dictionary and return
        c_update._init_group(self, vars)
        self.custom_updates[cu_name] = c_update
        return c_update
    
    def add_custom_connectivity_update(self, cu_name: str, group_name: str, 
                                       syn_group: SynapseGroup,
                                       custom_conn_update_model: Union[CustomConnectivityUpdateModelBase, str],
                                       params: PopParamVals = {}, vars: PopVarVals = {}, pre_vars: PopVarVals = {},
                                       post_vars: PopVarVals = {}, var_refs: PopWUVarRefs = {},
                                       pre_var_refs: PopVarRefs = {}, post_var_refs: PopVarRefs = {},
                                       egp_refs: PopEGPRefs = {}):
        """Add a custom connectivity update to the GeNN model

        Args:
            cu_name:                    unique name
            group_name:                 name of the 'custom update group' to include this update in. 
                                        All custom updates in the same group are executed simultaneously.
            syn_group:                  Synapse group to attach custom connectivity update to
            custom_conn_update_model:   custom connectivity update model either as a string referencing a built-in model 
                                        (see :mod:`.custom_connectivity_update_models`) or an instance of 
                                        :class:`.CustomConnectivityUpdateModelBaseUpdateModelBase`
                                        (for example returned by :func:`.create_custom_connectivity_update_model`)
            params:                     parameter values for the custom connectivity model (see :ref:`section-parameters`)
            vars:                       initial synaptic variable values or
                                        initialisers (see :ref:`section-variables`)
            pre_vars:                   initial presynaptic variable values or
                                        initialisers (see :ref:`section-variables`)
            post_vars:                  initial postsynaptic variable values or initialisers
                                        (see :ref:`section-variables`)
            var_refs:                   references to synaptic variables,
                                        typically created using :func:`.create_wu_var_ref`
                                        (see :ref:`section-variables-references`)
            pre_var_refs:               references to presynaptic variables,
                                        typically created using :func:`.create_var_ref`
                                        (see :ref:`section-variables-references`)
            post_var_refs:              references to postsynaptic variables,
                                        typically created using :func:`.create_var_ref`
                                        (see :ref:`section-variables-references`)
            egp_refs:                   references to extra global parameters in other populations
                                        to access from this update, typically created using
                                        :func:`.create_egp_ref` (see :ref:`section-extra-global-parameter-references`).

        """
        if self._built:
            raise Exception("GeNN model already built")

        # Resolve custom update model
        custom_connectivity_update_model = _get_snippet(
            custom_conn_update_model, CustomConnectivityUpdateModelBase,
            custom_connectivity_update_models)

        # Extract parts of vars which should be initialised by GeNN
        var_init = _get_var_init(vars)
        pre_var_init = _get_var_init(pre_vars)
        post_var_init = _get_var_init(post_vars)

        # Use superclass to add population
        c_update = self._add_custom_connectivity_update(
            cu_name, group_name, syn_group, custom_connectivity_update_model,
            _prepare_param_vals(params), var_init, pre_var_init, post_var_init,
            var_refs, pre_var_refs, post_var_refs, egp_refs)

        # Setup back-reference, store group in dictionary and return
        c_update._init_group(self, vars, pre_vars, post_vars)
        self.custom_connectivity_updates[cu_name] = c_update
        return c_update
        
    def build(self, path_to_model: str = "./", always_rebuild: bool = False, 
              never_rebuild: bool = False):
        """Finalize and build a GeNN model

        Args:
            path_to_model:  path where to place the generated model code.
                            Defaults to the local directory.
            always_rebuild: should model be rebuilt even if
                            it doesn't appear to be required
            never_rebuild:  should model never be rebuilt even it appears to
                            need it. This should only ever be used to prevent
                            file overwriting when performing parallel runs
        """

        if self._built:
            raise Exception("GeNN model already built")
        self._path_to_model = path_to_model

        # Create output path
        output_path = path.join(path_to_model, self.name + "_CODE")
        share_path = path.join(path.split(__file__)[0], "share")

        # Finalize model
        self._finalise()

        # Create suitable preferences object for backend
        self._preferences = self._backend_module.Preferences()

        # Set attributes on preferences object from kwargs
        for k, v in self._preference_kwargs.items():
            if hasattr(self._preferences, k):
                setattr(self._preferences, k, v)
            else:
                raise ValueError(f"Unknown preference '{k}'")
        
        # Create backend
        self._backend = self._backend_module._create_backend(
            self, output_path, self.backend_log_level, self._preferences)

        # Generate code
        self._model_merged = generate_code(self, self._backend, share_path,
                                           output_path, always_rebuild, never_rebuild)

        # Build code
        if not never_rebuild:
            if system() == "Windows":
                check_call([_msbuild, "/p:Configuration=Release", "/m", "/verbosity:quiet",
                            path.join(output_path, "runner.vcxproj")])
            else:
                check_call(["make", "-j", str(cpu_count(logical=False)), "-C", output_path])

        self._built = True

    def load(self, num_recording_timesteps: Optional[int] = None):
        """Load the previously built model into memory;
        
        Args:
            num_recording_timesteps:    Number of timesteps to record spikes
                                        for. :meth:`.pull_recording_buffers_from_device` 
                                        must be called after this number of timesteps
        """
        if self._loaded:
            raise Exception("GeNN model already loaded")
        if not self._built:
            raise Exception("GeNN model has not been built")
        
        # Create runtime
        self._runtime = Runtime(self._path_to_model, self._model_merged,
                                self._backend)
        
        # If model uses recording system and recording timesteps is not set
        if self._recording_in_use and num_recording_timesteps is None:
            raise Exception("Cannot use recording system without passing "
                            "number of recording timesteps to GeNNModel.load")

        # Allocate memory
        self._runtime.allocate(num_recording_timesteps)

        # Loop through neuron populations and load any
        # extra global parameters required for initialization
        for pop_data in self.neuron_populations.values():
            pop_data._load_init_egps()

        # Loop through synapse populations and load any 
        # extra global parameters required for initialization
        for pop_data in self.synapse_populations.values():
            pop_data._load_init_egps()

        # Loop through current sources
        for src_data in self.current_sources.values():
            src_data._load_init_egps()

        # Loop through custom connectivity updates
        for cu_data in self.custom_connectivity_updates.values():
            cu_data._load_init_egps()

        # Loop through custom updates
        for cu_data in self.custom_updates.values():
            cu_data._load_init_egps()

        # Initialize model
        self._runtime.initialize()

        # Loop through neuron populations
        for pop_data in self.neuron_populations.values():
            pop_data._load()

        # Loop through synapse populations
        for pop_data in self.synapse_populations.values():
            pop_data._load()

        # Loop through current sources
        for src_data in self.current_sources.values():
            src_data._load()

        # Loop through custom connectivity updates
        for cu_data in self.custom_connectivity_updates.values():
            cu_data._load()

        # Loop through custom updates
        for cu_data in self.custom_updates.values():
            cu_data._load()

        # Now everything is set up call the sparse initialisation function
        self._runtime.initialize_sparse()

        # Set loaded flag and built flag
        self._loaded = True
        self._built = True


    def unload(self):
        """Unload a previously loaded model, freeing all memory"""
        if not self._loaded:
            raise Exception("GeNN model has not been built")
            
        # Loop through custom updates and unload
        for cu_data in self.custom_updates.values():
            cu_data._unload()
        
        # Loop through custom connectivity updates and unload
        for cu_data in self.custom_connectivity_updates.values():
            cu_data._unload()
    
        # Loop through current sources and unload
        for src_data in self.current_sources.values():
            src_data._unload()

        # Loop through synapse populations and unload
        for pop_data in self.synapse_populations.values():
            pop_data._unload()

        # Loop through neuron populations and unload
        for pop_data in self.neuron_populations.values():
            pop_data._unload()

        # Close runtime
        self._runtime = None

        # Clear loaded flag
        self._loaded = False

    def step_time(self):
        """Make one simulation step"""
        if not self._loaded:
            raise Exception("GeNN model has to be loaded before stepping")

        self._runtime.step_time()
    
    def custom_update(self, name: str):
        """Perform custom update

        Args:
            name:   Name of custom update. Corresponds to the ``group_name``
                    parameter passed to :meth:`.add_custom_update` and
                    :meth:`.add_custom_connectivity_update`.
        """
        if not self._loaded:
            raise Exception("GeNN model has to be loaded before performing custom update")
            
        self._runtime.custom_update(name)
   

    def pull_recording_buffers_from_device(self):
        """Pull recording buffers from device"""
        if not self._loaded:
            raise Exception("GeNN model has to be loaded before pulling recording buffers")

        if not self._recording_in_use:
            raise Exception("Cannot pull recording buffer if recording system is not in use")

        # Pull recording buffers from device
        self._runtime.pull_recording_buffers_from_device()

def init_var(snippet: Union[InitVarSnippetBase, str],
             params: PopParamVals = {}):
    """Initialises a variable initialisation snippet with parameter values
     
    Args:
        snippet:        variable init snippet, either as a string referencing
                        a built-in snippet (see :mod:`.init_var_snippets`) 
                        or an instance of :class:`.InitVarSnippetBase` 
                        (for example returned by :func:`.create_var_init_snippet`)
        params:         parameter values for the variable init snippet (see :ref:`section-parameters`)
    
    For example, the built-in model "Normal" could be used to initialise a variable 
    by sampling from the normal distribution with a mean of 0 and a standard deviation of 1:

    ..  code-block:: python

        init = init_var("Normal", {"mean": 0.0, "sd": 1.0})
    """
    # Get snippet and wrap in VarInit object
    snippet = _get_snippet(snippet, InitVarSnippetBase, init_var_snippets)

    # Use add function to create suitable VarInit
    return VarInit(snippet, _prepare_param_vals(params))

def init_sparse_connectivity(snippet: Union[InitSparseConnectivitySnippetBase, str],
                             params: PopParamVals = {}):
    """Initialises a sparse connectivity initialisation snippet with parameter values
     
    Args:
        snippet:        sparse connectivity init snippet, either as a string referencing
                        a built-in snippet (see :mod:`.init_sparse_connectivity_snippets`) 
                        or an instance of :class:`.InitSparseConnectivitySnippetBase` 
                        (for example returned by :func:`.create_sparse_connect_init_snippet`)
        params:         parameter values for the sparse connectivity init snippet (see :ref:`section-parameters`)
    
    For example, the built-in "FixedProbability" snippet could be used to generate connectivity 
    where each pair of pre and postsynaptic neurons is connected with a probability of 0.1:

    ..  code-block:: python

        init = init_sparse_connectivity("FixedProbability", {"prob": 0.1})
    """
    # Get snippet and wrap in SparseConnectivityInit object
    snippet = _get_snippet(snippet, InitSparseConnectivitySnippetBase,
                           init_sparse_connectivity_snippets)
    return SparseConnectivityInit(snippet, _prepare_param_vals(params))


def init_postsynaptic(snippet: Union[PostsynapticModelBase, str], 
                      params: PopParamVals = {}, vars: PopVarVals = {}, 
                      var_refs: PopLocalVarRefs = {}):
    """Initialises a postsynaptic model with parameter values, 
    variable initialisers and variable references

    Args:
        snippet:        postsynaptic model either as a string referencing a built-in model 
                        (see :mod:`.postsynaptic_models`) or an instance of 
                        :class:`.PostsynapticModelBase` (for example returned 
                        by :func:`.create_postsynaptic_model`)
        params:         parameter values for the postsynaptic model (see :ref:`section-Parameters`)
        vars:           initial synaptic variable values or initialisers 
                        for the postsynaptic model (see :ref:`section-variables`)
        var_refs:       references to postsynaptic neuron variables,
                        either specified by name or created using :func:`.create_var_ref`
                        (see :ref:`section-variables-references`)

    For example, the built-in conductance model with exponential 
    current shaping could be initialised as follows:

    ..  code-block:: python

        postsynaptic_init = init_postsynaptic("ExpCond", {"tau": 1.0, "E": -80.0}, 
                                              var_refs={"V": create_var_ref(pop1, "V")})
    
    where ``pop1`` is a reference to the postsynaptic neuron population 
    (as returned by :meth:`.GeNNModel.add_neuron_population`)
    """
    # Get snippet and wrap in PostsynapticInit object
    snippet = _get_snippet(snippet, PostsynapticModelBase,
                           postsynaptic_models)
    
    # Extract parts of var spaces which should be initialised by GeNN
    var_init = _get_var_init(vars)
    
    return (PostsynapticInit(snippet, _prepare_param_vals(params), 
                             var_init, var_refs), 
            vars)

def init_weight_update(snippet, params: PopParamVals = {}, vars: PopVarVals = {},
                       pre_vars: PopVarVals = {}, post_vars: PopVarVals = {}, 
                       pre_var_refs: PopLocalVarRefs = {}, 
                       post_var_refs: PopLocalVarRefs = {},
                       psm_var_refs: PopLocalVarRefs = {}):
    """Initialises a weight update model with parameter values, 
    variable initialisers and variable references.

    Args:
        snippet:        weight update model either as a string referencing a built-in model 
                        (see :mod:`.weight_update_models`) or an instance of 
                        :class:`.WeightUpdateModelBase` (for example returned 
                        by :func:`.create_weight_update_model`)
        params:         parameter values (see :ref:`section-parameters`)
        vars:           initial synaptic variable values or
                        initialisers (see :ref:`section-variables`)
        pre_vars:       initial presynaptic variable values or
                        initialisers (see :ref:`section-variables`)
        post_vars:      initial postsynaptic variable values or initialisers
                        (see :ref:`section-variables`)
        pre_var_refs:   references to presynaptic neuron variables,
                        either specified by name or created using :func:`.create_var_ref`
                        (see :ref:`section-variables-references`)
        post_var_refs:  references to postsynaptic neuron variables,
                        either specified by name or created using :func:`.create_var_ref`
                        (see :ref:`section-variables-references`)
        psm_var_refs:   references to postsynaptic model variables,
                        specified by name (see :ref:`section-variables-references`)

    For example, the built-in static pulse model with 
    constant weights could be initialised as follows:

    ..  code-block:: python

        weight_init = init_weight_update("StaticPulseConstantWeight", {"g": 1.0})
    """
    # Get snippet and wrap in WeightUpdateInit object
    snippet = _get_snippet(snippet, WeightUpdateModelBase,
                           weight_update_models)
    
    var_init = _get_var_init(vars)
    pre_var_init = _get_var_init(pre_vars)
    post_var_init = _get_var_init(post_vars)
    
    return (WeightUpdateInit(snippet, _prepare_param_vals(params), var_init, 
                             pre_var_init, post_var_init,
                             pre_var_refs, post_var_refs, psm_var_refs),
            vars, pre_vars, post_vars)

@deprecated("The name of this function was ambiguous, use init_sparse_connectivity instead")
def init_connectivity(init_sparse_connect_snippet, params={}):
    return init_sparse_connectivity(init_sparse_connect_snippet, params)

def init_toeplitz_connectivity(init_toeplitz_connect_snippet, params={}):
    """Initialises a toeplitz connectivity 
    initialisation snippet with parameter values
     
    Args:
        snippet:        toeplitz connectivity init snippet, either as a string referencing
                        a built-in snippet (see :mod:`.init_toeplitz_connectivity_snippets`) 
                        or an instance of :class:`.InitToeplitzConnectivitySnippetBase` 
                        (for example returned by :func:`.create_toeplitz_connect_init_snippet`)
        params:         parameter values for the toeplitz connectivity init snippet (see :ref:`section-parameters`)
    
    For example, the built-in "Conv2D" snippet could be used to generate 2D convolutional 
    connectivity with a :math:`3 \\times 3` kernel, a :math:`64 \\times 64 \\times 1` input
    and a :math:`62 \\times 62 \\times 1` output:

    ..  code-block:: python

        params = {"conv_kh": 3, "conv_kw": 3,
                  "conv_ih": 64, "conv_iw": 64, "conv_ic": 1,
                  "conv_oh": 62, "conv_ow": 62, "conv_oc": 1}
    
        init = init_toeplitz_connectivity("Conv2D", params))

    .. note::
        This should be used to connect a presynaptic neuron population with 
        :math:`64 \\times 64 \\times 1 = 4096` neurons to a postsynaptic neuron
        population with :math:`62 \\times 62 \\times 1 = 3844` neurons.
    """
    # Get snippet and wrap in InitToeplitzConnectivitySnippet object
    init_toeplitz_connect_snippet = _get_snippet(init_toeplitz_connect_snippet,
                                                 InitToeplitzConnectivitySnippetBase,
                                                 init_toeplitz_connectivity_snippets)
    return ToeplitzConnectivityInit(init_toeplitz_connect_snippet, 
                                    _prepare_param_vals(params))

def _upgrade_code_string(code, class_name):
    # Apply special-case upgrades
    upgraded = False

    for obj, replace in _code_upgrades:
        # If there's no supported replacement
        if replace is None:
            # Search and give error if found
            match = obj.search(code)
            if match is not None:
                raise RuntimeError(f"'{match.group(0)}' call in "
                                   f"'{class_name}' is no longer supported")
        # Otherwise
        else:
            # Replace pattern in code
            code, n_subs = obj.subn(replace, code)
            
            # If any substitutions were made, give warning
            if n_subs > 0:
                upgraded = True

    # Replace old style $(XX) variables with plain XX
    # **NOTE** this is done after functions as single-parameter
    # function calls and variables were indistinguishable with old syntax
    code, n_subs = _var_upgrade.subn(r"\1", code)
    if n_subs > 0:
        upgraded = True

    # If any upgrades were made, give warning
    if upgraded:
        warn(f"Legacy $() syntax in '{class_name}' has been automatically "
             f"removed but this functionality will be removed in future so "
             f"please update your model", FutureWarning)
    return code

def _create_model(class_name: str, base, params, param_names, derived_params,
                  extra_global_params, custom_body):
    def ctor(self):
        base.__init__(self)

    body = {
        "__init__": ctor,
    }

    if param_names is not None:
        warn("The 'param_names' parameter has been renamed to 'params' "
             "and will be removed in future", FutureWarning)
        params = param_names

    if params is not None:
        body["get_params"] =\
            lambda self: [Param(p) if isinstance(p, str) else Param(*p)
                          for p in params]

    if derived_params is not None:
        # Helper to wrap lambda function so as to extract underlying value from NumericValue
        # parameters and wrap the resulting derived parameter value in a NumericValue!
        # **NOTE** seperate function is required to ensure f is bound correctly
        def wrap_lambda(f):
            return lambda pars, dt: NumericValue(f({n: p.value 
                                                    for n, p in pars.items()},
                                                   dt))

        body["get_derived_params"] = \
            lambda self: [DerivedParam(dp[0], wrap_lambda(dp[1]), *dp[2:])
                          for dp in derived_params]

    if extra_global_params is not None:
        body["get_extra_global_params"] = \
            lambda self: [EGP(*egp) for egp in extra_global_params]

    if custom_body is not None:
        body.update(custom_body)

    return type(class_name, (base,), body)()

def create_neuron_model(class_name: str, params: ModelParamsType = None,
                        param_names=None, vars: ModelVarsType = None,
                        var_name_types=None, 
                        derived_params: ModelDerivedParamsType = None,
                        sim_code: Optional[str] = None,
                        threshold_condition_code: Optional[str] = None,
                        reset_code: Optional[str] = None,
                        extra_global_params: ModelEGPType = None,
                        additional_input_vars=None,
                        auto_refractory_required: bool = False):
    """Creates a new neuron model.
    Within all of the code strings, the variables, parameters,
    derived parameters, additional input variables and extra global
    parameters defined in this model can all be referred to by name.
    Additionally, the code may refer to the following built-in read-only variables

    - ``dt`` which represents the simulation time step (as specified via  :meth:`.GeNNModel.dt`).
    - ``Isyn`` which represents the total incoming synaptic input.
    - ``id`` which represents a neurons index within a population (starting from zero).
    - ``num_neurons`` which represents the number of neurons in the population.

    Args:
        class_name:                 name of the new class (only for debugging)
        params:                     name and optional types of model parameters
        vars:                       names, types and optional variable access
                                    modifiers of model variables
        derived_params:             names, types and callables to calculate
                                    derived parameter values from params
        sim_code:                   string containing the simulation code
                                    statements to be run every timestep
        threshold_condition_code:   string containing a threshold condition
                                    expression to test whether a spike
                                    should be emitted
        reset_code:                 string containing the reset code
                                    statements to run after emitting a spike
        extra_global_params:        names and types of model
                                    extra global parameters
        additional_input_vars:      list of tuples with names and types as
                                    strings and initial values of additional
                                    local input variables
        auto_refractory_required:   does this model require auto-refractory
                                    logic to be generated?
    
    For example, we can define a leaky integrator :math:`\\tau\\frac{dV}{dt}= -V + I_{{\\rm syn}}` solved using Euler's method:

    ..  code-block:: python

        leaky_integrator_model = pygenn.create_neuron_model(
            "leaky_integrator",

            sim_code=
                \"""
                V += (-V + Isyn) * (dt / tau);
                \""",
            threshold_condition_code="V >= 1.0",
            reset_code=
                \"""
                V = 0.0;
                \""",

            params=["tau"],
            vars=[("V", "scalar", pygenn.VarAccess.READ_WRITE)])

    Additional input variables
    --------------------------
    Normally, neuron models receive the linear sum of the inputs coming from all of their synaptic inputs through the ``Isyn`` variable. 
    However neuron models can define additional input variables, allowing input from different synaptic inputs to be combined non-linearly.
    For example, if we wanted our leaky integrator to operate on the the product of two input currents, we could modify our model as follows:

    ..  code-block:: python

        ...
        additional_input_vars=[("Isyn2", "scalar", 1.0)],
        sim_code=
            \"""
            const scalar input = Isyn * Isyn2;
            sim_code="V += (-V + input) * (dt / tau);
            \""",
        ...

    """
    body = {}
    
    if var_name_types is not None:
        warn("The 'var_name_types' parameter has been renamed to 'vars' "
             "and will be removed in future", FutureWarning)
        vars = var_name_types

    if sim_code is not None:
        body["get_sim_code"] =\
            lambda self: dedent(_upgrade_code_string(sim_code, class_name))

    if threshold_condition_code is not None:
        body["get_threshold_condition_code"] = \
            lambda self: dedent(_upgrade_code_string(threshold_condition_code,
                                                    class_name))

    if reset_code is not None:
        body["get_reset_code"] = lambda self: dedent(_upgrade_code_string(reset_code,
                                                                          class_name))

    if additional_input_vars:
        body["get_additional_input_vars"] = \
            lambda self: [ParamVal(a[0], a[1], NumericValue(a[2]))
                          for a in additional_input_vars]

    if vars is not None:
        body["get_vars"] = \
            lambda self: [Var(*vn) for vn in vars]

    if auto_refractory_required is not None:
        body["is_auto_refractory_required"] = \
            lambda self: auto_refractory_required

    return _create_model(class_name, NeuronModelBase, params, param_names,
                         derived_params, extra_global_params, body)


def create_postsynaptic_model(class_name, params=None, param_names=None,
                              vars=None, var_name_types=None, 
                              neuron_var_refs: ModelVarRefsType = None,
                              derived_params: ModelDerivedParamsType = None,
                              sim_code: Optional[str] = None, decay_code=None,
                              apply_input_code=None,
                              extra_global_params: ModelEGPType = None):
    """Creates a new postsynaptic update model.
    Within all of the code strings, the variables, parameters,
    derived parameters and extra global parameters defined in this model
    can all be referred to by name. Additionally, the code may refer to the
    following built-in read-only variables:

    - ``dt`` which represents the simulation time step (as specified via  :meth:`.GeNNModel.dt`)
    - ``id`` which represents a neurons index within a population (starting from zero)
    - ``num_neurons`` which represents the number of neurons in the population
    - ``inSyn`` which contains the summed input received from the weight update model through ``addToPost()`` or ``addToPostDelay()``

    Finally, the function ``injectCurrent(x)`` can be used to inject a current
    ``x`` into the postsynaptic neuron. The variable it goes into can be
    configured using the :attr:`SynapseGroup.post_target_var`. By default it targets ``Isyn``.

    Args:
        class_name:                 name of the new class (only for debugging)
        params:                     name and optional types of model parameters
        vars:                       names, types and optional variable access
                                    modifiers of model variables
        neuron_var_refs:            names, types and optional variable access
                                    of references to be assigned to postsynaptic
                                    neuron variables
        derived_params:             names, types and callables to calculate
                                    derived parameter values from params
        sim_code:                   string containing the simulation code
                                    statements to be run every timestep
        extra_global_params:        names and types of model
                                    extra global parameters
    """
    body = {}
    if decay_code is not None or apply_input_code is not None:
        raise RuntimeError("Creating postsynaptic models with seperate "
                           "'decay_code' and 'apply_code' code strings is no "
                           "longer supported. Please provide 'sim_code' using "
                           "the injectCurrent(X) function to provide input.")
    if var_name_types is not None:
        warn("The 'var_name_types' parameter has been renamed to 'vars' "
             "and will be removed in future", FutureWarning)
        vars = var_name_types

    if sim_code is not None:
        body["get_sim_code"] =\
            lambda self: dedent(_upgrade_code_string(sim_code, class_name))

    if vars is not None:
        body["get_vars"] = \
            lambda self: [Var(*vn) for vn in vars]
    
    if neuron_var_refs is not None:
        body["get_neuron_var_refs"] =\
            lambda self: [VarRef(*v) for v in neuron_var_refs]
    
    return _create_model(class_name, PostsynapticModelBase, params,
                         param_names, derived_params,
                         extra_global_params, body)


def create_weight_update_model(
        class_name: str, params: ModelParamsType = None, param_names=None,
        vars: ModelVarsType = None, var_name_types=None,
        pre_vars: ModelVarsType = None, pre_var_name_types=None,
        post_vars: ModelVarsType = None, post_var_name_types=None,
        pre_neuron_var_refs: ModelVarRefsType = None,
        post_neuron_var_refs: ModelVarRefsType = None,
        psm_var_refs: ModelVarRefsType = None,
        derived_params: ModelDerivedParamsType = None, sim_code=None,
        pre_spike_syn_code: Optional[str] = None, event_code=None,
        pre_event_syn_code: Optional[str] = None,
        post_event_syn_code: Optional[str] = None, learn_post_code=None,
        post_spike_syn_code: Optional[str] = None,
        synapse_dynamics_code: Optional[str] = None,
        event_threshold_condition_code=None,
        pre_event_threshold_condition_code: Optional[str] = None,
        post_event_threshold_condition_code: Optional[str] = None,
        pre_spike_code: Optional[str] = None,
        post_spike_code: Optional[str] = None,
        pre_dynamics_code: Optional[str] = None,
        post_dynamics_code: Optional[str] = None,
        extra_global_params: ModelEGPType  = None):
    """Creates a new weight update model.
    GeNN operates on the assumption that the postsynaptic output of the synapses are added linearly at the postsynaptic neuron.
    Within all of the synaptic code strings (``pre_spike_syn_code``, ``pre_event_syn_code``,
    ``post_event_syn_code``, ``post_spike_syn_code`` and ``synapse_dynamics_code`` ) these currents are delivered using the ``addToPost(inc)`` function.
    For example,

    ..  code-block:: python

        pre_spike_syn_code="addToPost(inc);"

    where ``inc`` is the amount to add to the postsynapse model's ``inSyn`` variable for each pre-synaptic spike.
    Dendritic delays can also be inserted between the synapse and the postsynaptic neuron by using the ``addToPostDelay(inc, delay)`` function.
    For example,

    ..  code-block:: python

        pre_spike_syn_code="addToPostDelay(inc, delay);"

    where, once again, ``inc`` is the amount to add to the postsynaptic neuron's ``inSyn`` variable and ``delay`` is the length of the dendritic delay in timesteps.
    By implementing ``delay`` as a weight update model variable, heterogeneous synaptic delays can be implemented.
    For an example, see :func:`.weight_update_models.StaticPulseDendriticDelay` for a simple synapse update model with heterogeneous dendritic delays.
    These delays can also be used to provide delayed access to ``post_vars`` and ``post_neuron_var_refs`` using ``[]`` syntax. For example,

    ..  code-block:: python

        pre_spike_syn_code="variable -= postVar[delay];"

    where, ``variable`` is a per-synapse variable; ``postVar`` is either a postsynaptic variable or postsynaptic variable reference; 
    and ``delay`` is some sort of integer expression. When using dendritic delays, the *maximum* dendritic delay for a synapse populations 
    must be specified via the :attr:`SynapseGroup.max_dendritic_delay_timesteps` property. One can also define synaptic effects that 
    occur in the reverse direction, i.e. terms that are added to a target variable in the _presynaptic_ neuron using the ``addToPre(inc)`` function.
    For example,

    ..  code-block:: python

        pre_spike_syn_code="addToPre(inc * V_post);"

    would add terms ``inc * V_post`` to for each *outgoing* synapse of a presynaptic neuron.
    Similar to postsynaptic models, by default these inputs are accumulated in ``Isyn`` in the presynaptic 
    neuron but they can also be directed to additional input variables by setting the 
    :attr:`SynapseGroup.pre_target_var` property. Unlike for normal forward synaptic 
    actions, reverse synaptic actions with ``addToPre(inc)`` are not modulated through 
    a post-synaptic model but added directly into the indicated presynaptic target input variable.

    Args:
        class_name:                             name of the new class (only for debugging)
        params:                                 name and optional types of model parameters
        vars:                                   names, types and optional variable access
                                                modifiers of per-synapse model variables
        pre_vars:                               names, types and optional variable access
                                                modifiers of per-presynaptic neuron model variables
        post_vars                               names, types and optional variable access
                                                modifiers of per-postsynaptic neuron model variables
        pre_neuron_var_refs:                    names, types and optional variable access
                                                of references to be assigned to presynaptic
                                                neuron variables
        post_neuron_var_refs:                   names, types and optional variable access
                                                of references to be assigned to postsynaptic
                                                neuron variables
        psm_var_refs:                           names, types and optional variable access
                                                of references to be assigned to postsynaptic
                                                model variables
        derived_params:                         names, types and callables to calculate
                                                derived parameter values from params
        pre_spike_syn_code:                     string with the presynaptic spike code
        pre_event_syn_code:                     string with the presynaptic event code
        post_event_syn_code:                    string with the postsynaptic event code
        post_spike_syn_code:                    string with the postsynaptic spike code
        synapse_dynamics_code:                  string with the synapse dynamics code
        pre_event_threshold_condition_code:     string with the presynaptic event threshold
                                                condition code
        post_event_threshold_condition_code:    string with the postsynaptic event threshold
                                                condition code
        pre_spike_code:                         string with the code run once per
                                                spiking presynaptic neuron. Only
                                                presynaptic variables and 
                                                variable references can be 
                                                referenced from this code.
        post_spike_code:                        string with the code run once per
                                                spiking postsynaptic neuron
        pre_dynamics_code:                      string with the code run every
                                                timestep on presynaptic neuron.
                                                Only presynaptic variables and
                                                variable references can be
                                                referenced from this code.
        post_dynamics_code:                     string with the code run every
                                                timestep on postsynaptic neuron.
                                                Only postsynaptic variables and
                                                variable references can be
                                                referenced from this code.
        extra_global_params:                    names and types of model
                                                extra global parameters
    
    For example, we can define a simple additive STDP rule with 
    nearest-neighbour spike pairing and the following time-dependence (equivalent to :func:`.weight_update_models.STDP`):

    ..  math::

        \\Delta w_{ij} & = \
            \\begin{cases}
                A_{+}\\exp\\left(-\\frac{\\Delta t}{\\tau_{+}}\\right) & if\\, \\Delta t>0\\\\
                A_{-}\\exp\\left(\\frac{\\Delta t}{\\tau_{-}}\\right) & if\\, \\Delta t\\leq0
            \\end{cases}

    in a fully event-driven manner as follows:

    ..  code-block:: python

        stdp_additive_model = pygenn.create_weight_update_model(
            "stdp_additive",
            params=["tauPlus", "tauMinus", "aPlus", "aMinus", "wMin", "wMax"],
            vars=[("g", "scalar")],

            pre_spike_syn_code=
                \"""
                addToPost(g);
                const scalar dt = t - st_post;
                if (dt > 0) {
                    const scalar timing = exp(-dt / tauMinus);
                    const scalar newWeight = g - (Aminus * timing);
                    g = fmax(Wmin, fmin(Wmax, newWeight));
                }
                \""",
            post_spike_syn_code=
                \"""
                const scalar dt = t - st_pre;
                if (dt > 0) {
                    const scalar timing = exp(-dt / tauPlus);
                    const scalar newWeight = g + (Aplus * timing);
                    g = fmax(Wmin, fmin(Wmax, newWeight));
                }
                \""")

    Pre and postsynaptic dynamics
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    The memory required for synapse variables and the computational cost of updating them tends to grow with :math:`O(N^2)` with the number of neurons.
    Therefore, if it is possible, implementing synapse variables on a per-neuron rather than per-synapse basis is a good idea. 
    The ``pre_var_name_types`` and ``post_var_name_types`` keyword arguments are used to define any pre or postsynaptic state variables.
    For example, using pre and postsynaptic variables, our event-driven STDP rule can be extended to use all-to-all spike pairing using pre and postsynaptic *trace* variables [Morrison2008]_ :

    ..  code-block:: python

        stdp_additive_2_model = genn_model.create_custom_weight_update_class(
            "stdp_additive_2",
            params=["tauPlus", "tauMinus", "aPlus", "aMinus", "wMin", "wMax"],
            vars=[("g", "scalar")],
            pre_vars=[("preTrace", "scalar")],
            post_vars=[("postTrace", "scalar")],
            
            pre_spike_syn_code=
                \"""
                addToPost(g);
                const scalar dt = t - st_post;
                if(dt > 0) {
                    const scalar newWeight = g - (aMinus * postTrace);
                    g = fmin(wMax, fmax(wMin, newWeight));
                }
                \""",
            post_spike_syn_code=
                \"""
                const scalar dt = t - st_pre;
                if(dt > 0) {
                    const scalar newWeight = g + (aPlus * preTrace);
                    g = fmin(wMax, fmax(wMin, newWeight));
                }
                \""",

            pre_spike_code="preTrace += 1.0;",
            pre_dynamics_code="preTrace *= tauPlusDecay;",
            post_spike_code="postTrace += 1.0;",
            post_dynamics_code="postTrace *= tauMinusDecay;")

    Synapse dynamics
    ----------------
    Unlike the event-driven updates previously described, synapse dynamics code is run for each synapse and each timestep, i.e. it is time-driven. 
    This can be used where synapses have internal variables and dynamics that are described in continuous time, e.g. by ODEs.
    However, using this mechanism is typically computationally very costly because of the large number of synapses in a typical network. 
    By using the ``addToPost()`` and ``addToPostDelay()`` functions discussed in the context of ``pre_spike_syn_code``, the synapse dynamics can also be used to implement continuous synapses for rate-based models.
    For example a continous synapse which multiplies a presynaptic neuron variable by the weight could be added to a weight update model definition as follows:

    ..  code-block:: python
        
        pre_neuron_var_refs=[("V_pre", "scalar")],
        synapse_dynamics_code="addToPost(g * V_pre);",

    Spike-like events
    -----------------
    As well as time-driven synapse dynamics and spike event-driven updates, GeNN weight update models also support "spike-like events". 
    These can be triggered by a threshold condition evaluated on the pre or postsynaptic neuron. 
    This typically involves pre or postsynaptic weight update model variables or variable references respectively.

    For example, to trigger a presynaptic spike-like event when the presynaptic neuron's voltage is greater than 0.02, the following could be added to a weight update model definition:

    ..  code-block:: python

        pre_neuron_var_refs=[("V_pre", "scalar")],
        pre_event_threshold_condition_code="V_pre > -0.02"

    Whenever this expression evaluates to true, the event code in ``pre_event_code`` will be executed. 
    """
    body = {}
    
    if sim_code is not None:
        warn("The 'sim_code' parameter has been renamed to "
             "'pre_spike_syn_code' and will be removed in future",
             FutureWarning)
        pre_spike_syn_code = sim_code
    if learn_post_code is not None:
        warn("The 'learn_post_code' parameter has been renamed to "
            "'post_spike_syn_code' and will be removed in future",
            FutureWarning)
        post_spike_syn_code = learn_post_code
    if event_code is not None:
        warn("The 'event_code' parameter has been renamed to 'pre_event_syn_code'"
             " and will be removed in future", FutureWarning)
        pre_event_syn_code = event_code
    if event_threshold_condition_code is not None:
        warn("The 'event_threshold_condition_code' parameter has been "
             "renamed to 'pre_event_threshold_condition_code' and will "
             "be removed in future", FutureWarning)
        pre_event_threshold_condition_code = event_threshold_condition_code
    if var_name_types is not None:
        warn("The 'var_name_types' parameter has been renamed to 'vars' "
             "and will be removed in future", FutureWarning)
        vars = var_name_types
    if pre_var_name_types is not None:
        warn("The 'pre_var_name_types' parameter has been renamed to 'pre_vars' "
             "and will be removed in future", FutureWarning)
        pre_vars = pre_var_name_types
    if post_var_name_types is not None:
        warn("The 'post_var_name_types' parameter has been renamed to 'post_vars' "
             "and will be removed in future", FutureWarning)
        post_vars = post_var_name_types
        
    if pre_spike_syn_code is not None:
        body["get_pre_spike_syn_code"] =\
            lambda self: dedent(_upgrade_code_string(pre_spike_syn_code,
                                                     class_name))

    if pre_event_syn_code is not None:
        body["get_pre_event_syn_code"] =\
            lambda self: dedent(_upgrade_code_string(pre_event_syn_code,
                                                     class_name))

    if post_event_syn_code is not None:
        body["get_post_event_syn_code"] =\
            lambda self: dedent(_upgrade_code_string(post_event_syn_code,
                                                     class_name))

    if post_spike_syn_code is not None:
        body["get_post_spike_syn_code"] =\
            lambda self: dedent(_upgrade_code_string(post_spike_syn_code,
                                                     class_name))

    if synapse_dynamics_code is not None:
        body["get_synapse_dynamics_code"] =\
            lambda self: dedent(_upgrade_code_string(synapse_dynamics_code,
                                                     class_name))

    if pre_event_threshold_condition_code is not None:
        body["get_pre_event_threshold_condition_code"] = \
            lambda self: dedent(_upgrade_code_string(pre_event_threshold_condition_code,
                                                     class_name))
    
    if post_event_threshold_condition_code is not None:
        body["get_post_event_threshold_condition_code"] = \
            lambda self: dedent(_upgrade_code_string(post_event_threshold_condition_code,
                                                     class_name))

    if pre_spike_code is not None:
        body["get_pre_spike_code"] =\
            lambda self: dedent(_upgrade_code_string(pre_spike_code,
                                                     class_name))

    if post_spike_code is not None:
        body["get_post_spike_code"] =\
            lambda self: dedent(_upgrade_code_string(post_spike_code,
                                                     class_name))

    if pre_dynamics_code is not None:
        body["get_pre_dynamics_code"] =\
            lambda self: dedent(_upgrade_code_string(pre_dynamics_code,
                                                     class_name))

    if post_dynamics_code is not None:
        body["get_post_dynamics_code"] =\
            lambda self: dedent(_upgrade_code_string(post_dynamics_code,
                                                     class_name))
    
    if vars is not None:
        body["get_vars"] = \
            lambda self: [Var(*vn) for vn in vars]
    
    if pre_vars is not None:
        body["get_pre_vars"] = \
            lambda self: [Var(*vn) for vn in pre_vars]

    if post_vars is not None:
        body["get_post_vars"] = \
            lambda self: [Var(*vn) for vn in post_vars]
    
    if pre_neuron_var_refs is not None:
        body["get_pre_neuron_var_refs"] =\
            lambda self: [VarRef(*v) for v in pre_neuron_var_refs]

    if post_neuron_var_refs is not None:
        body["get_post_neuron_var_refs"] =\
            lambda self: [VarRef(*v) for v in post_neuron_var_refs]

    if psm_var_refs is not None:
        body["get_psm_var_refs"] =\
            lambda self: [VarRef(*v) for v in psm_var_refs]
    
    return _create_model(class_name, WeightUpdateModelBase, params,
                         param_names, derived_params,
                         extra_global_params, body)


def create_current_source_model(class_name: str, params: ModelParamsType = None,
                                param_names=None, vars: ModelVarsType = None, var_name_types=None,
                                neuron_var_refs: ModelVarRefsType = None, 
                                derived_params: ModelDerivedParamsType = None,
                                injection_code: Optional[str] = None, 
                                extra_global_params: ModelEGPType = None):
    """Creates a new current source model.
    Within the ``injection_code`` code string, the variables, parameters,
    derived parameters, neuron variable references and extra global
    parameters defined in this model can all be referred to by name.
    Additionally, the code may refer to the following built-in read-only variables

    - ``dt`` which represents the simulation time step (as specified via  :meth:`.GeNNModel.dt`)
    - ``id`` which represents a neurons index within a population (starting from zero)
    - ``num_neurons`` which represents the number of neurons in the population
    
    Finally, the function ``injectCurrent(x)`` can be used to inject a current
    ``x`` into the attached neuron. The variable it goes into can be
    configured using the :attr:`CurrentSource.target_var`. It defaults to ``Isyn``.

    Args:
        class_name:             name of the new class (only for debugging)
        params:                 name and optional types of model parameters
        vars:                   names, types and optional variable access
                                modifiers of model variables
        neuron_var_refs:        names, types and optional variable access
                                of references to be assigned to variables
                                in neuron population current source is attached to
        derived_params:         names, types and callables to calculate
                                derived parameter values from params
        injection_code:         string containing the simulation code
                                statements to be run every timestep
        extra_global_params:    names and types of model
                                extra global parameters
    
    For example, we can define a simple current source that
    injects uniformly-distributed noise as follows:
    
    ..  code-block:: python

        uniform_noise_model = pygenn.create_current_source_model(
            "uniform_noise",
            params=["magnitude"],
            injection_code="injectCurrent(gennrand_uniform() * magnitude);")

    """
    body = {}
    if var_name_types is not None:
        warn("The 'var_name_types' parameter has been renamed to 'vars' "
             "and will be removed in future", FutureWarning)
        vars = var_name_types

    if injection_code is not None:
        body["get_injection_code"] =\
            lambda self: dedent(_upgrade_code_string(injection_code,
                                                     class_name))

    if vars is not None:
        body["get_vars"] = \
            lambda self: [Var(*vn) for vn in vars]
    
    if neuron_var_refs is not None:
        body["get_neuron_var_refs"] =\
            lambda self: [VarRef(*v) for v in var_refs]

    return _create_model(class_name, CurrentSourceModelBase, params,
                         param_names, derived_params,
                         extra_global_params, body)


def create_custom_update_model(class_name: str, params: ModelParamsType = None,
                               param_names=None, vars: CUModelVarsType = None, 
                               var_name_types=None, 
                               derived_params: ModelDerivedParamsType = None, 
                               var_refs: ModelVarRefsType = None, 
                               update_code: Optional[str] = None, 
                               extra_global_params: ModelEGPType = None,
                               extra_global_param_refs=None):
    """Creates a new custom update model.
    Within the ``update_code`` code string, the variables, parameters,
    derived parameters, variable references, extra global parameters 
    and extra global parameter references defined in this model can all be referred to by name.
    Additionally, the code may refer to the following built-in read-only variables

    - ``dt`` which represents the simulation time step (as specified via  :meth:`.GeNNModel.dt`)

     And, if a custom update using this model is attached to per-neuron variables:

    - ``id`` which represents a neurons index within a population (starting from zero)
    - ``num_neurons`` which represents the number of neurons in the population

    or, to per-synapse variables:

    - ``id_pre`` which represents the index of the presynaptic neuron (starting from zero)
    - ``id_post`` which represents the index of the postsynaptic neuron (starting from zero)
    - ``num_pre`` which represents the number of presynaptic neurons
    - ``num_post`` which represents the number of postsynaptic neurons

    Args:
        class_name:                 name of the new class (only for debugging)
        params:                     name and optional types of model parameters
        vars:                       names, types and optional variable access
                                    modifiers of model variables
        var_refs:                   names, types and optional variable access
                                    of references to be assigned to variables
                                    in population(s) custom update is attached to
        derived_params:             names, types and callables to calculate
                                    derived parameter values from params
        update_code:                string containing the code statements 
                                    to be run when custom update is launched
        extra_global_params:        names and types of model
                                    extra global parameters
        extra_global_param_refs:    names and types of extra global
                                    parameter references
    
    For example, we can define a custom update which will set a referenced variable to the value of a custom update model state variable:

    ..  code-block:: python

        reset_model = pygenn.create_custom_update_model(
            "reset",
            vars=[("v", "scalar", pygenn.CustomUpdateVarAccess.READ_ONLY)],
            var_refs=[("r", "scalar", pygenn.VarAccessMode.READ_WRITE)],
            update_code="r = v;")
    
    When used in a model with batch size > 1, whether custom updates of this sort are batched or not depends on the variables their references point to.
    If any referenced variables have :attr:`.VarAccess.READ_ONLY_DUPLICATE` or :attr:`.VarAccess.READ_WRITE` access modes, then the update will be batched 
    and any variables associated with the custom update with :attr:`.VarAccess.READ_ONLY_DUPLICATE` or :attr:`.VarAccess.READ_WRITE` access modes will be duplicated across the batches.
    
    Batch reduction
    ---------------
    As well as the standard variable access modes described previously, custom updates support variables with 'batch reduction' access modes 
    such as :attr:`.CustomUpdateVarAccess.REDUCE_BATCH_SUM` and :attr:`.CustomUpdateVarAccess.REDUCE_BATCH_MAX`.
    These access modes allow values read from variables duplicated across batches to be reduced into variables that are shared across batches.
    For example, in a gradient-based learning scenario, a model like this could be used to sum gradients from across all batches so they can be used as the input to a learning rule operating on shared synaptic weights:
    
    ..  code-block:: python

        reduce_model = pygenn.create_custom_update_model(
            "gradient_batch_reduce",
            vars=[("reducedGradient", "scalar", pygenn.CustomUpdateVarAccess.REDUCE_BATCH_SUM)],
            var_refs=[("gradient", "scalar", pygenn.VarAccessMode.READ_ONLY)],
            update_code=
                \"""
                reducedGradient = gradient;
                gradient = 0;
                \""")
    
    Batch reductions can also be performed into variable references with 
    the :attr:`.VarAccessMode.REDUCE_SUM` or :attr:`VarAccessMode.REDUCE_MAX` access modes.
    
    Neuron reduction
    ----------------
    Similarly to the batch reduction modes discussed previously, custom updates also support variables with several 'neuron reduction' access modes
    such as :attr:`.CustomUpdateVarAccess.REDUCE_NEURON_SUM` and :attr:`.CustomUpdateVarAccess.REDUCE_NEURON_MAX`.

    These access modes allow values read from per-neuron variables to be reduced into variables that are shared across neurons.
    For example, a model like this could be used to calculate the maximum value of a state variable in a population of neurons:

    ..  code-block:: python

        reduce_model = pygenn.create_custom_update_model(
            "neuron_reduce",
            vars=[("reduction", "scalar", pygenn.CustomUpdateVarAccess.REDUCE_NEURON_SUM)],
            var_refs=[("gradient", "scalar", pygenn.VarAccessMode.READ_ONLY)],
            update_code=
                \"""
                reduction = source;
                \""")

    Again, like batch reductions, neuron reductions can also be performed into variable references with 
    the :attr:`.VarAccessMode.REDUCE_SUM` or :attr:`VarAccessMode.REDUCE_MAX` access modes.
    """
    body = {}
    if var_name_types is not None:
        warn("The 'var_name_types' parameter has been renamed to 'vars' "
             "and will be removed in future", FutureWarning)
        vars = var_name_types

    if update_code is not None:
        body["get_update_code"] =\
            lambda self: dedent(_upgrade_code_string(update_code, class_name))

    if var_refs is not None:
        body["get_var_refs"] = lambda self: [VarRef(*v) for v in var_refs]

    if vars is not None:
        body["get_vars"] = \
            lambda self: [CustomUpdateVar(*vn) for vn in vars]

    if extra_global_param_refs is not None:
        body["get_extra_global_param_refs"] =\
            lambda self: [EGPRef(*e) for e in extra_global_param_refs]

    return _create_model(class_name, CustomUpdateModelBase, params,
                         param_names, derived_params,
                         extra_global_params, body)

def create_custom_connectivity_update_model(class_name: str, 
                                            params: ModelParamsType = None,
                                            vars: ModelVarsType = None, 
                                            pre_vars: ModelVarsType = None,
                                            post_vars: ModelVarsType = None,
                                            derived_params: ModelDerivedParamsType = None, 
                                            var_refs: ModelVarRefsType = None,
                                            pre_var_refs: ModelVarRefsType = None,
                                            post_var_refs: ModelVarRefsType = None,
                                            row_update_code: Optional[str] = None,
                                            host_update_code: Optional[str] = None,
                                            extra_global_params=None,
                                            extra_global_param_refs=None):
    """Creates a new custom connectivity update model.
    
    Within host update code, you have full access to parameters, derived parameters, 
    extra global parameters and pre and postsynaptic variables. By design you do
    not have access to per-synapse variables or variable references and, currently, 
    you cannot access pre and postsynaptic variable references as there are issues regarding delays. 
    Each variable has an accompanying push and pull function to copy it to and from the device. 
    For variables these have no parameters as illustrated in the example in :ref:`section-pull-push`, and for 
    extra global parameters they have a single parameter specifying the size of the array.
    Within the row update code you have full access to parameters, derived parameters,
    extra global parameters, presynaptic variables and presynaptic variables references.
    Postsynaptic and synaptic variables and variables references can only be accessed 
    from within one of the ``for_each_synapse`` loops illustrated below.
    Additionally, both the host and row update code cam refer to the following built-in 
    read-only variables:

    - ``dt`` which represents the simulation time step (as specified via  :meth:`.GeNNModel.dt`)
    - ``row_stride`` which represents the maximum number of synapses which each presynaptic neuron can have (this can be increased via :attr:`.SynapseGroup.max_connections`).
    - ``num_pre`` which represents the number of presynaptic neurons
    - ``num_post`` which represents the number of postsynaptic neurons
    
    Host code can also access the current number of synapses emanating from each presynaptic
    neuron using the ``row_length`` array whereas, in row-update code, this contains the number of
    synapses emanating from the current presynaptic neuron (identified by ``id_pre``).

    Args:
        class_name:                 name of the new class (only for debugging)
        params:                     name and optional types of model parameters
        vars:                       names, types and optional variable access
                                    modifiers of per-synapse model variables
        pre_vars:                   names, types and optional variable access
                                    modifiers of per-presynaptic neuron model variables
        post_vars                   names, types and optional variable access
                                    modifiers of per-postsynaptic neuron model variables
        derived_params:             names, types and callables to calculate
                                    derived parameter values from params
        var_refs:                   names, types and optional variable access
                                    of references to be assigned to synaptic variables
        pre_neuron_var_refs:        names, types and optional variable access
                                    of references to be assigned to presynaptic
                                    neuron variables
        post_neuron_var_refs:       names, types and optional variable access
                                    of references to be assigned to postsynaptic
                                    neuron variables
        row_update_code:            string containing the code statements 
                                    to be run when custom update is launched
        host_update_code:           string containing the code statements to be run
                                    on CPU when custom connectivity update is launched
        extra_global_params:        names and types of model
                                    extra global parameters
        extra_global_param_refs:    names and types of extra global
                                    parameter references

    Parallel synapse iteration and removal
    --------------------------------------
    The main GPU operation that custom connectivity updates expose is the ability to generate per-presynaptic neuron update code. This can be used to implement a very simple model which removes 'diagonals' from the connectivity matrix:

    ..  code-block:: python

        remove_diagonal_model = pygenn.create_custom_connectivity_update_model(
            "remove_diagonal",
            row_update_code=
                \"""
                for_each_synapse {
                    if(id_post == id_pre) {
                        remove_synapse();
                        break;
                    }
                }
                \""")

    Parallel synapse creation
    -------------------------
    Similarly you could implement a custom connectivity model which adds diagonals back into the connection matrix like this:

    ..  code-block:: python

        add_diagonal_model = pygenn.create_custom_connectivity_update_model(
            "add_diagonal",
            row_update_code=
                \"""
                add_synapse(id_pre);
                \""")

    One important issue here is that lots of other parts of the model (e.g. other custom connectivity updates or custom weight updates) *might* have state variables 'attached' to the same connectivity that the custom update is modifying. GeNN will automatically detect this and add and shuffle all these variables around accordingly which is fine for removing synapses but has no way of knowing what value to add synapses with. If you want new synapses to be created with state variables initialised to values other than zero, you need to use variables references to hook them to the custom connectivity update. For example, if you wanted to be able to provide weights for your new synapse, you could update the previous example model like:

    ..  code-block:: python

        add_diagonal_model = pygenn.create_custom_connectivity_update_model(
            "add_diagonal",
            var_refs=[("g", "scalar")],
            row_update_code=
                \"""
                add_synapse(id_pre, 1.0);
                \""")

    Host updates
    ------------
    Some common connectivity update scenarios involve some computation which can't be easily parallelized. If, for example you wanted to determine which elements on each row you wanted to remove on the host, you can include ``host_update_code`` which gets run before the row update code:

    ..  code-block:: python

        remove_diagonal_model = pygenn.create_custom_connectivity_update_model(
            "remove_diagonal",
            pre_var_name_types=[("postInd", "unsigned int")],
            row_update_code=
                \"""
                for_each_synapse {
                    if(id_post == postInd) {
                        remove_synapse();
                        break;
                    }
                }
                \""",
            host_update_code=
                \"""
                for(unsigned int i = 0; i < num_pre; i++) {
                   postInd[i] = i;
                }
                pushpostIndToDevice();
                \""")

    """
    body = {}

    if row_update_code is not None:
        body["get_row_update_code"] =\
            lambda self: dedent(_upgrade_code_string(row_update_code,
                                                     class_name))

    if host_update_code is not None:
        body["get_host_update_code"] =\
            lambda self: dedent(_upgrade_code_string(host_update_code,
                                                     class_name))

    if vars is not None:
        body["get_vars"] = \
            lambda self: [Var(*vn) for vn in vars]

    if pre_vars is not None:
        body["get_pre_vars"] = \
            lambda self: [Var(*vn) for vn in pre_vars]

    if post_vars is not None:
        body["get_post_vars"] = \
            lambda self: [Var(*vn) for vn in post_vars]

    if var_refs is not None:
        body["get_var_refs"] = lambda self: [VarRef(*v) for v in var_refs]

    if pre_var_refs is not None:
        body["get_pre_var_refs"] = \
            lambda self: [VarRef(*v) for v in pre_var_refs]

    if post_var_refs is not None:
        body["get_post_var_refs"] = \
            lambda self: [VarRef(*v) for v in post_var_refs]

    if extra_global_param_refs is not None:
        body["get_extra_global_param_refs"] =\
            lambda self: [EGPRef(*e) for e in extra_global_param_refs]

    return _create_model(class_name, CustomConnectivityUpdateModelBase,
                         params, None, derived_params,
                         extra_global_params, body)


def create_var_init_snippet(class_name: str, params: ModelParamsType = None,
                            param_names=None, 
                            derived_params: ModelDerivedParamsType = None,
                            var_init_code: Optional[str] = None,
                            extra_global_params=None):
    """Creates a new variable initialisation snippet.
    Within the ``var_init_code``, the parameters, derived parameters and
    extra global parameters defined in this snippet can all be referred to by name.
    Additionally, the code may refer to the following built-in read-only variables

    - ``dt`` which represents the simulation time step (as specified via  :meth:`.GeNNModel.dt`)

    And, if the snippet is used to initialise a per-neuron variable:

    - ``id`` which represents a neurons index within a population (starting from zero)
    - ``num_neurons`` which represents the number of neurons in the population

    or, a per-synapse variable:

    - ``id_pre`` which represents the index of the presynaptic neuron (starting from zero)
    - ``id_post`` which represents the index of the postsynaptic neuron (starting from zero)
    - ``num_pre`` which represents the number of presynaptic neurons
    - ``num_post`` which represents the number of postsynaptic neurons

    Finally, the variable being initialised is represented by
    the write-only ``value`` variable.

    Args:
        class_name:             name of the new model (only for debugging)
        params:                 name and optional types of model parameters
        derived_params:         names, types and callables to calculate
                                derived parameter values from paramss
        var_init_code:          string containing the code statements
                                required to initialise the variable
        extra_global_params:    names and types of model
                                extra global parameters

    For example, if we wanted to define a snippet to initialise variables by sampling from a normal distribution, 
    redrawing if the value is negative (which could be useful to ensure delays remain causal):

    ..  code-block:: python

        normal_positive_model = pygenn.create_var_init_snippet(
            'normal_positive',
            params=['mean', 'sd'],
            var_init_code=
                \"""
                scalar normal;
                do {
                    normal = mean + (gennrand_normal() * sd);
                } while (normal < 0.0);
                value = normal;
                \""")
    """
    body = {}

    if var_init_code is not None:
        body["get_code"] =\
            lambda self: dedent(_upgrade_code_string(var_init_code,
                                                     class_name))

    return _create_model(class_name, InitVarSnippetBase,
                         params, param_names, derived_params,
                         extra_global_params, body)


def create_sparse_connect_init_snippet(class_name: str, params=None, 
                                       param_names: ModelParamsType = None, 
                                       derived_params: ModelDerivedParamsType = None,
                                       row_build_code: Optional[str] = None,
                                       col_build_code: Optional[str] =None,
                                       calc_max_row_len_func: Optional[Callable] = None,
                                       calc_max_col_len_func: Optional[Callable] = None,
                                       calc_kernel_size_func: Optional[Callable] = None,
                                       extra_global_params: ModelEGPType = None):
    """Creates a new sparse connectivity initialisation snippet.
    Within the code strings, the parameters, derived parameters and
    extra global parameters defined in this snippet can all be referred to by name.
    Additionally, the code may refer to the following built-in read-only variables
    
    - ``dt`` which represents the simulation time step (as specified via  :meth:`.GeNNModel.dt`)
    - ``num_pre`` which represents the number of presynaptic neurons
    - ``num_post`` which represents the number of postsynaptic neurons
    - ``thread`` when some procedural connectivity is used with multiple 
      threads per presynaptic neuron, represents the index of the current thread

    and, in ``row_build_code``:

    - ``id_pre`` represents the index of the presynaptic neuron (starting from zero)
    - ``id_post_begin`` when some procedural connectivity is used with multiple 
      threads per presynaptic neuron, represents the index of the first postsynaptic neuron to connect.
    
    and, in ``col_build_code``:

    - ``id_post`` which represents the index of the postsynaptic neuron (starting from zero).

    Finally, the function ``addSynapse(x)`` can be used to add a new synapse to the connectivity 
    where, in ``row_build_code``, ``x`` is the index of the postsynaptic neuron to connect ``id_pre`` to
    and, in ``col_build_code``, ``x`` is the index of the presynaptic neuron to connect to ``id_post``
    
    Args:
        class_name:             name of the snippet (only for debugging)
        params:                 name and optional types of model parameters
        derived_params:         names, types and callables to calculate
                                derived parameter values from paramss
        row_build_code:         code for building connectivity row by row
        col_build_code:         code for building connectivity column by column
        calc_max_row_len_func:  used to calculate the maximum
                                row length of the synaptic matrix created using this snippet
        calc_max_col_len_func:  used to calculate the maximum
                                column length of the synaptic matrix created using this snippet
        calc_kernel_size_func:  used to calculate the size of the kernel if snippet requires one
        extra_global_params:    names and types of snippet extra global parameters
    
    For example, if we wanted to define a snippet to initialise connectivity where each
    presynaptic neuron targets a fixed number of postsynaptic neurons, sampled uniformly 
    with replacement, we could define a snippet as follows:
    
    ..  code-block:: python
        
        from scipy.stats import binom

        fixed_number_post = pygenn.create_sparse_connect_init_snippet(
            "fixed_number_post",
            params=[("num", "unsigned int")],
            row_build_code=
                \"""
                for(unsigned int c = num; c != 0; c--) {
                    const unsigned int idPost = gennrand() % num_post;
                    addSynapse(idPost + id_post_begin);
                }
                \""",
            calc_max_row_len_func=lambda num_pre, num_post, pars: pars["num"],
            calc_max_col_len_func=lambda num_pre, num_post, pars: binom.ppf(0.9999 ** (1.0 / num_post),
                                                                            pars["num"] * num_pre,
                                                                            1.0 / num_post))

    For full details of how maximum column lengths are calculated, you should refer to our paper [Knight2018]_ but, 
    in short, the number of connections that end up in a column are distributed binomially with :math:`n=\\text{num}` and :math:`p=\\frac{1}{\\text{num_post}}`
    Therefore, we can calculate the maximum column length by looking at the inverse cummulative distribution function (CDF) for the binomial distribution,
    looking at the point in the inverse CDF where there is a 0.9999 chance of the bound being correct when drawing synapses from ``num_post`` columns.
    """

    body = {}

    if row_build_code is not None:
        body["get_row_build_code"] =\
            lambda self: dedent(_upgrade_code_string(row_build_code,
                                                     class_name))

    if col_build_code is not None:
        body["get_col_build_code"] =\
            lambda self: dedent(_upgrade_code_string(col_build_code,
                                                     class_name))

    if calc_max_row_len_func is not None:
        body["get_calc_max_row_length_func"] = \
            lambda self: _wrap_max_length_lambda(calc_max_row_len_func)

    if calc_max_col_len_func is not None:
        body["get_calc_max_col_length_func"] = \
            lambda self: _wrap_max_length_lambda(calc_max_col_len_func)

    if calc_kernel_size_func is not None:
        body["get_calc_kernel_size_func"] = \
            lambda self: _wrap_kernel_size_lambda(calc_kernel_size_func)

    return _create_model(class_name, InitSparseConnectivitySnippetBase, params,
                         param_names, derived_params,
                         extra_global_params, body)

def create_toeplitz_connect_init_snippet(class_name: str, params: ModelParamsType=None,
                                         param_names=None,
                                         derived_params: ModelDerivedParamsType = None,
                                         diagonal_build_code: Optional[str] = None,
                                         calc_max_row_len_func: Optional[Callable] = None,
                                         calc_kernel_size_func: Optional[Callable] = None,
                                         extra_global_params: ModelEGPType = None):
    """Creates a new Toeplitz connectivity initialisation snippet.
    Each *diagonal* of Toeplitz connectivity is initialised independently by running the 
    snippet of code specified using the ``diagonal_build_code``.
    Within the code strings, the parameters, derived parameters and
    extra global parameters defined in this snippet can all be referred to by name.
    Additionally, the code may refer to the following built-in read-only variables
    
    - ``dt`` which represents the simulation time step (as specified via  :meth:`.GeNNModel.dt`)
    - ``num_pre`` which represents the number of presynaptic neurons
    - ``num_post`` which represents the number of postsynaptic neurons
    - ``id_diag`` when some procedural connectivity is used with multiple threads

    Additionally, the function ``addSynapse(id_post, id_kern_0, id_kern_1, ..., id_kern_N)`` 
    can be used to generate a new synapse to postsynaptic neuron ``id_post`` using
    N-dimensional kernel variables indexed with ``id_kern_0, id_kern_1, ..., id_kern_N``.
    Finally the ``for_each_synapse{}`` construct can be used to loop through incoming spikes
    and, inside this, ``id_pre`` will represent the index of the spiking presynaptic neuron.
    
    Args:
        class_name:             name of the snippet (only for debugging)
        params:                 name and optional types of model parameters
        derived_params:         names, types and callables to calculate
                                derived parameter values from paramss
        diagonal_build_code:    code for building connectivity row by row
        calc_max_row_len_func:  used to calculate the maximum
                                row length of synaptic matrix created using this snippet
        calc_kernel_size_func:  used to calculate the size of the kernel
        extra_global_params:    names and types of snippet extra global parameters
    
    For example, the following Toeplitz connectivity initialisation snippet could be used to 
    convolve a :math:`\\text{kern_dim} \\times \\text{kern_dim}` square kernel with the spikes from a population of :math:`\\text{pop_dim} \\times \\text{pop_dim}` neurons.
    
    ..  code-block:: python
        
        simple_conv2d_model = pynn.create_toeplitz_connect_init_snippet(
            "simple_conv2d",
            params=[("kern_size", "int"), ("pop_dim", "int")],
            diagonal_build_code=
                \"""
                const int kernRow = id_diag / kern_dim;
                const int kernCol = id_diag % kern_dim;

                for_each_synapse {
                    const int preRow = id_pre / pop_dim;
                    const int preCol = id_pre % pop_dim;
                    // If we haven't gone off edge of output
                    const int postRow = preRow + kernRow - 1;
                    const int postCol = preCol + kernCol - 1;
                    if(postRow >= 0 && postCol >= 0 && postRow < pop_dim && postCol < pop_dim) {
                        // Calculate postsynaptic index
                        const int postInd = (postRow * pop_dim) + postCol;
                        addSynapse(postInd,  kernRow, kernCol);
                    }
                }
                \""",

            calc_max_row_len_func=lambda num_pre, num_post, pars: pars["kern_size"] * pars["kern_size"],
            calc_kernel_size_func=lambda pars: [pars["kern_size"], pars["kern_size"]])
    
    For full details of how convolution-like connectivity is expressed in this way, please see our paper [Turner2022]_.
    """

    body = {}

    if diagonal_build_code is not None:
        body["get_diagonal_build_code"] =\
            lambda self: dedent(_upgrade_code_string(diagonal_build_code,
                                                     class_name))

    if calc_max_row_len_func is not None:
        body["get_calc_max_row_length_func"] = \
            lambda self: _wrap_max_length_lambda(calc_max_row_len_func)

    if calc_kernel_size_func is not None:
        body["get_calc_kernel_size_func"] = \
            lambda self: _wrap_kernel_size_lambda(calc_kernel_size_func)

    return _create_model(class_name, InitToeplitzConnectivitySnippetBase,
                         params, param_names, derived_params,
                         extra_global_params, body)

@deprecated("This wrapper is now unnecessary - use callables directly")
def create_dpf_class(dp_func):
    """Helper function to create derived parameter function class

    Args:
    dp_func --  a function which computes the derived parameter and takes
                two args "pars" (vector of double) and "dt" (double)
    """
    return lambda: dp_func

@deprecated("This wrapper is now unnecessary - use callables directly")
def create_cmlf_class(cml_func):
    """Helper function to create function class for calculating sizes of
    matrices initialised with sparse connectivity initialisation snippet

    Args:
    cml_func -- a function which computes the length and takes
                three args "num_pre" (unsigned int), "num_post" (unsigned int)
                and "pars" (vector of double)
    """
    return lambda: cml_func

@deprecated("This wrapper is now unnecessary - use callables directly")
def create_cksf_class(cks_func):
    """Helper function to create function class for calculating sizes 
    of kernels from connectivity initialiser parameters 

    Args:
    cks_func -- a function which computes the kernel size and takes
                one arg "pars" (vector of double)
    """
    return lambda: cks_func

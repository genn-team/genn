## @namespace pygenn.genn_model
"""
This module provides the GeNNModel class to simplify working with pygenn module and
helper functions to derive custom model classes.

GeNNModel should be used to configure a model, build, load and
finally run it. Recording is done manually by pulling from the population of
interest and then copying the values from Variable.view attribute. Each
simulation step must be triggered manually by calling step_time function.

Example:
The following example shows in a (very) simplified manner how to build and
run a simulation using GeNNModel::

    from pygenn import GeNNModel
    gm = GeNNModel("float", "test")

    # add populations
    neuron_pop = gm.add_neuron_population(_parameters_truncated_)
    syn_pop = gm.add_synapse_population(_parameters_truncated_)

    # build and load model
    gm.build()
    gm.load()

    Vs = numpy.empty((simulation_length, population_size))
    # Variable.view provides a view into a raw C array
    # here a Variable call V (voltage) will be recorded
    v_view = neuron_pop.vars["V"].view

    # run a simulation for 1000 steps
    for i in range 1000:
        # manually trigger one simulation step
        gm.step_time()
        # when you pull state from device, views of all variables
        # are updated and show current simulated values
        neuron_pop.pull_state_from_device()
        # finally, record voltage by copying form view into array.
        Vs[i,:] = v_view
"""
# python imports
from collections import OrderedDict
from deprecated import deprecated
from distutils.spawn import find_executable
from importlib import import_module
from os import path, environ
from platform import system
from psutil import cpu_count
from setuptools import msvc
from subprocess import check_call  # to call make
from typing import Optional, List, Sequence, Tuple, Union

import re
import sys
from textwrap import dedent
from warnings import warn
from weakref import proxy

# 3rd party imports
import numpy as np

# pygenn imports
from ._genn import (generate_code, init_logging, CurrentSource,
                    CurrentSourceModelBase, CustomConnectivityUpdate,
                    CustomConnectivityUpdateModelBase, CustomUpdate,
                    CustomUpdateModelBase, CustomUpdateVar, CustomUpdateWU,
                    DerivedParam, EGP, EGPRef,
                    InitSparseConnectivitySnippetBase,
                    InitToeplitzConnectivitySnippetBase, InitVarSnippetBase,
                    ModelSpecInternal, NeuronGroup, NeuronModelBase,
                    NumericValue, Param, ParamVal, PlogSeverity,
                    PostsynapticInit, PostsynapticModelBase, ResolvedType,
                    SparseConnectivityInit, SynapseGroup, SynapseMatrixType,
                    ToeplitzConnectivityInit, UnresolvedType, Var, VarAccess,
                    VarAccessMode, VarInit, VarLocation, VarRef, WeightUpdateInit,
                    WeightUpdateModelBase)

from .runtime import Runtime
from ._genn_groups import (CurrentSourceMixin, CustomConnectivityUpdateMixin,
                           CustomUpdateMixin, CustomUpdateWUMixin,
                           NeuronGroupMixin, SynapseGroupMixin)
from ._model_preprocessor import get_snippet, get_var_init, prepare_param_vals
from . import (current_source_models, custom_connectivity_update_models,
               custom_update_models, init_sparse_connectivity_snippets, 
               init_toeplitz_connectivity_snippets, init_var_snippets,
               neuron_models, postsynaptic_models, types, weight_update_models)

TypeType = Union[str, ResolvedType]
ModelParamsType = Optional[Sequence[Union[str, Tuple[str, TypeType]]]]
ModelVarsType = Optional[Sequence[Union[Tuple[str, TypeType],
                                        Tuple[str, TypeType, VarAccess]]]]
ModelVarRefsType = Optional[Sequence[Union[Tuple[str, TypeType],
                                          Tuple[str, TypeType, VarAccessMode]]]]
ModelEGPType = Optional[Sequence[Tuple[str, TypeType]]]

# Dynamically add Python mixin to wrapped class
CurrentSource.__bases__ += (CurrentSourceMixin,)
CustomConnectivityUpdate.__bases__ += (CustomConnectivityUpdateMixin,)
CustomUpdate.__bases__ += (CustomUpdateMixin,)
CustomUpdateWU.__bases__ += (CustomUpdateWUMixin,)
NeuronGroup.__bases__ += (NeuronGroupMixin,)
SynapseGroup.__bases__ += (SynapseGroupMixin,)

# If we're on windows
if system() == "Windows":
    # Get environment and cache in class, convertings
    # all keys to upper-case for consistency
    _msvc_env = msvc.msvc14_get_vc_env("x86_amd64")
    _msvc_env = {k.upper(): v for k, v in _msvc_env.items()}
    
    # Update process's environment with this
    # **NOTE** this handles both child processes (manually launching msbuild)
    # and stuff within this process (running the code generator)
    environ.update(_msvc_env)
    
    # Find MSBuild in path
    # **NOTE** we need to do this because setting the path via 
    # check_call's env kwarg does not effect finding the executable
    # **NOTE** shutil.which would be nicer, but isn't in Python < 3.3
    _msbuild = find_executable("msbuild",  _msvc_env["PATH"])

    # If Python version is newer than 3.8 and CUDA path is in environment
    if sys.version_info >= (3, 8) and "CUDA_PATH" in environ:
        # Add CUDA bin directory to DLL search directories
        from os import add_dll_directory
        add_dll_directory(path.join(environ["CUDA_PATH"], "bin"))


# Loop through backends in preferential order
backend_modules = OrderedDict()
for b in ["cuda", "single_threaded_cpu", "opencl"]:
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

class GeNNModel(ModelSpecInternal):
    """GeNNModel class
    This class helps to define, build and run a GeNN model from python
    """

    def __init__(self, precision: TypeType = "float",
                 model_name: str = "GeNNModel",
                 backend=None, time_precision: Optional[TypeType] = None,
                 genn_log_level: PlogSeverity = PlogSeverity.WARNING,
                 code_gen_log_level: PlogSeverity = PlogSeverity.WARNING,
                 transpiler_log_level: PlogSeverity = PlogSeverity.WARNING,
                 runtime_log_level: PlogSeverity = PlogSeverity.WARNING,
                 backend_log_level: PlogSeverity = PlogSeverity.WARNING,
                 **preference_kwargs):
        """Init GeNNModel
        Keyword args:
        precision               -- string precision as string ("float" or "double"). 
        model_name              -- string name of the model. Defaults to "GeNNModel".
        backend                 -- string specifying name of backend module to use
                                   Defaults to one to pick 'best' backend for your system
        time_precision          -- string time precision as string ("float" or "double")
        genn_log_level          -- Log level for GeNN
        code_gen_log_level      -- Log level for GeNN code-generator
        transpiler_log_level    -- Log level for GeNN transpiler
        runtime_log_level       -- Log level for GeNN runtime
        backend_log_level       -- Log level for backend
        preference_kwargs       -- Additional keyword arguments to set in backend preferences structure
        """
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
    def backend_name(self):
        return self._backend_name

    @backend_name.setter
    def backend_name(self, backend_name):
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
    @deprecated("The name of this property was inconsistent, use dt instead",
                category=FutureWarning)
    def dT(self):
        return self.dt
    
    @dT.setter
    @deprecated("The name of this property was inconsistent, use dt instead",
                category=FutureWarning)
    def dT(self, dt):
        self.dt = dt
    
    # **TODO** is there a better way of exposing inner class properties?
    @property
    def t(self):
        """Simulation time in ms"""
        return self._runtime.time

    @property
    def timestep(self):
        """Simulation time step"""
        return self._runtime.timestep

    @timestep.setter
    def timestep(self, timestep):
        self._runtime.timestep = timestep

    @property
    def free_device_mem_bytes(self):
        return self._runtime.free_device_mem_bytes;

    @property
    def neuron_update_time(self):
        return self._runtime.neuron_update_time

    @property
    def init_time(self):
        return self._runtime.init_time

    @property
    def presynaptic_update_time(self):
        return self._runtime.presynaptic_update_time

    @property
    def postsynaptic_update_time(self):
        return self._runtime.postsynaptic_update_time

    @property
    def synapse_dynamics_time(self):
        return self._runtime.synapse_dynamics_time

    @property
    def init_sparse_time(self):
        return self._runtime.init_sparse_time

    def get_custom_update_time(self, name):
        return self._runtime.get_custom_update_time(name)

    def get_custom_update_transpose_time(self, name):
        return self._runtime.get_custom_update_transpose_time(name)

    def add_neuron_population(self, pop_name, num_neurons, neuron,
                              params={}, vars={}):
        """Add a neuron population to the GeNN model

        Args:
        pop_name    --  name of the new population
        num_neurons --  number of neurons in the new population
        neuron      --  type of the NeuronModels class as string or instance of
                        neuron class derived from
                        ``pygenn.genn_wrapper.NeuronModels.Custom`` (see also
                        pygenn.genn_model.create_neuron_model)
        params      --  dict with param values for the NeuronModels class
        vars        --  dict with initial variable values for the
                        NeuronModels class
        """
        if self._built:
            raise Exception("GeNN model already built")

        # Resolve neuron model
        neuron = get_snippet(neuron, NeuronModelBase, neuron_models)
        
        # Extract parts of vars which should be initialised by GeNN
        var_init = get_var_init(vars)
        
        # Use superclass to add population
        n_group = super(GeNNModel, self).add_neuron_population(
            pop_name, int(num_neurons), neuron, 
            prepare_param_vals(params), var_init)
        
        # Initialise group, store group in dictionary and return
        n_group._init_group(self, vars)
        self.neuron_populations[pop_name] = n_group
        return n_group

    def add_synapse_population(self, pop_name, matrix_type,
                               source, target, weight_update_init,
                               postsynaptic_init, connectivity_init=None):
        """Add a synapse population to the GeNN model

        Args:
        pop_name             --  name of the new population
        matrix_type          --  SynapseMatrixType describing type of the matrix
        source               --  source neuron group (NeuronGroup object)
        target               --  target neuron group (NeuronGroup object)
        
        connectivity_init    --  SparseConnectivityInit or 
                                 ToeplitzConnectivityInit used to 
                                 configure connectivity
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
        s_group = super(GeNNModel, self).add_synapse_population(
            pop_name, matrix_type, source, target, 
            weight_update_init[0], postsynaptic_init[0],
            connectivity_init)

        # Initialise group, store group in dictionary and return
        s_group._init_group(self, postsynaptic_init[1], weight_update_init[1],
                            weight_update_init[2], weight_update_init[3],
                            source, target)
        self.synapse_populations[pop_name] = s_group
        return s_group

    def add_current_source(self, cs_name, current_source_model, pop,
                           params={}, vars={}, var_refs = {}):
        """Add a current source to the GeNN model

        Args:
        cs_name                 --  name of the new current source
        current_source_model    --  type of the CurrentSourceModels class as
                                    string or instance of CurrentSourceModels
                                    class derived from
                                    ``pygenn.genn_wrapper.CurrentSourceModels.Custom`` (see also
                                    pygenn.genn_model.create_current_source_model)
        pop                     --  population into which the current source 
                                    should be injected (NeuronGroup object)
        params                  --  dict with param values for the
                                    CurrentSourceModel class
        vars                    --  dict with initial variable values for the
                                    CurrentSourceModel class
        var_refs                --  dict with variable references to pop for the
                                    CurrentSourceModel class
        """
        if self._built:
            raise Exception("GeNN model already built")

        # Resolve current source model
        current_source_model = get_snippet(current_source_model, CurrentSourceModelBase,
                                           current_source_models)
        
        # Extract parts of vars which should be initialised by GeNN
        var_init = get_var_init(vars)
        
        # Use superclass to add population
        c_source = super(GeNNModel, self).add_current_source(
            cs_name, current_source_model, pop, 
            prepare_param_vals(params), var_init, var_refs)
        
        # Initialise group, store group in dictionary and return
        c_source._init_group(self, vars, pop)
        self.current_sources[cs_name] = c_source
        return c_source
    
    def add_custom_update(self, cu_name, group_name, custom_update_model,
                          params={}, vars={}, var_refs={}, egp_refs={}):
        """Add a current source to the GeNN model

        Args:
        cu_name                 -- name of the new current source
        group_name              -- name of custom update group this
                                   update belongs to
        custom_update_model     -- type of the CustomUpdateModel class as
                                   string or instance of CustomUpdateModel
                                   class derived from
                                   ``CustomUpdateModelBase`` (see also
                                   pygenn.genn_model.create_custom_update_model)
        params                  -- dict with param values for the
                                   CustomUpdateModel class
        vars                    -- dict with initial variable values for the
                                   CustomUpdateModel class
        var_refs                -- dict with variable references for the
                                   CustomUpdateModel class
        egp_refs                -- dict with extra global parameter references 
                                   for the CustomUpdateModel class
        """
        if self._built:
            raise Exception("GeNN model already built")
        
        # Resolve custom update model
        custom_update_model = get_snippet(custom_update_model, CustomUpdateModelBase,
                                          custom_update_models)
        
        # Extract parts of vars which should be initialised by GeNN
        var_init = get_var_init(vars)

        # Use superclass to add population
        c_update = super(GeNNModel, self).add_custom_update(
            cu_name, group_name, custom_update_model,
            prepare_param_vals(params), var_init, var_refs, egp_refs)

        # Setup back-reference, store group in dictionary and return
        c_update._init_group(self, vars)
        self.custom_updates[cu_name] = c_update
        return c_update
    
    def add_custom_connectivity_update(self, cu_name, group_name, syn_group,
                                       custom_conn_update_model,
                                       params={}, vars={}, pre_vars={},
                                       post_vars={}, var_refs={},
                                       pre_var_refs={}, post_var_refs={},
                                       egp_refs={}):
        """Add a custom connectivity update to the GeNN model

        Args:
        cu_name                     -- name of the new custom connectivity update
        group_name                  -- name of group this custom connectivity update
                                       should be performed within
        syn_group                   -- synapse group this custom connectivity
                                       update should be attached to
                                       (SynapseGroup object)
        custom_conn_update_model    -- type of the CustomConnetivityUpdateModel 
                                       class as string or instance of 
                                       CustomConnectivityUpdateModel class 
                                       derived from 
                                       ``pygenn.genn_wrapper.CustomConnectivityUpdateModel.Custom``
                                       (see also pygenn.genn_model.create_custom_connectivity_update_model)
        params                      -- dict with param values for the
                                       CustomConnectivityUpdateModel class
        vars                        -- dict with initial variable values for the
                                       CustomConnectivityUpdateModel class
        pre_vars                    -- dict with initial presynaptic variable
                                       values for the 
                                       CustomConnectivityUpdateModel class
        post_vars                   -- dict with initial postsynaptic variable
                                       values for the
                                       CustomConnectivityUpdateModel class
        var_refs                    -- dict with variable references for the
                                       CustomConnectivityUpdateModel class
        pre_var_refs                -- dict with presynaptic variable
                                       references for the 
                                       CustomConnectivityUpdateModel class
        post_var_refs               -- dict with postsynaptic variable
                                       references for the
                                       CustomConnectivityUpdateModel class
        """
        if self._built:
            raise Exception("GeNN model already built")

        # Resolve custom update model
        custom_connectivity_update_model = get_snippet(
            custom_conn_update_model, CustomConnectivityUpdateModelBase,
            custom_connectivity_update_models)

        # Extract parts of vars which should be initialised by GeNN
        var_init = get_var_init(vars)
        pre_var_init = get_var_init(pre_vars)
        post_var_init = get_var_init(post_vars)

        # Use superclass to add population
        c_update = super(GeNNModel, self).add_custom_connectivity_update(
            cu_name, group_name, syn_group, custom_connectivity_update_model,
            prepare_param_vals(params), var_init, pre_var_init, post_var_init,
            var_refs, pre_var_refs, post_var_refs, egp_refs)

        # Setup back-reference, store group in dictionary and return
        c_update._init_group(self, vars, pre_vars, post_vars)
        self.custom_connectivity_updates[cu_name] = c_update
        return c_update
        
    def build(self, path_to_model="./", always_rebuild=False, never_rebuild=False):
        """Finalize and build a GeNN model

        Keyword args:
        path_to_model   --  path where to place the generated model code.
                            Defaults to the local directory.
        always_rebuild   -- should model be rebuilt even if
                            it doesn't appear to be required
        never_rebuild   --  should model never be rebuilt even it appears to
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
        self.finalise()

        # Create suitable preferences object for backend
        self._preferences = self._backend_module.Preferences()

        # Set attributes on preferences object from kwargs
        for k, v in self._preference_kwargs.items():
            if hasattr(self._preferences, k):
                setattr(self._preferences, k, v)
        
        # Create backend
        self._backend = self._backend_module.create_backend(
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

    def load(self, path_to_model="./", num_recording_timesteps=None):
        """import the model as shared library and initialize it"""
        if self._loaded:
            raise Exception("GeNN model already loaded")
        self._path_to_model = path_to_model
        
        # Create runtime
        self._runtime = Runtime(self._path_to_model, self._model_merged,
                                self._backend)
        
        # If model uses recording system and recording timesteps is not set
        if self.recording_in_use and num_recording_timesteps is None:
            raise Exception("Cannot use recording system without passing "
                            "number of recording timesteps to GeNNModel.load")

        # Allocate memory
        self._runtime.allocate(num_recording_timesteps)

        # Loop through neuron populations and load any
        # extra global parameters required for initialization
        for pop_data in self.neuron_populations.values():
            pop_data.load_init_egps()

        # Loop through synapse populations and load any 
        # extra global parameters required for initialization
        for pop_data in self.synapse_populations.values():
            pop_data.load_init_egps()

        # Loop through current sources
        for src_data in self.current_sources.values():
            src_data.load_init_egps()

        # Loop through custom connectivity updates
        for cu_data in self.custom_connectivity_updates.values():
            cu_data.load_init_egps()

        # Loop through custom updates
        for cu_data in self.custom_updates.values():
            cu_data.load_init_egps()

        # Initialize model
        self._runtime.initialize()

        # Loop through neuron populations
        for pop_data in self.neuron_populations.values():
            pop_data.load(num_recording_timesteps)

        # Loop through synapse populations
        for pop_data in self.synapse_populations.values():
            pop_data.load()

        # Loop through current sources
        for src_data in self.current_sources.values():
            src_data.load()

        # Loop through custom connectivity updates
        for cu_data in self.custom_connectivity_updates.values():
            cu_data.load()

        # Loop through custom updates
        for cu_data in self.custom_updates.values():
            cu_data.load()

        # Now everything is set up call the sparse initialisation function
        self._runtime.initialize_sparse()

        # Set loaded flag and built flag
        self._loaded = True
        self._built = True


    def unload(self):
        # Loop through custom updates and unload
        for cu_data in self.custom_updates.values():
            cu_data.unload()
        
        # Loop through custom connectivity updates and unload
        for cu_data in self.custom_connectivity_updates.values():
            cu_data.unload()
    
        # Loop through current sources and unload
        for src_data in self.current_sources.values():
            src_data.unload()

        # Loop through synapse populations and unload
        for pop_data in self.synapse_populations.values():
            pop_data.unload()

        # Loop through neuron populations and unload
        for pop_data in self.neuron_populations.values():
            pop_data.unload()

        # Close runtime
        self._runtime = None

        # Clear loaded flag
        self._loaded = False

    def step_time(self):
        """Make one simulation step"""
        if not self._loaded:
            raise Exception("GeNN model has to be loaded before stepping")

        self._runtime.step_time()
    
    def custom_update(self, name):
        """Perform custom update"""
        if not self._loaded:
            raise Exception("GeNN model has to be loaded before performing custom update")
            
        self._runtime.custom_update(name)
   

    def pull_recording_buffers_from_device(self):
        """Pull recording buffers from device"""
        if not self._loaded:
            raise Exception("GeNN model has to be loaded before pulling recording buffers")

        if not self.recording_in_use:
            raise Exception("Cannot pull recording buffer if recording system is not in use")

        # Pull recording buffers from device
        self._runtime.pull_recording_buffers_from_device()

def init_var(snippet, params={}):
    """This helper function creates a VarInit object
    to easily initialise a variable using a snippet.

    Args:
    snippet -- type of the InitVarSnippet class as string or
               instance of class derived from
               InitVarSnippetBase class.
    params  --  dict with param values for the InitVarSnippet class
    """
    # Get snippet and wrap in VarInit object
    snippet = get_snippet(snippet, InitVarSnippetBase, init_var_snippets)

    # Use add function to create suitable VarInit
    return VarInit(snippet, prepare_param_vals(params))

def init_sparse_connectivity(snippet, params={}):
    """This helper function creates a InitSparseConnectivitySnippet::Init
    object to easily initialise connectivity using a snippet.

    Args:
    snippet -- type of the InitSparseConnectivitySnippet
               class as string or instance of class
               derived from
               InitSparseConnectivitySnippetBase
    params  -- dict with param values for the
               InitSparseConnectivitySnippet class
    """
    # Get snippet and wrap in SparseConnectivityInit object
    snippet = get_snippet(snippet, InitSparseConnectivitySnippetBase,
                          init_sparse_connectivity_snippets)
    return SparseConnectivityInit(snippet, prepare_param_vals(params))

                                              

def init_postsynaptic(snippet, params={}, vars={}, var_refs={}):
    # Get snippet and wrap in PostsynapticInit object
    snippet = get_snippet(snippet, PostsynapticModelBase,
                          postsynaptic_models)
    
    # Extract parts of var spaces which should be initialised by GeNN
    var_init = get_var_init(vars)
    
    return (PostsynapticInit(snippet, prepare_param_vals(params), 
                             var_init, var_refs), 
            vars)

def init_weight_update(snippet, params={}, vars={}, pre_vars={},
                       post_vars={}, pre_var_refs={}, post_var_refs={}):
    # Get snippet and wrap in WeightUpdateInit object
    snippet = get_snippet(snippet, WeightUpdateModelBase,
                          weight_update_models)
    
    var_init = get_var_init(vars)
    pre_var_init = get_var_init(pre_vars)
    post_var_init = get_var_init(post_vars)
    
    return (WeightUpdateInit(snippet, prepare_param_vals(params), var_init, 
                             pre_var_init, post_var_init,
                             pre_var_refs, post_var_refs),
            vars, pre_vars, post_vars)

@deprecated("The name of this function was ambiguous, use init_sparse_connectivity instead",
            category=FutureWarning)
def init_connectivity(init_sparse_connect_snippet, params={}):
    """This helper function creates a InitSparseConnectivitySnippet::Init
    object to easily initialise connectivity using a snippet.

    Args:
    init_sparse_connect_snippet --  type of the InitSparseConnectivitySnippet
                                    class as string or instance of class
                                    derived from
                                    InitSparseConnectivitySnippetBase
    params                      --  dict with param values for the
                                    InitSparseConnectivitySnippet class
    """
    return init_sparse_connectivity(init_sparse_connect_snippet, params)

def init_toeplitz_connectivity(init_toeplitz_connect_snippet, params={}):
    """This helper function creates a InitToeplitzConnectivitySnippet::Init
    object to easily initialise connectivity using a snippet.

    Args:
    init_toeplitz_connect_snippet   -- type of the InitToeplitzConnectivitySnippet
                                       class as string or instance of class
                                       derived from
                                       InitSparseConnectivitySnippetBase
    params                          -- dict with param values for the
                                       InitToeplitzConnectivitySnippet class
    """
    # Get snippet and wrap in InitToeplitzConnectivitySnippet object
    init_toeplitz_connect_snippet = get_snippet(init_toeplitz_connect_snippet,
                                                InitToeplitzConnectivitySnippetBase,
                                                init_toeplitz_connectivity_snippets)
    return ToeplitzConnectivityInit(init_toeplitz_connect_snippet, 
                                    prepare_param_vals(params))

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
                        var_name_types=None, derived_params=None,
                        sim_code: Optional[str] = None,
                        threshold_condition_code: Optional[str] = None,
                        reset_code: Optional[str] = None,
                        extra_global_params: ModelEGPType = None,
                        additional_input_vars=None,
                        auto_refractory_required: bool = False):
    """Creates a new neuron model.
    Within all of the code strings, the variables, parameters,
    derived parameters, additional input variables and extra global
    parameters defined in this model can all be referred to be name.
    Additionally, the code may refer to the following built in read-only variables

    - ``dt`` which represents the simulation time step (as specified via  :meth:`.GeNNModel.dt`)
    - ``Isyn`` which represents the total incoming synaptic input.
    - ``id`` which represents a neurons index within a population (starting from zero)
    - ``num_neurons`` which represents the number of neurons in the population

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
                              derived_params=None,
                              sim_code: Optional[str] = None, decay_code=None,
                              apply_input_code=None,
                              extra_global_params: ModelEGPType = None):
    """Creates a new postsynaptic update model.
    Within all of the code strings, the variables, parameters,
    derived parameters and extra global parameters defined in this model
    can all be referred to be name. Additionally, the code may refer to the
    following built in read-only variables:

    - ``dt`` which represents the simulation time step (as specified via  :meth:`.GeNNModel.dt`)
    - ``id`` which represents a neurons index within a population (starting from zero)
    - ``num_neurons`` which represents the number of neurons in the population

    Finally, the function ``injectCurrent(x)`` can be used to inject a current
    ``x`` into the postsynaptic neuron. The variable it goes into can be
    configured using the :attr:`SynapseGroup.post_target_var``.

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
        derived_params=None, sim_code=None,
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

    where ``inc`` is the amount to add to the postsynaptic neuron for each pre-synaptic spike.
    Dendritic delays can also be inserted between the synapse and the postsynaptic neuron by using the ``addToPostDelay(inc, delay)`` function.
    For example,

    ..  code-block:: python

        pre_spike_syn_code="addToPostDelay(inc, delay);"

    where, once again, ``inc`` is the amount to add to the postsynaptic neuron and ``delay`` is the length of the dendritic delay in timesteps.
    By implementing ``delay`` as a weight update model variable, heterogeneous synaptic delays can be implemented.
    For an example, see WeightUpdateModels::StaticPulseDendriticDelay for a simple synapse update model with heterogeneous dendritic delays.

    When using dendritic delays, the *maximum* dendritic delay for a synapse populations must be specified via the
    :attr:`SynapseGroup.max_dendritic_delay_timesteps` property. One can also define synaptic effects that occur in the reverse direction,
    i.e. terms that are added to a target variable in the _presynaptic_ neuron using the ``addToPre(inc)`` function. For example,

    ..  code-block:: python

        pre_spike_syn_code="addToPre(inc * V_post);"

    would add terms ``inc * V_post`` to for each _outgoing_ synapse of a presynaptic neuron.
    Like postsynaptic models, by default these inputs are accumulated in ``Isyn`` in the presynaptic 
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
    
    return _create_model(class_name, WeightUpdateModelBase, params,
                         param_names, derived_params,
                         extra_global_params, body)


def create_current_source_model(class_name, params=None, param_names=None,
                                vars=None, var_name_types=None,
                                neuron_var_refs=None, derived_params=None,
                                injection_code=None, extra_global_params=None):
    """This helper function creates a custom NeuronModel class.
    See also:
    create_neuron_model
    create_weight_update_model
    create_current_source_model
    create_var_init_snippet
    create_sparse_connect_init_snippet

    Args:
    class_name          --  name of the new class

    Keyword args:
    param_names         --  list of strings with param names of the model
    vars                --  list of pairs of strings with varible names and
                            types of the model
    neuron_var_refs     --  references to neuron variables
    derived_params      --  list of pairs, where the first member is string
                            with name of the derived parameter and the second
                            should be a functor returned by create_dpf_class
    injection_code      --  string with the current injection code
    extra_global_params --  list of pairs of strings with names and types of
                            additional parameters
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


def create_custom_update_model(class_name, params=None, param_names=None, 
                               vars=None, var_name_types=None,
                               derived_params=None, var_refs=None, 
                               update_code=None, extra_global_params=None,
                               extra_global_param_refs=None):
    """This helper function creates a custom CustomUpdate class.
    See also:
    create_neuron_model
    create_weight_update_model
    create_current_source_model
    create_var_init_snippet
    create_sparse_connect_init

    Args:
    class_name              --  name of the new class

    Keyword args:
    param_names         --  list of strings with param names of the model
    var_name_types      --  list of tuples of strings with varible names and
                            types of the variable
    derived_params      --  list of tuples, where the first member is string
                            with name of the derived parameter and the second
                            should be a functor returned by create_dpf_class
    var_refs            --  list of tuples of strings with varible names and
                            types of variabled variable
    update_code         --  string with the current injection code
    extra_global_params --  list of pairs of strings with names and types of
                            additional parameters
    extra_global_param_refs --  list of pairs of strings with names and types of
                                extra global parameter references
    
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

def create_custom_connectivity_update_model(class_name, params=None,
                                            vars=None, pre_vars=None,
                                            post_vars=None,
                                            derived_params=None, 
                                            var_refs=None,
                                            pre_var_refs=None,
                                            post_var_refs=None,
                                            row_update_code=None,
                                            host_update_code=None,
                                            extra_global_params=None,
                                            extra_global_param_refs=None):
    """This helper function creates a custom CustomConnectivityUpdate class.
    See also:
    create_neuron_model
    create_weight_update_model
    create_current_source_model
    create_var_init_snippet
    create_sparse_connect_init

    Args:
    class_name          --  name of the new class

    Keyword args:
    param_names         --  list of strings with param names of the model
    vars                --  list of tuples of strings with variable names and
                            types of the variable
    pre_vars            --  list of tuples of strings with variable names and
                            types of presynaptic  variable
    post_vars           --  list of tuples of strings with variable names and
                            types of postsynaptic variable
    derived_params      --  list of tuples, where the first member is string
                            with name of the derived parameter and the second
                            should be a functor returned by create_dpf_class
    var_refs            --  list of tuples of strings with variable names and
                            types of synaptic variable references
    pre_var_refs        --  list of tuples of strings with variable names and
                            types of presynaptic variable references
    post_var_refs       --  list of tuples of strings with variable names and
                            types of postsynaptic variable references
    row_update_code     --  string with per-row update code
    host_update_code    --  string with host update code
    extra_global_params --  list of pairs of strings with names and types of
                            additional parameters
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
                            param_names=None, derived_params=None,
                            var_init_code: Optional[str] = None,
                            extra_global_params=None):
    """Creates a new variable initialisation snippet
    Within the ``var_init_code``, the parameters, derived parameters and
    extra global parameters defined in this snippet can all be referred to be name.
    Additionally, the code may refer to the following built in read-only variables

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
    """
    body = {}

    if var_init_code is not None:
        body["get_code"] =\
            lambda self: dedent(_upgrade_code_string(var_init_code,
                                                     class_name))

    return _create_model(class_name, InitVarSnippetBase,
                         params, param_names, derived_params,
                         extra_global_params, body)


def create_sparse_connect_init_snippet(class_name, params=None, 
                                       param_names=None, derived_params=None,
                                       row_build_code=None,
                                       col_build_code=None,
                                       calc_max_row_len_func=None,
                                       calc_max_col_len_func=None,
                                       calc_kernel_size_func=None,
                                       extra_global_params=None):
    """This helper function creates a custom
    InitSparseConnectivitySnippet class.
    See also:
    create_neuron_model
    create_weight_update_model
    create_postsynaptic_model
    create_current_source_model
    create_var_init_snippet

    Args:
    class_name              --  name of the new class

    Keyword args:
    param_names             --  list of strings with param names of the model
    derived_params          --  list of pairs, where the first member is string
                                with name of the derived parameter and the
                                second MUST be an instance of the class which
                                inherits from pygenn.genn_wrapper.DerivedParamFunc
    row_build_code          --  string with row building initialization code
    col_build_code          --  string with column building initialization code

    calc_max_row_len_func   --  instance of class inheriting from
                                CalcMaxLengthFunc used to calculate maximum
                                row length of synaptic matrix
    calc_max_col_len_func   --  instance of class inheriting from
                                CalcMaxLengthFunc used to calculate maximum
                                col length of synaptic matrix
    calc_kernel_size_func   --  instance of class inheriting from CalcKernelSizeFunc
                                used to calculate kernel dimensions
    extra_global_params     --  list of pairs of strings with names and
                                types of additional parameters
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

def create_toeplitz_connect_init_snippet(class_name, params=None,
                                         param_names=None,
                                         derived_params=None,
                                         diagonal_build_code=None,
                                         calc_max_row_len_func=None,
                                         calc_kernel_size_func=None,
                                         extra_global_params=None):
    """This helper function creates a custom
    InitToeplitzConnectivitySnippet class.
    See also:
    create_neuron_model
    create_weight_update_model
    create_postsynaptic_model
    create_current_source_model
    create_var_init_snippet

    Args:
    class_name              --  name of the new class

    Keyword args:
    param_names             --  list of strings with param names of the model
    derived_params          --  list of pairs, where the first member is string
                                with name of the derived parameter and the
                                second MUST be an instance of the class which
                                inherits from pygenn.genn_wrapper.DerivedParamFunc
    diagonal_build_code     --  string with diagonal building code
    calc_max_row_len_func   --  instance of class inheriting from
                                CalcMaxLengthFunc used to calculate maximum
                                row length of synaptic matrix
    calc_kernel_size_func   --  instance of class inheriting from CalcKernelSizeFunc
                                used to calculate kernel dimensions
    extra_global_params     --  list of pairs of strings with names and
                                types of additional parameters
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

@deprecated("This wrapper is now unnecessary - use callables directly",
            category=FutureWarning)
def create_dpf_class(dp_func):
    """Helper function to create derived parameter function class

    Args:
    dp_func --  a function which computes the derived parameter and takes
                two args "pars" (vector of double) and "dt" (double)
    """
    return lambda: dp_func

@deprecated("This wrapper is now unnecessary - use callables directly",
            category=FutureWarning)
def create_cmlf_class(cml_func):
    """Helper function to create function class for calculating sizes of
    matrices initialised with sparse connectivity initialisation snippet

    Args:
    cml_func -- a function which computes the length and takes
                three args "num_pre" (unsigned int), "num_post" (unsigned int)
                and "pars" (vector of double)
    """
    return lambda: cml_func

@deprecated("This wrapper is now unnecessary - use callables directly",
            category=FutureWarning)
def create_cksf_class(cks_func):
    """Helper function to create function class for calculating sizes 
    of kernels from connectivity initialiser parameters 

    Args:
    cks_func -- a function which computes the kernel size and takes
                one arg "pars" (vector of double)
    """
    return lambda: cks_func

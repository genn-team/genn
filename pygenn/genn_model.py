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
import sys
from textwrap import dedent
from warnings import warn
from weakref import proxy

# 3rd party imports
import numpy as np
from six import iteritems, itervalues, string_types

# pygenn imports
from .genn import (generate_code, init_logging, CurrentSource,
                   CurrentSourceModelBase, CustomConnectivityUpdate, 
                   CustomConnectivityUpdateModelBase, CustomUpdate, 
                   CustomUpdateModelBase, CustomUpdateWU, DerivedParam, 
                   EGP, EGPRef, InitSparseConnectivitySnippetBase,
                   InitToeplitzConnectivitySnippetBase, InitVarSnippetBase,
                   ModelSpecInternal, NeuronGroup, NeuronModelBase,
                   ParamVal, PlogSeverity, PostsynapticModelBase,
                   SparseConnectivityInit, SynapseGroup, SynapseMatrixType,
                   ToeplitzConnectivityInit, UnresolvedType, Var, VarInit, 
                   VarLocation, VarRef, WeightUpdateModelBase)
from .shared_library_model import (SharedLibraryModelDouble, 
                                   SharedLibraryModelFloat)
                                   
from .genn_groups import (CurrentSourceMixin, CustomConnectivityUpdateMixin,
                          CustomUpdateMixin, CustomUpdateWUMixin,
                          NeuronGroupMixin, SynapseGroupMixin)
from .model_preprocessor import get_snippet, get_var_init
from . import (current_source_models, custom_connectivity_update_models,
               custom_update_models, init_sparse_connectivity_snippets, 
               init_toeplitz_connectivity_snippets, init_var_snippets,
               neuron_models, postsynaptic_models, types, weight_update_models)

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
    _msvc_env = {k.upper(): v for k, v in iteritems(_msvc_env)}
    
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


class GeNNModel(ModelSpecInternal):
    """GeNNModel class
    This class helps to define, build and run a GeNN model from python
    """

    def __init__(self, precision="float", model_name="GeNNModel",
                 backend=None, time_precision=None,
                 genn_log_level=PlogSeverity.WARNING,
                 code_gen_log_level=PlogSeverity.WARNING,
                 transpiler_log_level=PlogSeverity.WARNING,
                 backend_log_level=PlogSeverity.WARNING,
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
        if self.time_precision == types.Float:
            self._slm = SharedLibraryModelFloat()
        elif self.time_precision == types.Double:
            self._slm = SharedLibraryModelDouble()
        else:
            raise ValueError(
                "Supported time precisions are float and double, "
                "but '{1}' was given".format(self._time_precision))

        # Initialise GeNN logging
        init_logging(genn_log_level, code_gen_log_level, transpiler_log_level)
        
        self._built = False
        self._loaded = False
        self.backend_name = backend
        self._preferences = preference_kwargs
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
            types.Uint32:   np.int64,
            types.Int32:    np.uint64,
            types.Int32:    np.int32,
            types.Uint32:   np.uint32,
            types.Int16:    np.int16,
            types.Uint16:   np.uint16,
            types.Int8:     np.int8,
            types.Uint8:    np.uint8,
            types.Bool:     np.bool8}

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
    @deprecated("The name of this property was inconsistent, use dt instead")
    def dT(self):
        return self.dt
    
    @dT.setter
    @deprecated("The name of this property was inconsistent, use dt instead")
    def dT(self, dt):
        self.dt = dt
    
    # **TODO** is there a better way of exposing inner class properties?
    @property
    def t(self):
        """Simulation time in ms"""
        return self._slm.time

    @t.setter
    def t(self, t):
        self._slm.time = t

    @property
    def timestep(self):
        """Simulation time step"""
        return self._slm.timestep

    @timestep.setter
    def timestep(self, timestep):
        self._slm.timestep = timestep

    @property
    def free_device_mem_bytes(self):
        return self._slm.free_device_mem_bytes;

    @property
    def neuron_update_time(self):
        return self._slm.neuron_update_time

    @property
    def init_time(self):
        return self._slm.init_time

    @property
    def presynaptic_update_time(self):
        return self._slm.presynaptic_update_time

    @property
    def postsynaptic_update_time(self):
        return self._slm.postsynaptic_update_time

    @property
    def synapse_dynamics_time(self):
        return self._slm.synapse_dynamics_time

    @property
    def init_sparse_time(self):
        return self._slm.init_sparse_time

    def get_custom_update_time(self, name):
        return self._slm.get_custom_update_time(name)

    def get_custom_update_transpose_time(self, name):
        return self._slm.get_custom_update_transpose_time(name)

    def add_neuron_population(self, pop_name, num_neurons, neuron,
                              param_space, var_space):
        """Add a neuron population to the GeNN model

        Args:
        pop_name    --  name of the new population
        num_neurons --  number of neurons in the new population
        neuron      --  type of the NeuronModels class as string or instance of
                        neuron class derived from
                        ``pygenn.genn_wrapper.NeuronModels.Custom`` (see also
                        pygenn.genn_model.create_custom_neuron_class)
        param_space --  dict with param values for the NeuronModels class
        var_space   --  dict with initial variable values for the
                        NeuronModels class
        """
        if self._built:
            raise Exception("GeNN model already built")

        # Resolve neuron model
        neuron = get_snippet(neuron, NeuronModelBase, neuron_models)
        
        # Extract parts of var_space which should be initialised by GeNN
        var_init = get_var_init(var_space)
        
        # Use superclass to add population
        n_group = super(GeNNModel, self).add_neuron_population(
            pop_name, int(num_neurons), neuron, param_space, var_init)
        
        # Initialise group, store group in dictionary and return
        n_group._init_group(self, var_space)
        self.neuron_populations[pop_name] = n_group
        return n_group

    def add_synapse_population(self, pop_name, matrix_type, delay_steps,
                               source, target, w_update_model, wu_param_space,
                               wu_var_space, wu_pre_var_space,
                               wu_post_var_space, postsyn_model,
                               ps_param_space, ps_var_space,
                               connectivity_initialiser=None):
        """Add a synapse population to the GeNN model

        Args:
        pop_name                    --  name of the new population
        matrix_type                 --  SynapseMatrixType describing type of the matrix
        delay_steps                 --  delay in number of steps
        source                      --  source neuron group (either name or NeuronGroup object)
        target                      --  target neuron group (either name or NeuronGroup object)
        w_update_model              --  type of the WeightUpdateModels class
                                        as string or instance of weight update
                                        model class derived from
                                        ``WeightUpdateModelBase`` (see also
                                        pygenn.create_custom_weight_update_class)
        wu_param_space              --  dict with param values for the
                                        WeightUpdateModels class
        wu_var_space                --  dict with initial values for
                                        WeightUpdateModels state variables
        wu_pre_var_space            --  dict with initial values for
                                        WeightUpdateModels presynaptic variables
        wu_post_var_space           --  dict with initial values for
                                        WeightUpdateModels postsynaptic variables
        postsyn_model               --  type of the PostsynapticModels class
                                        as string or instance of postsynaptic
                                        model class derived from
                                        ``PostsynapticModelBase`` (see also
                                        pygenn.create_custom_postsynaptic_class)
        ps_param_space              --  dict with param values for the
                                        PostsynapticModels class
        ps_var_space                --  dict with initial variable values for
                                        the PostsynapticModels class
        connectivity_initialiser    --  SparseConnectivityInit or 
                                        ToeplitzConnectivityInit used to 
                                        configure connectivity
        """
        if self._built:
            raise Exception("GeNN model already built")

        # Validate source and target groups
        # **TODO** remove once underlying 
        source = self._validate_neuron_group(source, "source")
        target = self._validate_neuron_group(target, "target")
        
        # If matrix type is a string, loop up enumeration value
        if isinstance(matrix_type, string_types):
            matrix_type = getattr(SynapseMatrixType, matrix_type)
        
        # If no connectivity initialiser is passed, 
        # use unitialised sparse connectivity
        if connectivity_initialiser is None:
            connectivity_initialiser = init_sparse_connectivity(
                init_sparse_connectivity_snippets.Uninitialised(), {})
        
        # Resolve postsynaptic and weight update models
        postsyn_model = get_snippet(postsyn_model, PostsynapticModelBase, 
                                    postsynaptic_models)
        w_update_model = get_snippet(w_update_model, WeightUpdateModelBase, 
                                     weight_update_models)
        
        # Extract parts of var spaces which should be initialised by GeNN
        ps_var_init = get_var_init(ps_var_space)
        wu_var_init = get_var_init(wu_var_space)
        wu_pre_var_init = get_var_init(wu_pre_var_space)
        wu_post_var_init = get_var_init(wu_post_var_space)
        
        # Use superclass to add population
        s_group = super(GeNNModel, self).add_synapse_population(
            pop_name, matrix_type, delay_steps, source.name, target.name, 
            w_update_model, wu_param_space, wu_var_init, wu_pre_var_init, wu_post_var_init,
            postsyn_model, ps_param_space, ps_var_init,
            connectivity_initialiser)
        
        # Initialise group, store group in dictionary and return
        s_group._init_group(self, ps_var_space, wu_var_space, wu_pre_var_space,
                            wu_post_var_space, source, target)
        self.synapse_populations[pop_name] = s_group
        return s_group

    def add_current_source(self, cs_name, current_source_model, pop,
                           param_space, var_space):
        """Add a current source to the GeNN model

        Args:
        cs_name                 --  name of the new current source
        current_source_model    --  type of the CurrentSourceModels class as
                                    string or instance of CurrentSourceModels
                                    class derived from
                                    ``pygenn.genn_wrapper.CurrentSourceModels.Custom`` (see also
                                    pygenn.genn_model.create_custom_current_source_class)
        pop                     --  population into which the current source 
                                    should be injected (either name or NeuronGroup object)
        param_space             --  dict with param values for the
                                    CurrentSourceModels class
        var_space               --  dict with initial variable values for the
                                    CurrentSourceModels class
        """
        if self._built:
            raise Exception("GeNN model already built")

        # Validate population
        # **TODO** remove once underlying 
        pop = self._validate_neuron_group(pop, "pop")

        # Resolve current source model
        current_source_model = get_snippet(current_source_model, CurrentSourceModelBase,
                                           current_source_models)
        
        # Extract parts of var_space which should be initialised by GeNN
        var_init = get_var_init(var_space)
        
        # Use superclass to add population
        c_source = super(GeNNModel, self).add_current_source(
            cs_name, current_source_model, pop.name, param_space, var_init)
        
        # Initialise group, store group in dictionary and return
        c_source._init_group(self, var_space, pop)
        self.current_sources[cs_name] = c_source
        return c_source
    
    def add_custom_update(self, cu_name, group_name, custom_update_model,
                          param_space, var_space, var_ref_space, egp_ref_space={}):
        """Add a current source to the GeNN model

        Args:
        cu_name                 -- name of the new current source
        group_name              -- name of custom update group this
                                   update belongs to
        custom_update_model     -- type of the CustomUpdateModel class as
                                   string or instance of CustomUpdateModel
                                   class derived from
                                   ``CustomUpdateModelBase`` (see also
                                   pygenn.genn_model.create_custom_custom_update_class)
        param_space             -- dict with param values for the
                                   CustomUpdateModel class
        var_space               -- dict with initial variable values for the
                                   CustomUpdateModel class
        var_ref_space           -- dict with variable references for the
                                   CustomUpdateModel class
        egp_ref_space           -- dict with extra global parameter references 
                                   for the CustomUpdateModel class
        """
        if self._built:
            raise Exception("GeNN model already built")
        
        # Resolve custom update model
        custom_update_model = get_snippet(custom_update_model, CustomUpdateModelBase,
                                          custom_update_models)
        
        # Extract parts of var_space which should be initialised by GeNN
        var_init = get_var_init(var_space)

        # Use superclass to add population
        c_update = super(GeNNModel, self).add_custom_update(
            cu_name, group_name, custom_update_model,
            param_space, var_init, var_ref_space, egp_ref_space)

        # Setup back-reference, store group in dictionary and return
        c_update._init_group(self, var_space)
        self.custom_updates[cu_name] = c_update
        return c_update
    
    def add_custom_connectivity_update(self, cu_name, group_name, syn_group,
                                       custom_conn_update_model,
                                       param_space, var_space, pre_var_space,
                                       post_var_space, var_ref_space,
                                       pre_var_ref_space, post_var_ref_space):
        """Add a custom connectivity update to the GeNN model

        Args:
        cu_name                     -- name of the new custom connectivity update
        group_name                  -- name of group this custom connectivity update
                                       should be performed within
        syn_group                   -- synapse group this custom connectivity
                                       update should be attached to (either
                                       name or SynapseGroup object)
        custom_conn_update_model    -- type of the CustomConnetivityUpdateModel 
                                       class as string or instance of 
                                       CustomConnectivityUpdateModel class 
                                       derived from 
                                       ``pygenn.genn_wrapper.CustomConnectivityUpdateModel.Custom``
                                       (see also pygenn.genn_model.create_custom_custom_connectivity_update_class)
        param_space                 -- dict with param values for the
                                       CustomConnectivityUpdateModel class
        var_space                   -- dict with initial variable values for the
                                       CustomConnectivityUpdateModel class
        pre_var_space               -- dict with initial presynaptic variable
                                       values for the 
                                       CustomConnectivityUpdateModel class
        post_var_space              -- dict with initial postsynaptic variable
                                       values for the
                                       CustomConnectivityUpdateModel class
        var_ref_space               -- dict with variable references for the
                                       CustomConnectivityUpdateModel class
        pre_var_ref_space           -- dict with presynaptic variable
                                       references for the 
                                       CustomConnectivityUpdateModel class
        post_var_ref_space          -- dict with postsynaptic variable
                                       references for the
                                       CustomConnectivityUpdateModel class
        """
        if self._built:
            raise Exception("GeNN model already built")
        
        # Resolve custom update model
        custom_connectivity_update_model = get_snippet(
            custom_conn_update_model, CustomConnectivityUpdateModelBase,
            custom_connectivity_update_models)
        
        # Extract parts of var_space which should be initialised by GeNN
        var_init = get_var_init(var_space)
        pre_var_init = get_var_init(pre_var_space)
        post_var_init = get_var_init(post_var_space)

        # Use superclass to add population
        syn_group = self._validate_synapse_group(syn_group, "syn_group")
        c_update = super(GeNNModel, self).add_custom_connectivity_update(
            cu_name, group_name, syn_group.name, custom_connectivity_update_model,
            param_space, var_init, pre_var_init, post_var_init,
            var_ref_space, pre_var_ref_space, post_var_ref_space)

        # Setup back-reference, store group in dictionary and return
        c_update._init_group(self, var_space, pre_var_space, 
                             post_var_space, syn_group)
        self.custom_connectivity_updates[cu_name] = c_update
        return c_update
        
    def build(self, path_to_model="./", force_rebuild=False):
        """Finalize and build a GeNN model

        Keyword args:
        path_to_model   --  path where to place the generated model code.
                            Defaults to the local directory.
        force_rebuild   --  should model be rebuilt even if 
                            it doesn't appear to be required
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
        preferences = self._backend_module.Preferences()

        # Set attributes on preferences object
        for k, v in iteritems(self._preferences):
            if hasattr(preferences, k):
                setattr(preferences, k, v)
        
        # Create backend
        backend = self._backend_module.create_backend(self, output_path,
                                                      self.backend_log_level,
                                                      preferences)

        # Generate code
        mem_alloc = generate_code(self, backend, share_path,
                                  output_path, force_rebuild)

        # Build code
        if system() == "Windows":
            check_call([_msbuild, "/p:Configuration=Release", "/m", "/verbosity:minimal",
                        path.join(output_path, "runner.vcxproj")])
        else:
            check_call(["make", "-j", str(cpu_count(logical=False)), "-C", output_path])

        self._built = True
        return mem_alloc

    def load(self, path_to_model="./", num_recording_timesteps=None):
        """import the model as shared library and initialize it"""
        if self._loaded:
            raise Exception("GeNN model already loaded")
        self._path_to_model = path_to_model

        self._slm.open(self._path_to_model, self.name)

        self._slm.allocate_mem()

        # If model uses recording system
        if self.recording_in_use:
            # Raise exception if recording timesteps is not set
            if num_recording_timesteps is None:
                raise Exception("Cannot use recording system without passing "
                                "number of recording timesteps to GeNNModel.load")

            # Allocate recording buffers
            self._slm.allocate_recording_buffers(num_recording_timesteps)

        # Loop through neuron populations and load any
        # extra global parameters required for initialization
        for pop_data in itervalues(self.neuron_populations):
            pop_data.load_init_egps()

        # Loop through synapse populations and load any 
        # extra global parameters required for initialization
        for pop_data in itervalues(self.synapse_populations):
            pop_data.load_init_egps()

        # Loop through current sources
        for src_data in itervalues(self.current_sources):
            src_data.load_init_egps()

        # Loop through custom connectivity updates
        for cu_data in itervalues(self.custom_connectivity_updates):
            cu_data.load_init_egps()

        # Loop through custom updates
        for cu_data in itervalues(self.custom_updates):
            cu_data.load_init_egps()

        # Initialize model
        self._slm.initialize()

        # Loop through neuron populations
        for pop_data in itervalues(self.neuron_populations):
            pop_data.load(num_recording_timesteps)

        # Loop through synapse populations
        for pop_data in itervalues(self.synapse_populations):
            pop_data.load()

        # Loop through current sources
        for src_data in itervalues(self.current_sources):
            src_data.load()

        # Loop through custom connectivity updates
        for cu_data in itervalues(self.custom_connectivity_updates):
            cu_data.load()

        # Loop through custom updates
        for cu_data in itervalues(self.custom_updates):
            cu_data.load()

        # Now everything is set up call the sparse initialisation function
        self._slm.initialize_sparse()

        # Set loaded flag and built flag
        self._loaded = True
        self._built = True


    def unload(self):
        # Loop through custom updates and unload
        for cu_data in itervalues(self.custom_updates):
            cu_data.unload()
        
        # Loop through custom connectivity updates and unload
        for cu_data in itervalues(self.custom_connectivity_updates):
            cu_data.unload()
    
        # Loop through current sources and unload
        for src_data in itervalues(self.current_sources):
            src_data.unload()

        # Loop through synapse populations and unload
        for pop_data in itervalues(self.synapse_populations):
            pop_data.unload()

        # Loop through neuron populations and unload
        for pop_data in itervalues(self.neuron_populations):
            pop_data.unload()

        # Close shared library model
        self._slm.close()

        # Clear loaded flag
        self._loaded = False

    def step_time(self):
        """Make one simulation step"""
        if not self._loaded:
            raise Exception("GeNN model has to be loaded before stepping")

        self._slm.step_time()
    
    def custom_update(self, name):
        """Perform custom update"""
        if not self._loaded:
            raise Exception("GeNN model has to be loaded before performing custom update")
            
        self._slm.custom_update(name)
   

    def pull_recording_buffers_from_device(self):
        """Pull recording buffers from device"""
        if not self._loaded:
            raise Exception("GeNN model has to be loaded before pulling recording buffers")

        if not self.recording_in_use:
            raise Exception("Cannot pull recording buffer if recording system is not in use")

        # Pull recording buffers from device
        self._slm.pull_recording_buffers_from_device()

    def end(self):
        """Free memory"""
        for group in [self.neuron_populations, self.synapse_populations,
                      self.current_sources, self.custom_connectivity_updates, 
                      self.custom_updates]:
            for g_name, g_dat in iteritems(group):
                for egp_name, egp_dat in iteritems(g_dat.extra_global_params):
                    # if auto allocation is not enabled, let the user care
                    # about freeing of the EGP
                    if egp_dat.needsAllocation:
                        self._slm.free_extra_global_param(g_name, egp_name)
        # "normal" variables are freed when SharedLibraryModel is destoyed

    def _validate_neuron_group(self, group, context):
        # If group is a string
        if isinstance(group, string_types):
            # If it's the name of a neuron group, return it
            if group in self.neuron_populations:
                return self.neuron_populations[group]
            # Otherwise, raise error
            else:
                raise ValueError("'%s' neuron group '%s' not found" % 
                                 (context, group))
        # Otherwise, if group is a neuron group, return it
        elif isinstance(group, NeuronGroup):
            return group
        # Otherwise, raise error
        else:
            raise ValueError("'%s' must be a NeuronGroup or string" % context)

    def _validate_synapse_group(self, group, context):
        # If group is a string
        if isinstance(group, string_types):
            # If it's the name of a neuron group, return it
            if group in self.synapse_populations:
                return self.synapse_populations[group]
            # Otherwise, raise error
            else:
                raise ValueError("'%s' synapse group '%s' not found" % 
                                 (context, group))
        # Otherwise, if group is a synapse group, return it
        elif isinstance(group, SynapseGroup):
            return group
        # Otherwise, raise error
        else:
            raise ValueError("'%s' must be a SynapseGroup or string" % context)

def init_var(init_var_snippet, param_space={}):
    """This helper function creates a VarInit object
    to easily initialise a variable using a snippet.

    Args:
    init_var_snippet    --  type of the InitVarSnippet class as string or
                            instance of class derived from
                            InitVarSnippetBase class.
    param_space         --  dict with param values for the InitVarSnippet class
    """
    # Get snippet and wrap in VarInit object
    init_var_snippet = get_snippet(init_var_snippet,
                                   InitVarSnippetBase,
                                   init_var_snippets)

    # Use add function to create suitable VarInit
    return VarInit(init_var_snippet, param_space)


def init_sparse_connectivity(init_sparse_connect_snippet, param_space={}):
    """This helper function creates a InitSparseConnectivitySnippet::Init
    object to easily initialise connectivity using a snippet.

    Args:
    init_sparse_connect_snippet --  type of the InitSparseConnectivitySnippet
                                    class as string or instance of class
                                    derived from
                                    InitSparseConnectivitySnippetBase
    param_space                 --  dict with param values for the
                                    InitSparseConnectivitySnippet class
    """
    # Get snippet and wrap in SparseConnectivityInit object
    init_sparse_connect_snippet = get_snippet(init_sparse_connect_snippet,
                                              InitSparseConnectivitySnippetBase,
                                              init_sparse_connectivity_snippets)
    return SparseConnectivityInit(init_sparse_connect_snippet, param_space)

@deprecated("The name of this function was ambiguous, use init_sparse_connectivity instead")
def init_connectivity(init_sparse_connect_snippet, param_space={}):
    """This helper function creates a InitSparseConnectivitySnippet::Init
    object to easily initialise connectivity using a snippet.

    Args:
    init_sparse_connect_snippet --  type of the InitSparseConnectivitySnippet
                                    class as string or instance of class
                                    derived from
                                    InitSparseConnectivitySnippetBase
    param_space                 --  dict with param values for the
                                    InitSparseConnectivitySnippet class
    """
    return init_sparse_connectivity(init_sparse_connect_snippet, param_space)

def init_toeplitz_connectivity(init_toeplitz_connect_snippet, param_space={}):
    """This helper function creates a InitToeplitzConnectivitySnippet::Init
    object to easily initialise connectivity using a snippet.

    Args:
    init_toeplitz_connect_snippet   -- type of the InitToeplitzConnectivitySnippet
                                       class as string or instance of class
                                       derived from
                                       InitSparseConnectivitySnippetBase
    param_space                     -- dict with param values for the
                                       InitToeplitzConnectivitySnippet class
    """
    # Get snippet and wrap in InitToeplitzConnectivitySnippet object
    init_toeplitz_connect_snippet = get_snippet(init_toeplitz_connect_snippet,
                                                InitToeplitzConnectivitySnippetBase,
                                                init_toeplitz_connectivity_snippets)
    return ToeplitzConnectivityInit(init_toeplitz_connect_snippet, param_space)

def create_neuron_model(class_name, param_names=None,
                        var_name_types=None, derived_params=None,
                        sim_code=None, threshold_condition_code=None,
                        reset_code=None, support_code=None,
                        extra_global_params=None,
                        additional_input_vars=None,
                        is_auto_refractory_required=None):
    """This helper function creates a custom NeuronModel class.
    See also:
    create_postsynaptic_model
    create_weight_update_model
    create_current_source_model
    create_var_init_snippet
    create_sparse_connect_init_snippet

    Args:
    class_name                  --  name of the new class

    Keyword args:
    param_names                 --  list of strings with param names
                                    of the model
    var_name_types              --  list of pairs of strings with varible names
                                    and types of the model
    derived_params              --  list of pairs, where the first member
                                    is string with name of the derived
                                    parameter and the second should be a 
                                    functor returned by create_dpf_class
    sim_code                    --  string with the simulation code
    threshold_condition_code    --  string with the threshold condition code
    reset_code                  --  string with the reset code
    support_code                --  string with the support code
    extra_global_params         --  list of pairs of strings with names and
                                    types of additional parameters
    additional_input_vars       --  list of tuples with names and types as
                                    strings and initial values of additional
                                    local input variables
    is_auto_refractory_required --  does this model require auto-refractory
                                    logic to be generated?
    """
    body = {}

    if sim_code is not None:
        body["get_sim_code"] = lambda self: dedent(sim_code)

    if threshold_condition_code is not None:
        body["get_threshold_condition_code"] = \
            lambda self: dedent(threshold_condition_code)

    if reset_code is not None:
        body["get_reset_code"] = lambda self: dedent(reset_code)

    if support_code is not None:
        body["get_support_code"] = lambda self: dedent(support_code)

    if additional_input_vars:
        body["get_additional_input_vars"] = \
            lambda self: [ParamVal(a[0], a[1], a[2])
                                   for a in additional_input_vars]

    if is_auto_refractory_required is not None:
        body["is_auto_refractory_required"] = \
            lambda self: is_auto_refractory_required

    return create_model(class_name, NeuronModelBase, param_names, 
                        var_name_types, derived_params, 
                        extra_global_params, body)


def create_postsynaptic_model(class_name, param_names=None,
                              var_name_types=None, derived_params=None,
                              decay_code=None, apply_input_code=None,
                              support_code=None, extra_global_params=None):
    """This helper function creates a custom PostsynapticModel class.
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
    var_name_types      --  list of pairs of strings with varible names and
                            types of the model
    derived_params      --  list of pairs, where the first member is string
                            with name of the derived parameter and the second
                            should be a functor returned by create_dpf_class
    decay_code          --  string with the decay code
    apply_input_code    --  string with the apply input code
    support_code        --  string with the support code
    extra_global_params --  list of pairs of strings with names and
                            types of additional parameters
    """
    body = {}

    if decay_code is not None:
        body["get_decay_code"] = lambda self: dedent(decay_code)

    if apply_input_code is not None:
        body["get_apply_input_code"] = lambda self: dedent(apply_input_code)

    if support_code is not None:
        body["get_support_code"] = lambda self: dedent(support_code)

    return create_model(class_name, PostsynapticModelBase, param_names,
                        var_name_types, derived_params, 
                        extra_global_params, body)


def create_weight_update_model(class_name, param_names=None,
                               var_name_types=None,
                               pre_var_name_types=None,
                               post_var_name_types=None,
                               derived_params=None, sim_code=None,
                               event_code=None, learn_post_code=None,
                               synapse_dynamics_code=None,
                               event_threshold_condition_code=None,
                               pre_spike_code=None,
                               post_spike_code=None,
                               pre_dynamics_code=None,
                               post_dynamics_code=None,
                               sim_support_code=None,
                               learn_post_support_code=None,
                               synapse_dynamics_suppport_code=None,
                               extra_global_params=None,
                               is_pre_spike_time_required=None,
                               is_post_spike_time_required=None,
                               is_pre_spike_event_time_required=None,
                               is_prev_pre_spike_time_required=None,
                               is_prev_post_spike_time_required=None,
                               is_prev_pre_spike_event_time_required=None):
    """This helper function creates a custom WeightUpdateModel class.
    See also:
    create_neuron_model
    create_postsynaptic_model
    create_current_source_model
    create_var_init_snippet
    create_sparse_connect_init_snippet

    Args:
    class_name                              --  name of the new class

    Keyword args:
    param_names                             --  list of strings with param names of
                                                the model
    var_name_types                          --  list of pairs of strings with variable
                                                names and types of the model
    pre_var_name_types                      --  list of pairs of strings with
                                                presynaptic variable names and
                                                types of the model
    post_var_name_types                     --  list of pairs of strings with
                                                postsynaptic variable names and
                                                types of the model
    derived_params                          --  list of pairs, where the first member
                                                is string with name of the derived
                                                parameter and the second should be 
                                                a functor returned by create_dpf_class
    sim_code                                --  string with the simulation code
    event_code                              --  string with the event code
    learn_post_code                         --  string with the code to include in
                                                learn_synapse_post kernel/function
    synapse_dynamics_code                   --  string with the synapse dynamics code
    event_threshold_condition_code          --  string with the event threshold
                                                condition code
    pre_spike_code                          --  string with the code run once per
                                                spiking presynaptic neuron
    post_spike_code                         --  string with the code run once per
                                                spiking postsynaptic neuron
    pre_dynamics_code                       --  string with the code run every
                                                timestep on presynaptic neuron
    post_dynamics_code                      --  string with the code run every
                                                timestep on postsynaptic neuron
    sim_support_code                        --  string with simulation support code
    learn_post_support_code                 --  string with support code for
                                                learn_synapse_post kernel/function
    synapse_dynamics_suppport_code          --  string with synapse dynamics
                                                support code
    extra_global_params                     --  list of pairs of strings with names and
                                                types of additional parameters
    is_pre_spike_time_required              --  boolean, is presynaptic spike time
                                                required in any weight update kernels?
    is_post_spike_time_required             --  boolean, is postsynaptic spike time
                                                required in any weight update kernels?
    is_pre_spike_event_time_required        --  boolean, is presynaptic spike-like-event
                                                time required in any weight update kernels?
    is_prev_pre_spike_time_required         --  boolean, is previous presynaptic spike time
                                                required in any weight update kernels?
    is_prev_post_spike_time_required        --  boolean, is previous postsynaptic spike time
                                                required in any weight update kernels?
    is_prev_pre_spike_event_time_required   --  boolean, is _previous_ presynaptic spike-like-event 
                                                time required in any weight update kernels?
    """
    body = {}

    if sim_code is not None:
        body["get_sim_code"] = lambda self: dedent(sim_code)

    if event_code is not None:
        body["get_event_code"] = lambda self: dedent(event_code)

    if learn_post_code is not None:
        body["get_learn_post_code"] = lambda self: dedent(learn_post_code)

    if synapse_dynamics_code is not None:
        body["get_synapse_dynamics_code"] = lambda self: dedent(synapse_dynamics_code)

    if event_threshold_condition_code is not None:
        body["get_event_threshold_condition_code"] = \
            lambda self: dedent(event_threshold_condition_code)

    if pre_spike_code is not None:
        body["get_pre_spike_code"] = lambda self: dedent(pre_spike_code)

    if post_spike_code is not None:
        body["get_post_spike_code"] = lambda self: dedent(post_spike_code)

    if pre_dynamics_code is not None:
        body["get_pre_dynamics_code"] = lambda self: dedent(pre_dynamics_code)

    if post_dynamics_code is not None:
        body["get_post_dynamics_code"] = lambda self: dedent(post_dynamics_code)

    if sim_support_code is not None:
        body["get_sim_support_code"] = lambda self: dedent(sim_support_code)

    if learn_post_support_code is not None:
        body["get_learn_post_support_code"] = \
            lambda self: dedent(learn_post_support_code)

    if synapse_dynamics_suppport_code is not None:
        body["get_synapse_dynamics_suppport_code"] = \
            lambda self: dedent(synapse_dynamics_suppport_code)

    if pre_var_name_types is not None:
        body["get_pre_vars"] = \
            lambda self: [Var(*vn) for vn in pre_var_name_types]

    if post_var_name_types is not None:
        body["get_post_vars"] = \
            lambda self: [Var(*vn) for vn in post_var_name_types]

    if is_pre_spike_time_required is not None:
        body["is_pre_spike_time_required"] = \
            lambda self: is_pre_spike_time_required

    if is_post_spike_time_required is not None:
        body["is_post_spike_time_required"] = \
            lambda self: is_post_spike_time_required
    
    if is_pre_spike_event_time_required is not None:
        body["is_pre_spike_event_time_required"] = \
            lambda self: is_pre_spike_event_time_required

    if is_prev_pre_spike_time_required is not None:
        body["is_prev_pre_spike_time_required"] = \
            lambda self: is_prev_pre_spike_time_required

    if is_prev_post_spike_time_required is not None:
        body["is_prev_post_spike_time_required"] = \
            lambda self: is_prev_post_spike_time_required

    if is_prev_pre_spike_event_time_required is not None:
        body["is_prev_pre_spike_event_time_required"] = \
            lambda self: is_prev_pre_spike_event_time_required

    return create_model(class_name, WeightUpdateModelBase, param_names,
                        var_name_types, derived_params, 
                        extra_global_params, body)


def create_current_source_model(class_name, param_names=None,
                                var_name_types=None,
                                derived_params=None,
                                injection_code=None,
                                extra_global_params=None):
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
    var_name_types      --  list of pairs of strings with varible names and
                            types of the model
    derived_params      --  list of pairs, where the first member is string
                            with name of the derived parameter and the second
                            should be a functor returned by create_dpf_class
    injection_code      --  string with the current injection code
    extra_global_params --  list of pairs of strings with names and types of
                            additional parameters
    """
    body = {}

    if injection_code is not None:
        body["get_injection_code"] = lambda self: dedent(injection_code)

    return create_model(class_name, CurrentSourceModelBase, param_names,
                        var_name_types, derived_params, 
                        extra_global_params, body)


def create_custom_update_model(class_name, param_names=None,
                               var_name_types=None,
                               derived_params=None,
                               var_refs=None,
                               update_code=None,
                               extra_global_params=None,
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

    if update_code is not None:
        body["get_update_code"] = lambda self: dedent(update_code)

    if var_refs is not None:
        body["get_var_refs"] = lambda self: [VarRef(*v) for v in var_refs]

    if extra_global_param_refs is not None:
        body["get_extra_global_param_refs"] =\
            lambda self: [EGPRef(*e) for e in extra_global_param_refs]

    return create_model(class_name, CustomUpdateModelBase, param_names,
                        var_name_types, derived_params,
                        extra_global_params, body)

def create_custom_connectivity_update_model(class_name, 
                                            param_names=None,
                                            var_name_types=None,
                                            pre_var_name_types=None,
                                            post_var_name_types=None,
                                            derived_params=None,
                                            var_refs=None,
                                            pre_var_refs=None,
                                            post_var_refs=None,
                                            row_update_code=None,
                                            host_update_code=None,
                                            extra_global_params=None):
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
    var_name_types      --  list of tuples of strings with variable names and
                            types of the variable
    pre_var_name_types  --  list of tuples of strings with variable names and
                            types of the variable
    var_name_types      --  list of tuples of strings with variable names and
                            types of the variable
    derived_params      --  list of tuples, where the first member is string
                            with name of the derived parameter and the second
                            should be a functor returned by create_dpf_class
    var_refs            --  list of tuples of strings with variable names and
                            types of variabled variable
    update_code         --  string with the current injection code
    extra_global_params --  list of pairs of strings with names and types of
                            additional parameters
    """
    body = {}

    if row_update_code is not None:
        body["get_row_update_code"] = lambda self: dedent(row_update_code)

    if host_update_code is not None:
        body["get_host_update_code"] = lambda self: dedent(host_update_code)

    if pre_var_name_types is not None:
        body["get_pre_vars"] = \
            lambda self: [Var(*vn) for vn in pre_var_name_types]

    if post_var_name_types is not None:
        body["get_post_vars"] = \
            lambda self: [Var(*vn) for vn in post_var_name_types]

    if var_refs is not None:
        body["get_var_refs"] = lambda self: [VarRef(*v) for v in var_refs]

    if pre_var_refs is not None:
        body["get_pre_var_refs"] = \
            lambda self: [VarRef(*v) for v in pre_var_refs]

    if post_var_refs is not None:
        body["get_post_var_refs"] = \
            lambda self: [VarRef(*v) for v in post_var_refs]

    return create_model(class_name, CustomConnectivityUpdateModelBase,
                        param_names, var_name_types, derived_params, 
                        extra_global_params, body)


def create_model(class_name, base, param_names, var_name_types,
                 derived_params, extra_global_params, custom_body):
    """This helper function completes a custom model class creation.

    This part is common for all model classes and is nearly useless on its own
    unless you specify custom_body.
    See also:
    create_neuron_model
    create_weight_update_model
    create_postsynaptic_model
    create_current_source_model
    create_var_init_snippet
    create_sparse_connect_init_snippet

    Args:
    class_name      --  name of the new class
    base            --  base class
    param_names     --  list of strings with param names of the model
    var_name_types  --  list of pairs of strings with varible names and
                        types of the model
    derived_params  --  list of pairs, where the first member is string with
                        name of the derived parameter and the second should 
                        be a functor returned by create_dpf_class
    extra_global_params --  list of pairs of strings with names and types of
                            additional parameters
    custom_body     --  dictionary with attributes and methods of the new class
    """

    def ctor(self):
        base.__init__(self)

    body = {
        "__init__": ctor,
    }

    if param_names is not None:
        body["get_param_names"] = lambda self: param_names

    if var_name_types is not None:
        body["get_vars"] = \
            lambda self: [Var(*vn) for vn in var_name_types]

    if derived_params is not None:
        body["get_derived_params"] = \
            lambda self: [DerivedParam(dp[0], dp[1]) 
                          for dp in derived_params]

    if extra_global_params is not None:
        body["get_extra_global_params"] = \
            lambda self: [EGP(egp[0], egp[1])
                          for egp in extra_global_params]

    if custom_body is not None:
        body.update(custom_body)

    return type(class_name, (base,), body)()


def create_var_init_snippet(class_name, param_names=None,
                            derived_params=None,
                            var_init_code=None, 
                            extra_global_params=None):
    """This helper function creates a custom InitVarSnippet class.
    See also:
    create_neuron_model
    create_weight_update_model
    create_postsynaptic_model
    create_current_source_model
    create_sparse_connect_init_snippet

    Args:
    class_name          --  name of the new class

    Keyword args:

    param_names         --  list of strings with param names of the model
    derived_params      --  list of pairs, where the first member is string with
                            name of the derived parameter and the second MUST be
                            an instance of the pygenn.genn_wrapper.DerivedParamFunc class
    var_init_code       --  string with the variable initialization code
    extra_global_params --  list of pairs of strings with names and
                            types of additional parameters
    """
    body = {}

    if var_init_code is not None:
        body["get_code"] = lambda self: dedent(var_init_code)

    return create_model(class_name, InitVarSnippetBase, 
                        param_names, None, derived_params, 
                        extra_global_params, body)


def create_sparse_connect_init_snippet(class_name,
                                       param_names=None,
                                       derived_params=None,
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
        body["get_row_build_code"] = lambda self: dedent(row_build_code)

    if col_build_code is not None:
        body["get_col_build_code"] = lambda self: dedent(col_build_code)

    if calc_max_row_len_func is not None:
        body["get_calc_max_row_length_func"] = \
            lambda self: make_cmlf(calc_max_row_len_func)

    if calc_max_col_len_func is not None:
        body["get_calc_max_col_length_func"] = \
            lambda self: make_cmlf(calc_max_col_len_func)

    if calc_kernel_size_func is not None:
        body["get_calc_kernel_size_func"] = \
            lambda self: make_cksf(calc_kernel_size_func)

    return create_model(class_name, InitSparseConnectivitySnippetBase, param_names,
                        None, derived_params, extra_global_params, body)

@deprecated("this wrapper is now unnecessary - use callables directly")
def create_dpf_class(dp_func):
    """Helper function to create derived parameter function class

    Args:
    dp_func --  a function which computes the derived parameter and takes
                two args "pars" (vector of double) and "dt" (double)
    """
    return lambda: dp_func

@deprecated("this wrapper is now unnecessary - use callables directly")
def create_cmlf_class(cml_func):
    """Helper function to create function class for calculating sizes of
    matrices initialised with sparse connectivity initialisation snippet

    Args:
    cml_func -- a function which computes the length and takes
                three args "num_pre" (unsigned int), "num_post" (unsigned int)
                and "pars" (vector of double)
    """
    return lambda: cml_func

@deprecated("this wrapper is now unnecessary - use callables directly")
def create_cksf_class(cks_func):
    """Helper function to create function class for calculating sizes 
    of kernels from connectivity initialiser parameters 

    Args:
    cks_func -- a function which computes the kernel size and takes
                one arg "pars" (vector of double)
    """
    return lambda: cks_func

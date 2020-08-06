"""GeNNModel

This module provides GeNNModel class to simplify working with pygenn module and
helper functions to derive custom model classes.

``GeNNModel`` can be (and should be) used to configure a model, build, load and
finally run it. Recording is done manually by pulling from the population of
interest and then copying the values from ``Variable.view`` attribute. Each
simulation step must be triggered manually by calling ``stepTime`` function.

Example:
    The following example shows in a (very) simplified manner how to build and
    run a simulation using GeNNModel::

        import GeNNModel
        gm = GeNNModel.GeNNModel()

        # add populations
        neuron_pop = gm.add_neuron_population(_parameters_truncated_)
        syn_pop = gm.add_synapse_population(_parameters_truncated_)

        # build and load model
        gm.build(path_to_model)
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
            gm.pull_state_from_device(neuron_pop_name)
            # finally, record voltage by copying form view into array.
            Vs[i,:] = v_view
"""
# python imports
from collections import OrderedDict
from importlib import import_module
from os import path
from platform import system
from psutil import cpu_count
from subprocess import check_call  # to call make
from textwrap import dedent

# 3rd party imports
import numpy as np
from six import iteritems, itervalues, string_types

# pygenn imports
from . import genn_wrapper
from .genn_wrapper import SharedLibraryModelNumpy as slm
from .genn_wrapper.Models import (Var, VarInit, VarVector)
from .genn_wrapper.InitSparseConnectivitySnippet import Init
from .genn_wrapper.Snippet import (make_dpf, EGP, ParamVal, DerivedParam,
                                   EGPVector, ParamValVector,
                                   DerivedParamVector)
from .genn_wrapper.InitSparseConnectivitySnippet import make_cmlf
from .genn_wrapper.StlContainers import StringVector
from .genn_wrapper import VarLocation_HOST_DEVICE
from .genn_groups import NeuronGroup, SynapseGroup, CurrentSource
from .model_preprocessor import prepare_snippet

# Loop through backends in preferential order
backend_modules = OrderedDict()
for b in ["CUDA", "SingleThreadedCPU"]:
    # Try and import
    try:
        m = import_module(".genn_wrapper." + b + "Backend", "pygenn")
    # Ignore failed imports - likely due to non-supported backends
    except ImportError as ex:
        pass
    # Raise any other errors
    except:
        raise
    # Otherwise add to (ordered) dictionary
    else:
        backend_modules[b] = m


class GeNNModel(object):
    """GeNNModel class
    This class helps to define, build and run a GeNN model from python
    """

    def __init__(self, precision=None, model_name="GeNNModel",
                 backend=None, genn_log_level=genn_wrapper.warning,
                 code_gen_log_level=genn_wrapper.warning,
                 backend_log_level=genn_wrapper.warning,
                 **preference_kwargs):
        """Init GeNNModel
        Keyword args:
        precision       --  string precision as string ("float", "double"
                            or "long double"). defaults to float.
        model_name      --  string name of the model. Defaults to "GeNNModel".
        backend         --  string specifying name of backend module to use
                            Defaults to None to pick 'best' backend for your system
        """
        precision = "float" if precision is not None else precision
        self._scalar = precision
        if precision == "float":
            genn_float_type = "GENN_FLOAT"
            self._slm = slm.SharedLibraryModelNumpy_f()
            self._np_type = np.float32
        elif precision == "double":
            genn_float_type = "GENN_DOUBLE"
            self._slm = slm.SharedLibraryModelNumpy_d()
            self._np_type = np.float64
        else:
            raise ValueError(
                "Supported precisions are float and double, "
                "but '{1}' was given".format(precision))

        # Initialise GeNN logging
        genn_wrapper.init_logging(genn_log_level, code_gen_log_level)

        self._built = False
        self._loaded = False
        self.use_backend = backend
        self._preferences = preference_kwargs
        self.backend_log_level=backend_log_level
        self._model = genn_wrapper.ModelSpecInternal()
        self._model.set_precision(getattr(genn_wrapper, genn_float_type))

        self.default_var_location = genn_wrapper.VarLocation_HOST_DEVICE
        self.model_name = model_name
        self.neuron_populations = {}
        self.synapse_populations = {}
        self.current_sources = {}
        self.dT = 0.1

        # For backward compatibility, if selected GPU is set, remove it from
        # preferences dictionary and add in underlying GeNN preferences
        selected_gpu = self._preferences.pop("selected_gpu", None)
        if selected_gpu is not None:
            self._preferences["deviceSelectMethod"] = self._backend_module.DeviceSelect_MANUAL
            self._preferences["preferences.manualDeviceID"] = selected_gpu

    @property
    def use_backend(self):
        return self._backend_name

    @use_backend.setter
    def use_backend(self, backend):
        # If no backend is specified
        if backend is None:
            # Check we have managed to import any bagenn_wrapperckends
            assert len(backend_modules) > 0

            # Set name to first (i.e. best) backend and lookup module from dictionary
            self._backend_name = next(iter(backend_modules))
            self._backend_module = backend_modules[self._backend_name]
        else:
            self._backend_name = backend
            self._backend_module = backend_modules[backend]

    @property
    def timing_enabled(self):
        return self._model.is_timing_enabled()

    @timing_enabled.setter
    def timing_enabled(self, timing):
        self._model.set_timing(timing)
    
    @property
    def default_var_location(self):
        """Default variable location - defines
        where state variables are initialised"""
        assert False
        # return self._model.get_default

    @default_var_location.setter
    def default_var_location(self, location):
        if self._built:
            raise Exception("GeNN model already built")

        self._model.set_default_var_location(location)

    @property
    def default_sparse_connectivity_location(location):
        """Default sparse connectivity mode - where
        connectivity is initialised"""
        assert False
        # return genn_wrapper.GeNNPreferences.cvar.defaultSparseConnectivityMode

    @default_sparse_connectivity_location.setter
    def default_sparse_connectivity_location(self, location):
        if self._built:
            raise Exception("GeNN model already built")

        self._model.set_default_sparse_connectivity_location(location)

    @property
    def model_name(self):
        """Name of the model"""
        return self._model.get_name()

    @model_name.setter
    def model_name(self, model_name):
        if self._built:
            raise Exception("GeNN model already built")
        self._model.set_name(model_name)

    @property
    def t(self):
        """Simulation time in ms"""
        return self._slm.get_time()

    @t.setter
    def t(self, t):
        self._slm.set_time(t)

    @property
    def timestep(self):
        """Simulation time step"""
        return self._slm.get_timestep()

    @timestep.setter
    def timestep(self, timestep):
        self._slm.set_timestep(timestep)

    @property
    def free_device_mem_bytes(self):
        return self._slm.get_free_device_mem_bytes();

    @property
    def dT(self):
        """Step size"""
        return self._model.get_dt()

    @dT.setter
    def dT(self, dt):
        if self._built:
            raise Exception("GeNN model already built")
        self._model.set_dt(dt)

    @property
    def neuron_update_time(self):
        return self._slm.get_neuron_update_time()
    
    @property
    def init_time(self):
        return self._slm.get_init_time()
        
    @property
    def presynaptic_update_time(self):
        return self._slm.get_presynaptic_update_time()
    
    @property
    def postsynaptic_update_time(self):
        return self._slm.get_postsynaptic_update_time()

    @property
    def synapse_dynamics_time(self):
        return self._slm.get_synapse_dynamics_time()

    @property
    def init_sparse_time(self):
        return self._slm.get_init_sparse_time()

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
        if pop_name in self.neuron_populations:
            raise ValueError("Neuron population '{0}'"
                             "already exists".format(pop_name))

        n_group = NeuronGroup(pop_name, self)
        n_group.set_neuron(neuron, param_space, var_space)
        n_group.add_to(self._model, int(num_neurons))

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
        matrix_type                 --  type of the matrix as string
        delay_steps                 --  delay in number of steps
        source                      --  source neuron group (either name or NeuronGroup object)
        target                      --  target neuron group (either name or NeuronGroup object)
        w_update_model              --  type of the WeightUpdateModels class
                                        as string or instance of weight update
                                        model class derived from
                                        ``pygenn.genn_wrapper.WeightUpdateModels.Custom`` (see also
                                        pygenn.genn_model.create_custom_weight_update_class)
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
                                        ``pygenn.genn_wrapper.PostsynapticModels.Custom`` (see also
                                        pygenn.genn_model.create_custom_postsynaptic_class)
        ps_param_space              --  dict with param values for the
                                        PostsynapticModels class
        ps_var_space                --  dict with initial variable values for
                                        the PostsynapticModels class
        connectivity_initialiser    --  InitSparseConnectivitySnippet::Init
                                        for connectivity
        """
        if self._built:
            raise Exception("GeNN model already built")

        if pop_name in self.synapse_populations:
            raise ValueError("synapse population '{0}' "
                             "already exists".format(pop_name))

        # Validate source and target groups
        source = self._validate_neuron_group(source, "source")
        target = self._validate_neuron_group(target, "target")
        
        s_group = SynapseGroup(pop_name, self)
        s_group.matrix_type = matrix_type
        s_group.set_connected_populations(source, target)
        s_group.set_weight_update(w_update_model, wu_param_space, wu_var_space,
                                  wu_pre_var_space, wu_post_var_space)
        s_group.set_post_syn(postsyn_model, ps_param_space, ps_var_space)
        s_group.connectivity_initialiser = connectivity_initialiser
        s_group.add_to(self._model, delay_steps)

        self.synapse_populations[pop_name] = s_group

        return s_group
    
    def add_slave_synapse_population(self, pop_name, master_pop, delay_steps,
                                     source, target, postsyn_model,
                                     ps_param_space, ps_var_space):
        """Add a 'slave' population to the GeNN model which shares 
        weights and connectivity with a 'master' population

        Args:
        pop_name                    --  name of the new population
        master_pop                  --  master synapse group to share weights with 
                                        (either name or SynapseGroup object)
        delay_steps                 --  delay in number of steps
        source                      --  source neuron group (either name or NeuronGroup object)
        target                      --  target neuron group (either name or NeuronGroup object)
        postsyn_model               --  type of the PostsynapticModels class
                                        as string or instance of postsynaptic
                                        model class derived from
                                        ``pygenn.genn_wrapper.PostsynapticModels.Custom`` (see also
                                        pygenn.genn_model.create_custom_postsynaptic_class)
        ps_param_space              --  dict with param values for the
                                        PostsynapticModels class
        ps_var_space                --  dict with initial variable values for
                                        the PostsynapticModels class
        """
        if self._built:
            raise Exception("GeNN model already built")

        if pop_name in self.synapse_populations:
            raise ValueError("synapse population '{0}' "
                             "already exists".format(pop_name))
        
        # Validate source and target groups
        source = self._validate_neuron_group(source, "source")
        target = self._validate_neuron_group(target, "target")
        
        master_pop = self._validate_synapse_group(master_pop, "master_pop")
        
        s_group = SynapseGroup(pop_name, self, master_pop)
        s_group.set_connected_populations(source, target)
        s_group.set_post_syn(postsyn_model, ps_param_space, ps_var_space)
        s_group.add_to(self._model, delay_steps)

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

        if cs_name in self.current_sources:
            raise ValueError("current source '{0}' "
                             "already exists".format(cs_name))

        # Validate population
        pop = self._validate_neuron_group(pop, "pop")
        
        c_source = CurrentSource(cs_name, self)
        c_source.set_current_source_model(current_source_model,
                                          param_space, var_space)
        c_source.add_to(self._model, pop)

        self.current_sources[cs_name] = c_source

        return c_source

    def build(self, path_to_model="./"):
        """Finalize and build a GeNN model

        Keyword args:
        path_to_model   --  path where to place the generated model code.
                            Defaults to the local directory.
        """

        if self._built:
            raise Exception("GeNN model already built")
        self._path_to_model = path_to_model

        # Create output path
        output_path = path.join(path_to_model, self.model_name + "_CODE")

        # Finalize model
        self._model.finalize()

        # Create suitable preferences object for backend
        preferences = self._backend_module.Preferences()

        # Set attributes on preferences object
        for k, v in iteritems(self._preferences):
            if hasattr(preferences, k):
                setattr(preferences, k, v)

        # Create backend
        backend = self._backend_module.create_backend(self._model, output_path, self.backend_log_level, preferences);

        # Generate code
        mem_alloc = genn_wrapper.generate_code(self._model, backend, output_path, 0)

        # **YUCK** SWIG doesn't handle return objects returned by value very well so delete manually
        backend = None

        # Build code
        if system() == "Windows":
            check_call(["msbuild", "/p:Configuration=Release", "/m", "/verbosity:minimal",
                        path.join(output_path, "runner.vcxproj")])
        else:
            check_call(["make", "-j", str(cpu_count(logical=False)), "-C", output_path])

        self._built = True
        return mem_alloc

    def load(self, path_to_model="./"):
        """import the model as shared library and initialize it"""
        if self._loaded:
            raise Exception("GeNN model already loaded")
        self._path_to_model = path_to_model

        self._slm.open(self._path_to_model, self.model_name)

        self._slm.allocate_mem()

        # Loop through synapse populations and load any 
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

        # Initialize model
        self._slm.initialize()

        # Loop through neuron populations
        for pop_data in itervalues(self.neuron_populations):
            pop_data.load()

        # Loop through synapse populations
        for pop_data in itervalues(self.synapse_populations):
            pop_data.load()

        # Loop through current sources
        for src_data in itervalues(self.current_sources):
            src_data.load()

        # Now everything is set up call the sparse initialisation function
        self._slm.initialize_sparse()

        # Set loaded flag and built flag
        self._loaded = True
        self._built = True

    def reinitialise(self):
        """reinitialise model to its original state without re-loading"""
        if not self._loaded:
            raise Exception("GeNN model has to be loaded before reinitialising")

        # Initialise dense and neuron variables
        self._slm.initialize()

        # Loop through neuron populations
        for pop_data in itervalues(self.neuron_populations):
            pop_data.reinitialise()

        # Loop through synapse populations
        for pop_data in itervalues(self.synapse_populations):
            pop_data.reinitialise()

        # Loop through current sources
        for src_data in itervalues(self.current_sources):
            src_data.reinitialise()

        # Initialise any sparse variables
        self._slm.initialize_sparse()

    def step_time(self):
        if not self._loaded:
            raise Exception("GeNN model has to be loaded before stepping")

        """Make one simulation step"""
        self._slm.step_time()

    def pull_state_from_device(self, pop_name):
        """Pull state from the device for a given population"""
        if not self._loaded:
            raise Exception("GeNN model has to be loaded before pulling")

        self._slm.pull_state_from_device(pop_name)

    def pull_spikes_from_device(self, pop_name):
        """Pull spikes from the device for a given population"""
        if not self._loaded:
            raise Exception("GeNN model has to be loaded before pulling")

        self._slm.pull_spikes_from_device(pop_name)

    def pull_current_spikes_from_device(self, pop_name):
        """Pull spikes from the device for a given population"""
        if not self._loaded:
            raise Exception("GeNN model has to be loaded before pulling")

        self._slm.pull_current_spikes_from_device(pop_name)

    def pull_connectivity_from_device(self, pop_name):
        """Pull connectivity from the device for a given population"""
        if not self._loaded:
            raise Exception("GeNN model has to be loaded before pulling")

        self._slm.pull_connectivity_from_device(pop_name)

    def pull_var_from_device(self, pop_name, var_name):
        """Pull variable from the device for a given population"""
        if not self._loaded:
            raise Exception("GeNN model has to be loaded before pulling")

        self._slm.pull_var_from_device(pop_name, var_name)

    def pull_extra_global_param_from_device(self, pop_name, egp_name, size=1):
        """Pull extra global parameter from the device for a given population"""
        if not self._loaded:
            raise Exception("GeNN model has to be loaded before pulling")

        self._slm.pull_extra_global_param(pop_name, egp_name, size)

    def push_state_to_device(self, pop_name):
        """Push state to the device for a given population"""
        if not self._loaded:
            raise Exception("GeNN model has to be loaded before pushing")

        self._slm.push_state_to_device(pop_name)

    def push_spikes_to_device(self, pop_name):
        """Push spikes to the device for a given population"""
        if not self._loaded:
            raise Exception("GeNN model has to be loaded before pushing")

        self._slm.push_spikes_to_device(pop_name)

    def push_current_spikes_to_device(self, pop_name):
        """Push current spikes to the device for a given population"""
        if not self._loaded:
            raise Exception("GeNN model has to be loaded before pushing")

        self._slm.push_current_spikes_to_device(pop_name)

    def push_connectivity_to_device(self, pop_name):
        """Push connectivity to the device for a given population"""
        if not self._loaded:
            raise Exception("GeNN model has to be loaded before pushing")

        self._slm.push_connectivity_to_device(pop_name)

    def push_var_to_device(self, pop_name, var_name):
        """Push variable to the device for a given population"""
        if not self._loaded:
            raise Exception("GeNN model has to be loaded before pushing")

        self._slm.push_var_to_device(pop_name, var_name)

    def push_extra_global_param_to_device(self, pop_name, egp_name, size=1):
        """Push extra global parameter to the device for a given population"""
        if not self._loaded:
            raise Exception("GeNN model has to be loaded before pushing")

        self._slm.push_extra_global_param(pop_name, egp_name, size)

    def end(self):
        """Free memory"""
        for group in [self.neuron_populations, self.current_sources]:
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

def init_var(init_var_snippet, param_space):
    """This helper function creates a VarInit object
    to easily initialise a variable using a snippet.

    Args:
    init_var_snippet    --  type of the InitVarSnippet class as string or
                            instance of class derived from
                            InitVarSnippet::Custom class.
    param_space         --  dict with param values for the InitVarSnippet class
    """
    # Prepare snippet
    (s_instance, s_type, param_names, params) = \
        prepare_snippet(init_var_snippet, param_space,
                        genn_wrapper.InitVarSnippet)

    # **YUCK** VarInit (and GeNN) assume that the snippet will live forever but
    # as far as Python is concerned, s_instance is never used again so it will be
    # destroyed. Disowning it here hands over it's ownership to C++
    # **NOTE** this isn't the case with models as references to neuron and synapse
    # models are kept within NeuronGroup and SynapseGroup objects
    s_instance.__disown__()

    # Use add function to create suitable VarInit
    return VarInit(s_instance, params)


def init_connectivity(init_sparse_connect_snippet, param_space):
    """This helper function creates a InitSparseConnectivitySnippet::Init
    object to easily initialise connectivity using a snippet.

    Args:
    init_sparse_connect_snippet --  type of the InitSparseConnectivitySnippet
                                    class as string or instance of class
                                    derived from
                                    InitSparseConnectivitySnippet::Custom.
    param_space                 --  dict with param values for the
                                    InitSparseConnectivitySnippet class
    """
    # Prepare snippet
    (s_instance, s_type, param_names, params) = \
        prepare_snippet(init_sparse_connect_snippet, param_space,
                        genn_wrapper.InitSparseConnectivitySnippet)

    # **YUCK** VarInit (and GeNN) assume that the snippet will live forever but
    # as far as Python is concerned, s_instance is never used again so it will be
    # destroyed. Disowning it here hands over it's ownership to C++
    # **NOTE** this isn't the case with models as references to neuron and synapse
    # models are kept within NeuronGroup and SynapseGroup objects
    s_instance.__disown__()

    # Use add function to create suitable VarInit
    return Init(s_instance, params)


def create_custom_neuron_class(class_name, param_names=None,
                               var_name_types=None, derived_params=None,
                               sim_code=None, threshold_condition_code=None,
                               reset_code=None, support_code=None,
                               extra_global_params=None,
                               additional_input_vars=None,
                               is_auto_refractory_required=None,
                               custom_body=None):
    """This helper function creates a custom NeuronModel class.
    See also:
    create_custom_postsynaptic_class
    create_custom_weight_update_class
    create_custom_current_source_class
    create_custom_init_var_snippet_class
    create_custom_sparse_connect_init_snippet_class

    Args:
    class_name                  --  name of the new class

    Keyword args:
    param_names                 --  list of strings with param names
                                    of the model
    var_name_types              --  list of pairs of strings with varible names
                                    and types of the model
    derived_params              --  list of pairs, where the first member
                                    is string with name of the derived
                                    parameter and the second MUST be an
                                    instance of a class which inherits from
                                    ``pygenn.genn_wrapper.Snippet.DerivedParamFunc``
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
    custom_body                 --  dictionary with additional attributes and
                                    methods of the new class
    """
    if not isinstance(custom_body, dict) and custom_body is not None:
        raise ValueError("custom_body must be an isinstance of dict or None")

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

    if extra_global_params is not None:
        body["get_extra_global_params"] = \
            lambda self: EGPVector([EGP(egp[0], egp[1])
                                    for egp in extra_global_params])

    if additional_input_vars:
        body["get_additional_input_vars"] = \
            lambda self: ParamValVector([ParamVal(a[0], a[1], a[2])
                                         for a in additional_input_vars])

    if is_auto_refractory_required is not None:
        body["is_auto_refractory_required"] = \
            lambda self: is_auto_refractory_required

    if custom_body is not None:
        body.update(custom_body)

    return create_custom_model_class(
        class_name, genn_wrapper.NeuronModels.Custom, param_names,
        var_name_types, derived_params, body)


def create_custom_postsynaptic_class(class_name, param_names=None,
                                     var_name_types=None, derived_params=None,
                                     decay_code=None, apply_input_code=None,
                                     support_code=None, custom_body=None):
    """This helper function creates a custom PostsynapticModel class.
    See also:
    create_custom_neuron_class
    create_custom_weight_update_class
    create_custom_current_source_class
    create_custom_init_var_snippet_class
    create_custom_sparse_connect_init_snippet_class

    Args:
    class_name          --  name of the new class

    Keyword args:
    param_names         --  list of strings with param names of the model
    var_name_types      --  list of pairs of strings with varible names and
                            types of the model
    derived_params      --  list of pairs, where the first member is string
                            with name of the derived parameter and the second
                            MUST be an instance of a class which inherits
                            from ``pygenn.genn_wrapper.DerivedParamFunc``
    decay_code          --  string with the decay code
    apply_input_code    --  string with the apply input code
    support_code        --  string with the support code
    custom_body         --  dictionary with additional attributes and methods
                            of the new class
    """
    if not isinstance(custom_body, dict) and custom_body is not None:
        raise ValueError()

    body = {}

    if decay_code is not None:
        body["get_decay_code"] = lambda self: dedent(decay_code)

    if apply_input_code is not None:
        body["get_apply_input_code"] = lambda self: dedent(apply_input_code)

    if support_code is not None:
        body["get_support_code"] = lambda self: dedent(support_code)

    if custom_body is not None:
        body.update(custom_body)

    return create_custom_model_class(
        class_name, genn_wrapper.PostsynapticModels.Custom, param_names,
        var_name_types, derived_params, body)


def create_custom_weight_update_class(class_name, param_names=None,
                                      var_name_types=None,
                                      pre_var_name_types=None,
                                      post_var_name_types=None,
                                      derived_params=None, sim_code=None,
                                      event_code=None, learn_post_code=None,
                                      synapse_dynamics_code=None,
                                      event_threshold_condition_code=None,
                                      pre_spike_code=None,
                                      post_spike_code=None,
                                      sim_support_code=None,
                                      learn_post_support_code=None,
                                      synapse_dynamics_suppport_code=None,
                                      extra_global_params=None,
                                      is_pre_spike_time_required=None,
                                      is_post_spike_time_required=None,
                                      custom_body=None):
    """This helper function creates a custom WeightUpdateModel class.
    See also:
    create_custom_neuron_class
    create_custom_postsynaptic_class
    create_custom_current_source_class
    create_custom_init_var_snippet_class
    create_custom_sparse_connect_init_snippet_class

    Args:
    class_name                      --  name of the new class

    Keyword args:
    param_names                     --  list of strings with param names of
                                        the model
    var_name_types                  --  list of pairs of strings with variable
                                        names and types of the model
    pre_var_name_types              --  list of pairs of strings with
                                        presynaptic variable names and
                                        types of the model
    post_var_name_types             --  list of pairs of strings with
                                        postsynaptic variable names and
                                        types of the model
    derived_params                  --  list of pairs, where the first member
                                        is string with name of the derived
                                        parameter and the second MUST be an
                                        instance of a class which inherits from
                                        ``pygenn.genn_wrapper.DerivedParamFunc``
    sim_code                        --  string with the simulation code
    event_code                      --  string with the event code
    learn_post_code                 --  string with the code to include in
                                        learn_synapse_post kernel/function
    synapse_dynamics_code           --  string with the synapse dynamics code
    event_threshold_condition_code  --  string with the event threshold
                                        condition code
    pre_spike_code                  --  string with the code run once per
                                        spiking presynaptic neuron
    post_spike_code                 --  string with the code run once per
                                        spiking postsynaptic neuron
    sim_support_code                --  string with simulation support code
    learn_post_support_code         --  string with support code for
                                        learn_synapse_post kernel/function
    synapse_dynamics_suppport_code  --  string with synapse dynamics
                                        support code
    extra_global_params             --  list of pairs of strings with names and
                                        types of additional parameters
    is_pre_spike_time_required      --  boolean, is presynaptic spike time
                                        required in any weight update kernels?
    is_post_spike_time_required     --  boolean, is postsynaptic spike time
                                        required in any weight update kernels?
    custom_body                     --  dictionary with additional attributes
                                        and methods of the new class
    """
    if not isinstance(custom_body, dict) and custom_body is not None:
        raise ValueError("custom_body must be an instance of dict or None")

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

    if sim_support_code is not None:
        body["get_sim_support_code"] = lambda self: dedent(sim_support_code)

    if learn_post_support_code is not None:
        body["get_learn_post_support_code"] = \
            lambda self: dedent(learn_post_support_code)

    if synapse_dynamics_suppport_code is not None:
        body["get_synapse_dynamics_suppport_code"] = \
            lambda self: dedent(synapse_dynamics_suppport_code)

    if extra_global_params is not None:
        body["get_extra_global_params"] = \
            lambda self: EGPVector([EGP(egp[0], egp[1])
                                    for egp in extra_global_params])

    if pre_var_name_types is not None:
        body["get_pre_vars"] = \
            lambda self: VarVector([Var(*vn)
                                    for vn in pre_var_name_types])

    if post_var_name_types is not None:
        body["get_post_vars"] = \
            lambda self: VarVector([Var(*vn)
                                    for vn in post_var_name_types])

    if is_pre_spike_time_required is not None:
        body["is_pre_spike_time_required"] = \
            lambda self: is_pre_spike_time_required

    if is_post_spike_time_required is not None:
        body["is_post_spike_time_required"] = \
            lambda self: is_post_spike_time_required

    if custom_body is not None:
        body.update(custom_body)

    return create_custom_model_class(
        class_name, genn_wrapper.WeightUpdateModels.Custom, param_names,
        var_name_types, derived_params, body)


def create_custom_current_source_class(class_name, param_names=None,
                                       var_name_types=None,
                                       derived_params=None,
                                       injection_code=None,
                                       extra_global_params=None,
                                       custom_body=None):
    """This helper function creates a custom NeuronModel class.
    See also:
    create_custom_neuron_class
    create_custom_weight_update_class
    create_custom_current_source_class
    create_custom_init_var_snippet_class
    create_custom_sparse_connect_init_snippet_class

    Args:
    class_name          --  name of the new class

    Keyword args:
    param_names         --  list of strings with param names of the model
    var_name_types      --  list of pairs of strings with varible names and
                            types of the model
    derived_params      --  list of pairs, where the first member is string
                            with name of the derived parameter and the second
                            MUST be an instance of the class which inherits
                            from  ``pygenn.genn_wrapper.DerivedParamFunc``
    injection_code      --  string with the current injection code
    extra_global_params --  list of pairs of strings with names and types of
                            additional parameters
    custom_body         --  dictionary with additional attributes and methods
                            of the new class
    """
    if not isinstance(custom_body, dict) and custom_body is not None:
        raise ValueError("custom_body must be an instance of dict or None")

    body = {}

    if injection_code is not None:
        body["get_injection_code"] = lambda self: dedent(injection_code)

    if extra_global_params is not None:
        body["get_extra_global_params"] = \
            lambda self: EGPVector([EGP(egp[0], egp[1])
                                    for egp in extra_global_params])

    if custom_body is not None:
        body.update(custom_body)

    return create_custom_model_class(
        class_name, genn_wrapper.CurrentSourceModels.Custom, param_names,
        var_name_types, derived_params, body)


def create_custom_model_class(class_name, base, param_names, var_name_types,
                              derived_params, custom_body):
    """This helper function completes a custom model class creation.

    This part is common for all model classes and is nearly useless on its own
    unless you specify custom_body.
    See also:
    create_custom_neuron_class
    create_custom_weight_update_class
    create_custom_postsynaptic_class
    create_custom_current_source_class
    create_custom_init_var_snippet_class
    create_custom_sparse_connect_init_snippet_class

    Args:
    class_name      --  name of the new class
    base            --  base class
    param_names     --  list of strings with param names of the model
    var_name_types  --  list of pairs of strings with varible names and
                        types of the model
    derived_params  --  list of pairs, where the first member is string with
                        name of the derived parameter and the second MUST be
                        an instance of the class which inherits from the
                        ``pygenn.genn_wrapper.DerivedParamFunc`` class
    custom_body     --  dictionary with attributes and methods of the new class
    """

    def ctor(self):
        base.__init__(self)

    body = {
        "__init__": ctor,
    }

    if param_names is not None:
        body["get_param_names"] = lambda self: StringVector(param_names)

    if var_name_types is not None:
        body["get_vars"] = \
            lambda self: VarVector([Var(*vn)
                                    for vn in var_name_types])

    if derived_params is not None:
        body["get_derived_params"] = \
            lambda self: DerivedParamVector([DerivedParam(dp[0], make_dpf(dp[1]))
                                             for dp in derived_params])

    if custom_body is not None:
        body.update(custom_body)

    return type(class_name, (base,), body)()


def create_dpf_class(dp_func):
    """Helper function to create derived parameter function class

    Args:
    dp_func --  a function which computes the derived parameter and takes
                two args "pars" (vector of double) and "dt" (double)
    """
    dpf = genn_wrapper.Snippet.DerivedParamFunc

    def ctor(self):
        dpf.__init__(self)

    def call(self, pars, dt):
        return dp_func(pars, dt)

    return type("", (dpf,), {"__init__": ctor, "__call__": call})


def create_cmlf_class(cml_func):
    """Helper function to create function class for calculating sizes of
    matrices initialised with sparse connectivity initialisation snippet

    Args:
    cml_func -- a function which computes the length and takes
                three args "num_pre" (unsigned int), "num_post" (unsigned int)
                and "pars" (vector of double)
    """
    cmlf = genn_wrapper.InitSparseConnectivitySnippet.CalcMaxLengthFunc

    def ctor(self):
        cmlf.__init__(self)

    def call(self, num_pre, num_post, pars):
        return cml_func(num_pre, num_post, pars)

    return type("", (cmlf,), {"__init__": ctor, "__call__": call})


def create_custom_init_var_snippet_class(class_name, param_names=None,
                                         derived_params=None,
                                         var_init_code=None, 
                                         extra_global_params=None,
                                         group_params=None,
                                         pre_params=None,
                                         post_params=None,
                                         custom_body=None):
    """This helper function creates a custom InitVarSnippet class.
    See also:
    create_custom_neuron_class
    create_custom_weight_update_class
    create_custom_postsynaptic_class
    create_custom_current_source_class
    create_custom_sparse_connect_init_snippet_class

    Args:
    class_name      --  name of the new class

    Keyword args:
    param_names     --  list of strings with param names of the model
    derived_params  --  list of pairs, where the first member is string with
                        name of the derived parameter and the second MUST be
                        an instance of the `pygenn.genn_wrapper.DerivedParamFunc`` class
    var_init_code   --  string with the variable initialization code
    extra_global_params     --  list of pairs of strings with names and
                                types of additional parameters
    custom_body     --  dictionary with additional attributes and methods of
                        the new class
    """

    if not isinstance(custom_body, dict) and custom_body is not None:
        raise ValueError("custom_body must be an instance of dict or None")

    body = {}

    if var_init_code is not None:
        body["get_code"] = lambda self: dedent(var_init_code)

    if extra_global_params is not None:
        body["get_extra_global_params"] = \
            lambda self: EGPVector([EGP(egp[0], egp[1])
                                    for egp in extra_global_params])
    
    if group_params is not None:
        body["get_group_params"] = \
            lambda self: ParamValVector([ParamVal(p[0], p[1], p[2])
                                         for p in group_params])
    
    if pre_params is not None:
        body["get_pre_params"] = \
            lambda self: ParamValVector([ParamVal(p[0], p[1], p[2])
                                         for p in pre_params])
    
    if post_params is not None:
        body["get_post_params"] = \
            lambda self: ParamValVector([ParamVal(p[0], p[1], p[2])
                                         for p in post_params])
                                         
    if custom_body is not None:
        body.update(custom_body)

    return create_custom_model_class(
        class_name, genn_wrapper.InitVarSnippet.Custom, param_names,
        None, derived_params, body)


def create_custom_sparse_connect_init_snippet_class(class_name,
                                                    param_names=None,
                                                    derived_params=None,
                                                    row_build_code=None,
                                                    row_build_state_vars=None,
                                                    calc_max_row_len_func=None,
                                                    calc_max_col_len_func=None,
                                                    extra_global_params=None,
                                                    custom_body=None):
    """This helper function creates a custom
    InitSparseConnectivitySnippet class.
    See also:
    create_custom_neuron_class
    create_custom_weight_update_class
    create_custom_postsynaptic_class
    create_custom_current_source_class
    create_custom_init_var_snippet_class

    Args:
    class_name              --  name of the new class

    Keyword args:
    param_names             --  list of strings with param names of the model
    derived_params          --  list of pairs, where the first member is string
                                with name of the derived parameter and the
                                second MUST be an instance of the class which
                                inherits from ``pygenn.genn_wrapper.DerivedParamFunc``
    row_build_code          --  string with row building initialization code
    row_build_state_vars    --  list of tuples of state variables, their types
                                and their initial values to use across
                                row building loop
    calc_max_row_len_func   --  instance of class inheriting from
                                CalcMaxLengthFunc used to calculate maximum
                                row length of synaptic matrix
    calc_max_col_len_func   --  instance of class inheriting from
                                CalcMaxLengthFunc used to calculate maximum
                                col length of synaptic matrix
    extra_global_params     --  list of pairs of strings with names and
                                types of additional parameters
    custom_body             --  dictionary with additional attributes and
                                methods of the new class
    """

    if not isinstance(custom_body, dict) and custom_body is not None:
        raise ValueError("custom_body must be an instance of dict or None")

    body = {}

    if row_build_code is not None:
        body["get_row_build_code"] = lambda self: dedent(row_build_code)

    if row_build_state_vars is not None:
        body["get_row_build_state_vars"] = \
            lambda self: ParamValVector([ParamVal(r[0], r[1], r[2])
                                         for r in row_build_state_vars])

    if calc_max_row_len_func is not None:
        body["get_calc_max_row_length_func"] = \
            lambda self: make_cmlf(calc_max_row_len_func)

    if calc_max_col_len_func is not None:
        body["get_calc_max_col_length_func"] = \
            lambda self: make_cmlf(calc_max_col_len_func)

    if extra_global_params is not None:
        body["get_extra_global_params"] = \
            lambda self: EGPVector([EGP(egp[0], egp[1])
                                    for egp in extra_global_params])

    if custom_body is not None:
        body.update(custom_body)

    return create_custom_model_class(
        class_name, genn_wrapper.InitSparseConnectivitySnippet.Custom, param_names,
        None, derived_params, body)

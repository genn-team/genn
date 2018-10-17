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
            # when you pull state from device, views of all variables are updated
            # and show current simulated values
            gm.pull_state_from_device(neuron_pop_name)
            # finally, record voltage by copying form view into array.
            Vs[i,:] = v_view
"""
# python imports
from os import path
from subprocess import check_call # to call make
# 3rd party imports
import numpy as np
from six import iteritems
# pygenn imports
import genn_wrapper
import genn_wrapper.SharedLibraryModel as slm
from genn_wrapper.NewModels import VarInit
from genn_wrapper.InitSparseConnectivitySnippet import Init
from genn_wrapper.Snippet import make_dpf
from genn_wrapper.InitSparseConnectivitySnippet import make_cmlf
from genn_wrapper.StlContainers import (StringPair, StringStringDoublePairPair,
                                        StringDPFPair, StringDoublePair,
                                        StringPairVector, StringVector,
                                        StringDPFPairVector,
                                        StringStringDoublePairPairVector)

from genn_groups import NeuronGroup, SynapseGroup, CurrentSource
from model_preprocessor import prepare_snippet

class GeNNModel(object):

    """GeNNModel class
    This class helps to define, build and run a GeNN model from python
    """

    def __init__(self, precision=None, model_name="GeNNModel",
                 enable_debug=False, cpu_only=False):
        """Init GeNNModel
        Keyword args:
        precision    -- string precision as string ("float" or "double" or "long double")
                     Defaults to float.
        model_name    -- string name of the model. Defaults to "GeNNModel".
        enable_debug  -- boolean enable debug mode. Disabled by default.
        cpu_only      -- boolean whether GeNN should run only on CPU. Disabled by default.
        """
        self._scalar = precision
        if precision is None or precision == "float":
            genn_float_type = "GENN_FLOAT"
            self._scalar = "float"
            self._slm = slm.SharedLibraryModel_f()
            self._np_type = np.float32
        elif precision == "double":
            genn_float_type = "GENN_DOUBLE"
            self._slm = slm.SharedLibraryModel_d()
            self._np_type = np.float64
        elif precision == "long double":
            genn_float_type = "GENN_LONG_DOUBLE"
            self._slm = slm.SharedLibraryModel_ld()
            self._np_type = np.float128
        else:
            raise ValueError(
                "Supported precisions are float, double and "\
                "long double, but '{1}' was given".format(precision))
        
        self._built = False
        self._cpu_only = cpu_only
        self._localhost = genn_wrapper.initMPI_pygenn()
        
        self.default_var_mode = genn_wrapper.VarMode_LOC_HOST_DEVICE_INIT_DEVICE
        genn_wrapper.GeNNPreferences.cvar.debugCode = enable_debug
        self._model = genn_wrapper.NNmodel()
        self._model.set_precision(getattr(genn_wrapper, genn_float_type))
        self.model_name = model_name
        self.neuron_populations = {}
        self.synapse_populations = {}
        self.current_sources = {}
        self.dT = 0.1

    @property
    def default_var_mode(self):
        """Default variable mode - defines how and where state variables are initialised"""
        return genn_wrapper.GeNNPreferences.cvar.defaultVarMode

    @default_var_mode.setter
    def default_var_mode(self, mode):
        if self._built:
            raise Exception("GeNN model already built")

        genn_wrapper.set_default_var_mode(mode)

    @property
    def default_sparse_connectivity_mode(self):
        """Default sparse connectivity mode - how and where connectivity is initialised"""
        return genn_wrapper.GeNNPreferences.cvar.defaultSparseConnectivityMode

    @default_sparse_connectivity_mode.setter
    def default_sparse_connectivity_mode(self, mode):
        if self._built:
            raise Exception("GeNN model already built")

        genn_wrapper.set_default_sparse_connectivity_mode(mode)

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
        return self._T[0]

    def timestep(self):
        """Simulation time step"""
        return self._TS[0]

    @property
    def dT(self):
        """Step size"""
        return self._model.get_dt()

    @dT.setter 
    def dT(self, dt):
        if self._built:
            raise Exception("GeNN model already built")
        self._model.set_dt(dt)
    
    def add_neuron_population( self, pop_name, num_neurons, neuron,
                             param_space, var_space ):
        """Add a neuron population to the GeNN model

        Args:
        pop_name     -- name of the new population
        num_neurons  -- number of neurons in the new population
        neuron      -- type of the NeuronModels class as string or instance of neuron class
                        derived from NeuronModels::Custom class.
                        sa create_custom_neuron_class
        param_space  -- dict with param values for the NeuronModels class
        var_space    -- dict with initial variable values for the NeuronModels class
        """
        if self._built:
            raise Exception("GeNN model already built")
        if pop_name in self.neuron_populations:
            raise ValueError("Neuron population '{0}' already exists".format(pop_name))
       
        n_group = NeuronGroup(pop_name)
        n_group.set_neuron(neuron, param_space, var_space)
        n_group.add_to(self._model, int(num_neurons))

        self.neuron_populations[pop_name] = n_group

        return n_group

    def add_synapse_population(self, pop_name, matrix_type, delay_steps,
                             source, target,
                             w_update_model, wu_param_space, wu_var_space, wu_pre_var_space, wu_post_var_space,
                             postsyn_model, ps_param_space, ps_var_space,
                             connectivity_initialiser=genn_wrapper.uninitialised_connectivity()):
        """Add a synapse population to the GeNN model

        Args:
        pop_name      -- name of the new population
        matrix_type   -- type of the matrix as string
        delay_steps   -- delay in number of steps
        source       -- name of the source population
        target       -- name of the target population
        w_uptate_model -- type of the WeightUpdateModels class as string or instance of weight update model class
                            derived from WeightUpdateModels::Custom class.
                            sa createCustomWeightUpdateClass
        wu_param_values   -- dict with param values for the WeightUpdateModels class
        wu_init_var_values -- dict with initial variable values for the WeightUpdateModels class
        postsyn_model    -- type of the PostsynapticModels class as string or instance of postsynaptic model class
                            derived from PostsynapticModels::Custom class.
                            sa create_custom_postsynaptic_class
        postsyn_param_values   -- dict with param values for the PostsynapticModels class
        postsyn_init_var_values -- dict with initial variable values for the PostsynapticModels class
        connectivity_initialiser -- InitSparseConnectivitySnippet::Init for connectivity
        """
        if self._built:
            raise Exception("GeNN model already built")

        if pop_name in self.synapse_populations:
            raise ValueError("synapse population '{0}' already exists".format(pop_name))

        s_group = SynapseGroup(pop_name)
        s_group.matrix_type = matrix_type
        s_group.set_connected_populations(
            source, self.neuron_populations[source].size,
            target, self.neuron_populations[target].size )
        s_group.set_weight_update(w_update_model, wu_param_space,
                          wu_var_space, wu_pre_var_space, wu_post_var_space)
        s_group.set_post_syn(postsyn_model, ps_param_space, ps_var_space)
        s_group.add_to(self._model, delay_steps, connectivity_initialiser)

        self.synapse_populations[pop_name] = s_group

        return s_group

    def add_current_source(self, cs_name, current_source_model, pop_name,
                         param_space, var_space):
        """Add a current source to the GeNN model

        Args:
        cs_name      -- name of the new current source
        current_source_model -- type of the CurrentSourceModels class as string or
                              instance of CurrentSourceModels class derived from
                              CurrentSourceModels::Custom class
                              sa createCustomCurrentSourceClass
        pop_name     -- name of the population into which the current source should be injected
        param_space  -- dict with param values for the CurrentSourceModels class
        var_space    -- dict with initial variable values for the CurrentSourceModels class
        """
        if self._built:
            raise Exception("GeNN model already built")
        if pop_name not in self.neuron_populations:
            raise ValueError("neuron population '{0}' does not exist".format(pop_name))
        if cs_name in self.current_sources:
            raise ValueError("current source '{0}' already exists".format(cs_name))

        c_source = CurrentSource(cs_name)
        c_source.set_current_source_model(current_source_model, param_space, var_space)
        c_source.add_to(self._model, self.neuron_populations[pop_name])

        self.current_sources[cs_name] = c_source

        return c_source

    def initialize_var_on_device(self, pop_name, var_name, mask, vals):
        """Set values for the given variable and population and push them to the device
        
        Note: shapes of mask and vals must be the same

        Args:
        pop_name -- name of the population for which the values must be set
        var_name -- name of the variable for which the values must be set
        mask    -- list with neurons ids
        vals    -- list with variable values
        """
        if pop_name not in self.neuron_populations:
            if pop_name not in self.synapse_populations:
                raise ValueError(
                    "Failed to initialize variable '{0}': " \
                    "population '{1}' does not exist".format(var_name, pop_name))
            else:
                var = self.synapse_populations[pop_name].vars[var_name]
        else:
            var = self.neuron_populations[pop_name].vars[var_name]

        var.view[mask] = vals
        self.push_state_to_device(pop_name)


    def initialize_spikes_on_device(self, pop_name, mask, targets, counts):
        """Set spike counts and targets for the given population and push them to the device
        
        Note: shapes of mask, targets and counts must be the same

        Args:
        pop_name -- name of the population for which the spikes must be set
        mask    -- list with source neurons ids
        targets -- list with target neuron ids
        counts  -- list with number of spikes for source neurons
        """
        if pop_name not in self.neuron_populations:
            raise ValueError(
                "Failed to initialize variable '{0}': " \
                "population '{1}' does not exist".format(var_name, pop_name))
        self.neuronPopulations[pop_name].spikes[mask] = targets
        self.neuronPopulations[pop_name].spike_count[mask] = counts
        self.push_spikes_to_device(pop_name)


    def build(self, path_to_model="./"):

        """Finalize and build a GeNN model
        
        Keyword args:
        path_to_model -- path where to place the generated model code. Defaults to the local directory.
        """

        if self._built:
            raise Exception("GeNN model already built")
        self._path_to_model = path_to_model

        for pop_name, pop_data in iteritems(self.synapse_populations):
            if pop_data.sparse:
                pop_data.pop.set_max_connections(pop_data.maxConn)

        self._model.finalize()
        genn_wrapper.generate_model_runner_pygenn(
            self._model, self._path_to_model, self._localhost)

        check_call(["make", "-C", path.join(path_to_model, self.model_name + "_CODE")])
        
        self._built = True

    def load(self):
        """import the model as shared library and initialize it"""
        if not self._built:
            raise Exception("GeNN model has to be built before running")
        
        self._slm.open(self._path_to_model, self.model_name)


        self._slm.allocate_mem()
        self._slm.initialize()
        
        if self._scalar == "float":
            self._T = self._slm.assign_external_pointer_single_f("t")
        if self._scalar == "double":
            self.T = self._slm.assign_external_pointer_single_d("t")
        if self._scalar == "long double":
            self._T = self._slm.assign_external_pointer_single_ld("t")
        self._TS = self._slm.assign_external_pointer_single_ull("iT")

        for pop_name, pop_data in iteritems(self.neuron_populations):
            self._slm.init_neuron_pop_io(pop_name)
            pop_data.spikes = self.assign_external_pointer_pop(
                pop_name, "glbSpk", pop_data.size * pop_data.delay_slots, "unsigned int")
            pop_data.spike_count = self.assign_external_pointer_pop(
                pop_name, "glbSpkCnt", pop_data.delay_slots, "unsigned int")
            if pop_data.delay_slots > 1:
                pop_data.spike_que_ptr = self._slm.assign_external_pointer_single_ui(
                      "spkQuePtr" + pop_name)

            for var_name, var_data in iteritems(pop_data.vars):
                var_data.view = self.assign_external_pointer_pop(
                    pop_name, var_name, pop_data.size, var_data.type )
                if var_data.init_required:
                    var_data.view[:] = var_data.values


            for egp_name, egp_data in iteritems(pop_data.extra_global_params):
                # if auto allocation is not enabled, let the user care about
                # allocation and initialization of the EGP
                if egp_data.needs_allocation:
                    self._slm.allocate_extra_global_param(pop_name, egp_name,
                                                       len(egp_data.values))
                    egp_data.view = self.assign_external_pointer_pop(
                        pop_name, egp_name,
                        len(egp_data.values), egp_data.type[:-1])
                    if egp_data.init_required:
                        egp_data.view[:] = egp_data.values

        for pop_name, pop_data in iteritems(self.synapse_populations):
            self._slm.init_synapse_pop_io(pop_name)
            
            if pop_data.sparse:
                if pop_data.connections_set:
                    self._slm.allocate_sparse_proj(pop_name, len(pop_data.ind))
                    self._slm.initialize_sparse_proj(pop_name, pop_data.ind,
                                                   pop_data.indInG)

                else:
                    raise Exception("For sparse projections, the connections"\
                                    "must be set before loading a model")

            for var_name, var_data in iteritems(pop_data.vars):
                size = pop_data.size
                if var_name in [vnt[0] for vnt in pop_data.postsyn.get_vars()]:
                    size = self.neuron_populations[pop_data.trg].size
                if var_name == "g" and pop_data.globalG:
                    continue
                var_data.view = self.assign_external_pointer_pop(
                        pop_name, var_name, size, var_data.type )
                if var_data.init_required:
                    if var_name == "g" and pop_data.connections_set and not pop_data.sparse:
                        var_data.view[:] = np.zeros((size,))
                        var_data.view[pop_data.gMask] = var_data.values
                    else:
                        var_data.view[:] = var_data.values

        for src_name, src_data in iteritems(self.current_sources):
            self._slm.init_current_source_io(src_name)

            for var_name, var_data in iteritems(src_data.vars):
                var_data.view = self.assign_external_pointer_pop(
                    src_name, var_name, src_data.size, var_data.type )
                if var_data.init_required:
                    var_data.view[:] = var_data.values

            for egp_name, egp_data in iteritems(src_data.extra_global_params):
                # if auto allocation is not enabled, let the user care about
                # allocation and initialization of the EGP
                if egp_data.needsAllocation:
                    self._slm.allocate_extra_global_param(src_name, egp_name,
                                                       len(egp_data.values))
                    egp_data.view = self.assign_external_pointer_pop(
                        src_name, egp_name, len(egp_data.values), egp_data.type[:-1])
                    if egp_data.init_required:
                        egp_data.view[:] = egp_data.values

        self._slm.initialize_model()

        if self._cpu_only:
            self.step_time = self._slm.step_time_cpu
        else:
            self.step_time = self._slm.step_time_gpu


    def assign_external_pointer_pop(self, pop_name, var_name, var_size, var_type):
        """Assign a population variable to an external numpy array
        
        Args:
        pop_name -- string population name
        var_name -- string a name of the variable to assing, without population name
        var_size -- int the size of the variable
        var_type -- string type of the variable. The supported types are
                   char, unsigned char, short, unsigned short, int, unsigned int,
                   long, unsigned long, long long, unsigned long long,
                   float, double, long double and scalar.

        Returns numpy array of type var_type

        Raises ValueError if variable type is not supported
        """

        return self.assign_external_pointer_array(var_name + pop_name, var_size, var_type)


    def assign_external_pointer_array(self, var_name, var_size, var_type):
        """Assign a variable to an external numpy array
        
        Args:
        var_name -- string a fully qualified name of the variable to assign
        var_size -- int the size of the variable
        var_type -- string type of the variable. The supported types are
                   char, unsigned char, short, unsigned short, int, unsigned int,
                   long, unsigned long, long long, unsigned long long,
                   float, double, long double and scalar.

        Returns numpy array of type var_type

        Raises ValueError if variable type is not supported
        """

        if var_type == "scalar":
            if self._scalar == "float":
                return self._slm.assign_external_pointer_array_f(var_name, var_size)
            elif self._scalar == "double":
                return self._slm.assign_external_pointer_array_d(var_name, var_size)
            elif self._scalar == "long double":
                return self._slm.assign_external_pointer_array_ld(var_name, var_size)

        elif var_type == "char":
            return self._slm.assign_external_pointer_array_c(var_name, var_size)
        elif var_type == "unsigned char":
            return self._slm.assign_external_pointer_array_uc(var_name, var_size)
        elif var_type == "short":
            return self._slm.assign_external_pointer_array_s(var_name, var_size)
        elif var_type == "unsigned short":
            return self._slm.assign_external_pointer_array_us(var_name, var_size)
        elif var_type == "int":
            return self._slm.assign_external_pointer_array_i(var_name, var_size)
        elif var_type == "unsigned int":
            return self._slm.assign_external_pointer_array_ui(var_name, var_size)
        elif var_type == "long":
            return self._slm.assign_external_pointer_array_l(var_name, var_size)
        elif var_type == "unsigned long":
            return self._slm.assign_external_pointer_array_ul(var_name, var_size)
        elif var_type == "long long":
            return self._slm.assign_external_pointer_array_ll(var_name, var_size)
        elif var_type == "unsigned long long":
            return self._slm.assign_external_pointer_array_ull(var_name, var_size)
        elif var_type == "float":
            return self._slm.assign_external_pointer_array_f(var_name, var_size)
        elif var_type == "double":
            return self._slm.assign_external_pointer_array_d(var_name, var_size)
        elif var_type == "long double":
            return self._slm.assign_external_pointer_array_ld(var_name, var_size)
        else:
            raise TypeError("unsupported var_type '{}'".format(var_type))

    def _stepTimeGPU(self):
        """Make one simulation step (for library built for CPU)"""
        self._slm.step_time_gpu()
    
    def _stepTimeCPU(self):
        """Make one simulation step (for library built for CPU)"""
        self._slm.step_time_cpu()

    def step_time(self):
        """Make one simulation step"""
        pass

    def pull_state_from_device(self, pop_name):
        """Pull state from the device for a given population"""
        if not self._built:
            raise Exception("GeNN model has to be built before running")
        if not self._cpu_only:
            self._slm.pull_state_from_device(pop_name)
    
    def pull_spikes_from_device(self, pop_name):
        """Pull spikes from the device for a given population"""
        if not self._built:
            raise Exception("GeNN model has to be built before running")
        if not self._cpu_only:
            self._slm.pull_spikes_from_device(pop_name)

    def pull_current_spikes_from_device(self, pop_name):
        """Pull spikes from the device for a given population"""
        if not self._built:
            raise Exception("GeNN model has to be built before running")
        if not self._cpu_only:
            self._slm.pull_current_spikes_from_device(pop_name)

    def push_state_to_device(self, pop_name):
        """Push state to the device for a given population"""
        if not self._built:
            raise Exception("GeNN model has to be built before running")
        if not self._cpu_only:
            self._slm.push_state_to_device(pop_name)

    def push_spikes_to_device(self, pop_name):
        """Push spikes from the device for a given population"""
        if not self._built:
            raise Exception("GeNN model has to be built before running")
        if not self._cpu_only:
            self._slm.push_spikes_to_device(pop_name)

    def push_current_spikes_from_device(self, pop_name):
        """Push spikes from the device for a given population"""
        if not self._built:
            raise Exception("GeNN model has to be built before running")
        if not self._cpu_only:
            self._slm.push_current_spikes_to_device(pop_name)

    def end(self):
        """Free memory"""
        for group in [self.neuron_populations, self.current_sources]:
            for group_name, group_data in iteritems(group):

                for egp_name, egp_data in iteritems(group_data.extra_global_params):
                    # if auto allocation is not enabled, let the user care about
                    # freeing of the EGP
                    if egp_data.needsAllocation:
                        self._slm.free_extra_global_param(group_name, egp_name)
        # "normal" variables are freed when SharedLibraryModel is destoyed

def init_var(init_var_snippet, param_space):
    """This helper function creates a VarInit object
    to easily initialise a variable using a snippet.

    Args:
    init_var_snippet --   type of the InitVarSnippet class as string or instance of class
                        derived from InitVarSnippet::Custom class.
    param_space --       dict with param values for the InitVarSnippet class
    """
    # Prepare snippet
    (s_instance, s_type, param_names, params) =\
        prepare_snippet(init_var_snippet, param_space,
                       genn_wrapper.InitVarSnippet)

    # Use add function to create suitable VarInit
    return VarInit(s_instance, params)

def init_connectivity(init_sparse_connectivity_snippet, param_space):
    """This helper function creates a InitSparseConnectivitySnippet::Init object
    to easily initialise connectivity using a snippet.

    Args:
    init_sparse_connectivity_snippet --    type of the InitSparseConnectivitySnippet class as string or instance of class
                                        derived from InitSparseConnectivitySnippet::Custom class.
    param_space --                       dict with param values for the InitSparseConnectivitySnippet class
    """
    # Prepare snippet
    (s_instance, s_type, param_names, params) =\
        prepare_snippet(init_sparse_connectivity_snippet, param_space,
                       genn_wrapper.InitSparseConnectivitySnippet)

    # Use add function to create suitable VarInit
    return Init(s_instance, params)

def create_custom_neuron_class(class_name, param_names=None, var_name_types=None,
                            derived_params=None, sim_code=None,
                            threshold_condition_code=None, reset_code=None,
                            support_code=None, extra_global_params=None,
                            additional_input_vars=None, custom_body=None):

    """This helper function creates a custom NeuronModel class.

    sa create_custom_postsynaptic_class
    sa create_custom_weight_update_class
    sa create_custom_current_source_class
    sa create_custom_init_var_snippet_class
    sa create_custom_sparse_connect_init_snippet_class

    Args:
    class_name     -- name of the new class

    Keyword args:
    param_names    -- list of strings with param names of the model
    var_name_types  -- list of pairs of strings with varible names and types of the model
    derived_params -- list of pairs, where the first member is string with name of
                        the derived parameter and the second MUST be an instance of the class
                        which inherits from libgenn.Snippet.DerivedParamFunc
    sim_code       -- string with the simulation code
    threshold_condition_code -- string with the threshold condition code
    reset_code     -- string with the reset code
    support_code   -- string with the support code
    extra_global_params -- list of pairs of strings with names and types of additional parameters
    additional_input_vars -- list of tuples with names and types as strings and
                            initial values of additional local input variables

    custom_body   -- dictionary with additional attributes and methods of the new class
    """
    if not isinstance(custom_body, dict) and custom_body is not None:
        raise ValueError("custom_body must be an isinstance of dict or None")

    body = {}

    if sim_code is not None:
        body["get_sim_code"] = lambda self: sim_code

    if threshold_condition_code is not None:
        body["get_threshold_condition_code"] = lambda self: threshold_condition_code

    if reset_code is not None:
        body["get_reset_code"] = lambda self: reset_code

    if support_code is not None:
        body["get_support_code"] = lambda self: support_code

    if extra_global_params is not None:
        body["get_extra_global_params"] =\
            lambda self: StringPairVector([StringPair(egp[0], egp[1])
                                           for egp in extra_global_params])
 
    if additional_input_vars:
        body["get_additional_input_vars"] =\
            lambda self: StringStringDoublePairPairVector(
                [StringStringDoublePairPair(aiv[0], StringDoublePair(aiv[1], aiv[2]))
                 for aiv in additional_input_vars])

    if custom_body is not None:
        body.update(custom_body)

    return create_custom_model_class(class_name, genn_wrapper.NeuronModels.Custom,
                                  param_names, var_name_types, derived_params,
                                  body)

def create_custom_postsynaptic_class(class_name, param_names=None, var_name_types=None,
                                  derived_params=None, decay_code=None,
                                  apply_input_code=None, support_code=None,
                                  custom_body=None):
    """This helper function creates a custom PostsynapticModel class.

    sa create_custom_neuron_class
    sa create_custom_weight_update_class
    sa create_custom_current_source_class
    sa create_custom_init_var_snippet_class
    sa create_custom_sparse_connect_init_snippet_class

    Args:
    class_name      -- name of the new class

    Keyword args:
    param_names     -- list of strings with param names of the model
    var_name_types   -- list of pairs of strings with varible names and types of the model
    derived_params  -- list of pairs, where the first member is string with name of
                        the derived parameter and the second MUST be an instance of the class
                        which inherits from libgenn.Snippet.DerivedParamFunc
    decay_code      -- string with the decay code
    apply_input_code -- string with the apply input code
    support_code    -- string with the support code

    custom_body    -- dictionary with additional attributes and methods of the new class
    """
    if not isinstance(custom_body, dict) and custom_body is not None:
        raise ValueError()

    body = {}

    if decay_code is not None:
        body["get_decay_code"] = lambda self: decay_code

    if apply_input_code is not None:
        body["get_apply_input_code"] = lambda self: apply_input_code

    if support_code is not None:
        body["get_support_code"] = lambda self: support_code

    if custom_body is not None:
        body.update(custom_body)

    return create_custom_model_class(
            class_name,
            genn_wrapper.PostsynapticModels.Custom,
            param_names,
            var_name_types,
            derived_params,
            body )


def create_custom_weight_update_class(class_name, param_names=None, var_name_types=None,
                                  pre_var_name_types=None, post_var_name_types=None,
                                  derived_params=None, sim_code=None,
                                  event_code=None, learn_post_code=None,
                                  synapse_dynamics_code=None,
                                  event_threshold_condition_code=None,
                                  pre_spike_code=None, post_spike_code=None,
                                  sim_support_code=None, learn_post_support_code=None,
                                  synapse_dynamics_suppport_code=None,
                                  extra_global_params=None,
                                  is_pre_spike_time_required=None,
                                  is_post_spike_time_required=None, custom_body=None):
    """This helper function creates a custom WeightUpdateModel class.

    sa create_custom_neuron_class
    sa create_custom_postsynaptic_class
    sa create_custom_current_source_class
    sa create_custom_init_var_snippet_class
    sa create_custom_sparse_connect_init_snippet_class

    Args:
    class_name           -- name of the new class

    Keyword args:
    param_names          -- list of strings with param names of the model
    var_name_types        -- list of pairs of strings with variable names and types of the model
    pre_var_name_types     -- list of pairs of strings with presynaptic variable names and types of the model
    post_var_name_types    -- list of pairs of strings with postsynaptic variable names and types of the model
    derived_params       -- list of pairs, where the first member is string with name of
                           the derived parameter and the second MUST be an instance of the class
                           which inherits from libgenn.Snippet.DerivedParamFunc
    sim_code             -- string with the simulation code
    event_code           -- string with the event code
    learn_post_code       -- string with the code to include in learn_synapse_post kernel/function
    synapse_dynamics_code -- string with the synapse dynamics code
    event_threshold_condition_code -- string with the event threshold condition code
    pre_spike_code                -- string with the code run once per spiking presynaptic neuron
    post_spike_code               -- string with the code run once per spiking postsynaptic neuron
    sim_support_code -- string with simulation support code
    learn_post_support_code -- string with support code for learn_synapse_post kernel/function
    synapse_dynamics_suppport_code -- string with synapse dynamics support code
    extra_global_params -- list of pairs of strings with names and types of additional parameters
    is_pre_spike_time_required -- boolean, is presynaptic spike time required?
    is_post_spike_time_required -- boolean, is postsynaptic spike time required?

    custom_body   -- dictionary with additional attributes and methods of the new class
    """
    if not isinstance(custom_body, dict) and custom_body is not None:
        raise ValueError("custom_body must be an instance of dict or None")

    body = {}

    if sim_code is not None:
        body["get_sim_code"] = lambda self: sim_code

    if event_code is not None:
        body["get_event_code"] = lambda self: event_code

    if learn_post_code is not None:
        body["get_learn_post_code"] = lambda self: learn_post_code

    if synapse_dynamics_code is not None:
        body["get_synapse_dynamics_code"] = lambda self: synapse_dynamics_code

    if event_threshold_condition_code is not None:
        body["get_event_threshold_condition_code"] = lambda self: event_threshold_condition_code

    if pre_spike_code is not None:
        body["get_pre_spike_code"] = lambda self: pre_spike_code

    if post_spike_code is not None:
        body["get_post_spike_code"] = lambda self: post_spike_code

    if sim_support_code is not None:
        body["get_sim_support_code"] = lambda self: sim_support_code

    if learn_post_support_code is not None:
        body["get_learn_post_support_code"] = lambda self: learn_post_support_code

    if synapse_dynamics_suppport_code is not None:
        body["get_synapse_dynamics_suppport_code"] = lambda self: synapse_dynamics_suppport_code

    if extra_global_params is not None:
        body["get_extra_global_params"] = lambda self: StringPairVector([StringPair(egp[0], egp[1])
                                                                      for egp in extra_global_params])

    if pre_var_name_types is not None:
        body["get_pre_vars"] = lambda self: StringPairVector([StringPair(vn[0], vn[1])
                                                            for vn in pre_var_name_types] )
    
    if post_var_name_types is not None:
        body["get_post_vars"] = lambda self: StringPairVector([StringPair(vn[0], vn[1])
                                                             for vn in post_var_name_types] )
    
    if is_pre_spike_time_required is not None:
        body["is_pre_spike_time_required"] = lambda self: is_pre_spike_time_required

    if is_post_spike_time_required is not None:
        body["is_post_spike_time_required"] = lambda self: is_post_spike_time_required
    
    if custom_body is not None:
        body.update(custom_body)
    
    return create_custom_model_class(class_name, genn_wrapper.WeightUpdateModels.Custom,
                                  param_names, var_name_types, derived_params, body)


def create_custom_current_source_class(class_name, param_names=None, var_name_types=None,
                                   derived_params=None, injection_code=None,
                                   extra_global_params=None, custom_body=None):
    """This helper function creates a custom NeuronModel class.

    sa create_custom_neuron_class
    sa create_custom_weight_update_class
    sa create_custom_current_source_class
    sa create_custom_init_var_snippet_class
    sa create_custom_sparse_connect_init_snippet_class
    
    Args:
    class_name     -- name of the new class

    Keyword args:
    param_names    -- list of strings with param names of the model
    var_name_types  -- list of pairs of strings with varible names and types of the model
    derived_params -- list of pairs, where the first member is string with name of
                        the derived parameter and the second MUST be an instance of the class
                        which inherits from libgenn.Snippet.DerivedParamFunc
    injection_code -- string with the current injection code
    extra_global_params -- list of pairs of strings with names and types of additional parameters

    custom_body   -- dictionary with additional attributes and methods of the new class
    """
    if not isinstance(custom_body, dict) and custom_body is not None:
        raise ValueError("custom_body must be an instance of dict or None")

    body = {}

    if injection_code is not None:
        body["get_injection_code"] = lambda self: injection_code

    if extra_global_params is not None:
        body["get_extra_global_params"] =\
            lambda self: StringPairVector([StringPair(egp[0], egp[1])
                                           for egp in extra_global_params])

    if custom_body is not None:
        body.update(custom_body)

    return create_custom_model_class(class_name, genn_wrapper.CurrentSourceModels.Custom,
                                  param_names, var_name_types, derived_params, body)

def create_custom_model_class(class_name, base, param_names, var_name_types,
                           derived_params, custom_body):
    """This helper function completes a custom model class creation.

    This part is common for all model classes and is nearly useless on its own
    unless you specify custom_body.
    sa create_custom_neuron_class
    sa create_custom_weight_update_class
    sa create_custom_postsynaptic_class
    sa create_custom_current_source_class
    sa create_custom_init_var_snippet_class
    sa create_custom_sparse_connect_init_snippet_class
    
    Args:
    class_name     -- name of the new class
    base          -- base class
    param_names    -- list of strings with param names of the model
    var_name_types  -- list of pairs of strings with varible names and types of the model
    derived_params -- list of pairs, where the first member is string with name of
                        the derived parameter and the second MUST be an instance of the class
                        which inherits from libgenn.Snippet.DerivedParamFunc
    custom_body   -- dictionary with attributes and methods of the new class
    """
    def ctor(self):
        base.__init__(self)

    body = {
            "__init__" : ctor,
    }
    
    if param_names is not None:
        body["get_param_names"] = lambda self: StringVector(param_names)

    if var_name_types is not None:
        body["get_vars"] =\
            lambda self: StringPairVector([StringPair(vn[0], vn[1])
                                           for vn in var_name_types])

    if derived_params is not None:
        body["get_derived_params"] =\
            lambda self: StringDPFPairVector([StringDPFPair(dp[0], make_dpf(dp[1]))
                                              for dp in derived_params])

    if custom_body is not None:
        body.update(custom_body)

    return type(class_name, (base,), body)


def create_dpf_class(dp_func):

    """Helper function to create derived parameter function class

    Args:
    dp_func -- a function which computes the derived parameter and takes
                two args "pars" (vector of double) and "dt" (double)
    """

    def ctor(self):
        genn_wrapper.Snippet.DerivedParamFunc.__init__(self)

    def call(self, pars, dt):
        return dp_func(pars, dt)

    return type("", (genn_wrapper.Snippet.DerivedParamFunc,),
                {"__init__" : ctor, "__call__" : call})


def create_cmlf_Class(cml_func):

    """Helper function to create function class for calculating sizes of
    matrices initialised with sparse connectivity initialisation snippet

    Args:
    cml_func -- a function which computes the length and takes
                three args "num_pre" (unsigned int), "num_post" (unsigned int)
                and "pars" (vector of double)
    """

    def ctor(self):
        genn_wrapper.InitSparseConnectivitySnippet.CalcMaxLengthFunc.__init__(self)

    def call(self, num_pre, num_post, pars):
        return cml_func(numPre, num_post, pars, dt)

    return type("", (genn_wrapper.InitSparseConnectivitySnippet.CalcMaxLengthFunc,),
                {"__init__" : ctor, "__call__" : call})

def create_custom_init_var_snippet_class(class_name, param_names=None, derived_params=None,
                                    var_init_code=None, custom_body=None):
    """This helper function creates a custom InitVarSnippet class.

    sa create_custom_neuron_class
    sa create_custom_weight_update_class
    sa create_custom_postsynaptic_class
    sa create_custom_current_source_class
    sa create_custom_sparse_connect_init_snippet_class
    
    Args:
    class_name     -- name of the new class

    Keyword args:
    param_names    -- list of strings with param names of the model
    derived_params -- list of pairs, where the first member is string with name of
                        the derived parameter and the second MUST be an instance of the class
                        which inherits from libgenn.Snippet.DerivedParamFunc
    var_initcode       -- string with the variable initialization code
    custom_body       -- dictionary with additional attributes and methods of the new class
    """

    if not isinstance(custom_body, dict) and custom_body is not None:
        raise ValueError("custom_body must be an instance of dict or None")

    body = {}

    if get_code is not None:
        body["get_code"] = lambda self: var_init_code

    if custom_body is not None:
        body.update(custom_body)

    return create_custom_model_class(class_name, genn_wrapper.InitVarSnippet.Custom,
                                  param_names, None, derived_params, body)

def create_custom_sparse_connect_init_snippet_class(class_name, param_names=None,
                                              derived_params=None, row_build_code=None,
                                              row_build_state_vars=None,
                                              calc_max_row_length_func=None,
                                              calc_max_col_length_func=None,
                                              extra_global_params=None, custom_body=None):
    """This helper function creates a custom InitSparseConnectivitySnippet class.

    sa create_custom_neuron_class
    sa create_custom_weight_update_class
    sa create_custom_postsynaptic_class
    sa create_custom_current_source_class
    sa create_custom_init_var_snippet_class

    Args:
    class_name     -- name of the new class

    Keyword args:
    param_names    -- list of strings with param names of the model
    derived_params -- list of pairs, where the first member is string with name of
                        the derived parameter and the second MUST be an instance of the class
                        which inherits from libgenn.Snippet.DerivedParamFunc
    row_build_code      -- string with the row building initialization code
    row_build_state_vars -- list of tuples of state variables, their types and
                         their initial values to use across row building loop
    extra_global_params -- list of pairs of strings with names and types of additional parameters

    custom_body       -- dictionary with additional attributes and methods of the new class
    """

    if not isinstance(custom_body, dict) and custom_body is not None:
        raise ValueError("custom_body must be an instance of dict or None")

    body = {}

    if row_build_code is not None:
        body["get_row_build_code"] = lambda self: row_build_code

    if row_build_state_vars is not None:
        body["get_row_build_state_vars"] =\
            lambda self: StringStringDoublePairPairVector(
                [StringStringDoublePairPair(r[0], StringDoublePair(r[1], r[2]))
                 for r in row_build_state_vars] )

    if calc_max_row_length_func is not None:
        body["get_calc_max_row_length_func"] = lambda self: make_cmlf(calc_max_row_length_func)

    if calc_max_col_length_func is not None:
        body["get_calc_max_col_length_func"] = lambda self: make_cmlf(calc_max_col_length_func)

    if extra_global_params is not None:
        body["get_extra_global_params"] =\
            lambda self: StringPairVector([StringStringDoublePairPair(egp[0], egp[1])
                                           for egp in extra_global_params])

    if custom_body is not None:
        body.update(custom_body)

    return create_custom_model_class(class_name, genn_wrapper.InitVarSnippet.Custom,
                                  param_names, None, derived_params, body)
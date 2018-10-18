"""GeNNGroups
This module provides classes which automatize model checks and parameter
convesions for GeNN Groups
"""

from six import iteritems
import genn_wrapper
import model_preprocessor
from model_preprocessor import Variable
from genn_wrapper import VarMode_LOC_HOST_DEVICE_INIT_HOST
from genn_wrapper import (SynapseMatrixConnectivity_SPARSE,
                          SynapseMatrixConnectivity_YALE,
                          SynapseMatrixConnectivity_RAGGED,
                          SynapseMatrixConnectivity_BITMASK,
                          SynapseMatrixConnectivity_DENSE,
                          SynapseMatrixWeight_GLOBAL)


class Group(object):

    """Parent class of NeuronGroup, SynapseGroup and CurrentSource"""

    def __init__(self, name):
        """Init Group

        Args:
        name    --  string name of the Group
        """
        self.name = name
        self.vars = {}
        self.extra_global_params = {}

    def set_var(self, var_name, values):
        """Set values for a Variable

        Args:
        var_name    --  string with the name of the variable
        values      --  iterable or a single value
        """
        self.vars[var_name].set_values(values)

    def _add_extra_global_param(self, param_name, param_values,
                                model, auto_alloc=True):
        """Add extra global parameter

        Args:
        param_name      --  string with the name of the extra global parameter
        param_values    --  iterable or a single value
        model           --  instance of the model
        auto_alloc      --  boolean whether the extra global parameter
                            should be allocated. Defaults to true.
        """
        pnt = list(model.get_extra_global_params())
        param_type = None
        for pn, pt in pnt:
            if pn == param_name:
                param_type = pt
                break

        egp = Variable(param_name, param_type, param_values)
        egp.needs_allocation = auto_alloc

        self.extra_global_params[param_name] = egp


class NeuronGroup(Group):

    """Class representing a group of neurons"""

    def __init__(self, name):
        """Init NeuronGroup

        Args:
        name    --  string name of the group
        """
        super(NeuronGroup, self).__init__(name)
        self.neuron = None
        self.spikes = None
        self.spike_count = None
        self.spike_que_ptr = [0]
        self.is_spike_source_array = False
        self._max_delay_steps = 0

    @property
    def current_spikes(self):
        """Current spikes from GeNN"""
        offset = self.spike_que_ptr[0] * self.size
        return self.spikes[
            offset:offset + self.spike_count[self.spike_que_ptr[0]]]

    @property
    def delay_slots(self):
        """Maximum delay steps needed for this group"""
        return self.pop.get_num_delay_slots()

    @property
    def size(self):
        return self.pop.get_num_neurons()

    def set_neuron(self, model, param_space, var_space):
        """Set neuron, its parameters and initial variables

        Args:
        model       --  type as string of intance of the model
        param_space --  dict with model parameters
        var_space   --  dict with model variables
        """
        (self.neuron, self.type, self.param_names, self.params,
         self.var_names, self.vars) = model_preprocessor.prepare_model(
             model, param_space, var_space,
             model_family=genn_wrapper.NeuronModels)

        if self.type == "SpikeSourceArray":
            self.is_spike_source_array = True

    def add_to(self, nn_model, num_neurons):
        """Add this NeuronGroup to the GeNN NNmodel

        Args:
        nn_model    --  GeNN NNmodel
        num_neurons --  int number of neurons
        """
        add_fct = getattr(nn_model, "add_neuron_population_" + self.type)

        var_ini = model_preprocessor.var_space_to_vals(self.neuron, self.vars)
        self.pop = add_fct(self.name, num_neurons, self.neuron,
                           self.params, var_ini)

        for var_name, var in iteritems(self.vars):
            if var.init_required:
                self.pop.set_var_mode(var_name,
                                      VarMode_LOC_HOST_DEVICE_INIT_HOST)

    def add_extra_global_param(self, param_name, param_values):
        """Add extra global parameter

        Args:
        param_name      --  string with the name of the extra global parameter
        param_values    --  iterable or a single value
        """
        self._add_extra_global_param(param_name, param_values, self.neuron)


class SynapseGroup(Group):

    """Class representing synaptic connection between two groups of neurons"""

    def __init__(self, name):
        """Init SynapseGroup

        Args:
        name    --  string name of the group
        """
        self.connections_set = False
        super(SynapseGroup, self).__init__(name)
        self.w_update = None
        self.postsyn = None
        self.src = None
        self.trg = None
        self.pre_vars = {}
        self.post_vars = {}
        self.connectivity_initialiser = None

    @property
    def size(self):
        """Size of connection matrix"""
        if self.is_dense:
            return self.trg.size * self.src.size
        elif self.is_yale:
            return self._num_connections
        elif self.is_ragged:
            return self.max_connections * self.src.size
        #elif self.is_ragged

    @property
    def max_connections(self):
        return self.pop.get_max_connections()

    def set_pre_var(self, var_name, values):
        """Set values for a presynaptic variable

        Args:
        var_name    --  string with the name of the presynaptic variable
        values      --  iterable or a single value
        """
        self.pre_vars[var_name].set_values(values)

    def set_post_var(self, var_name, values):
        """Set values for a postsynaptic variable

        Args:
        var_name    --  string with the name of the presynaptic variable
        values      --  iterable or a single value
        """
        self.post_vars[var_name].set_values(values)

    def set_weight_update(self, model, param_space,
                          var_space, pre_var_space, post_var_space):
        """Set weight update model, its parameters and initial variables

        Args:
        model           --  type as string of intance of the model
        param_space     --  dict with model parameters
        var_space       --  dict with model variables
        pre_var_space   --  dict with model presynaptic variables
        post_var_space  --  dict with model postsynaptic variables
        """
        (self.w_update, self.wu_type, self.wu_param_names, self.wu_params,
         self.wu_var_names, var_dict, self.wu_pre_var_names, pre_var_dict,
         self.wu_post_var_names, post_var_dict) =\
             model_preprocessor.prepare_model(
                 model, param_space, var_space, pre_var_space,
                 post_var_space, model_family=genn_wrapper.WeightUpdateModels)

        self.vars.update(var_dict)
        self.pre_vars.update(pre_var_dict)
        self.post_vars.update(post_var_dict)

    def set_post_syn(self, model, param_space, var_space):
        """Set postsynaptic model, its parameters and initial variables

        Args:
        model       --  type as string of intance of the model
        param_space --  dict with model parameters
        var_space   --  dict with model variables
        """
        (self.postsyn, self.ps_type, self.ps_param_names, self.ps_params,
         self.ps_var_names, var_dict) = model_preprocessor.prepare_model(
             model, param_space, var_space,
             model_family=genn_wrapper.PostsynapticModels)

        self.vars.update(var_dict)

    @property
    def is_connectivity_init_required(self):
        return self.connectivity_initialiser is None

    @property
    def matrix_type(self):
        """Type of the projection matrix"""
        return self._matrix_type

    @matrix_type.setter
    def matrix_type(self, matrix_type):
        self._matrix_type = getattr(genn_wrapper,
                                    "SynapseMatrixType_" + matrix_type)

    @property
    def is_yale(self):
        """Tests whether synaptic connectivity uses Yale format"""
        return (self._matrix_type & SynapseMatrixConnectivity_YALE) != 0

    @property
    def is_ragged(self):
        """Tests whether synaptic connectivity uses Ragged format"""
        return (self._matrix_type & SynapseMatrixConnectivity_RAGGED) != 0

    @property
    def is_bitmask(self):
        """Tests whether synaptic connectivity uses Bitmask format"""
        return (self._matrix_type & SynapseMatrixConnectivity_BITMASK) != 0

    @property
    def is_dense(self):
        """Tests whether synaptic connectivity uses dense format"""
        return (self._matrix_type & SynapseMatrixConnectivity_DENSE) != 0

    @property
    def global_weights(self):
        """Tests whether synaptic connectivity has global weights"""
        return (self._matrix_type & SynapseMatrixWeight_GLOBAL) != 0

    def set_connections(self, conns, g):
        """Set connections between two groups of neurons

        Args:
        conns   --  connections as tuples (pre, post)
        g       --  strength of the connection
        """
        if (self.is_yale) != 0:
            conns.sort()
            self._num_connections = len(conns)
            self.ind = [post for (_, post) in conns]
            self.indInG = []
            self.indInG.append(0)
            cur_pre = 0
            # convert connection tuples to indInG
            for i, (pre, _) in enumerate(conns):
                while pre != cur_pre:
                    self.indInG.append(i)
                    cur_pre += 1
            # if there are any "hanging" presynaptic neurons without
            # connections, they should all point to the end of indInG
            while len(self.indInG) < self.src.size + 1:
                self.indInG.append(len(conns))

            # compute max number of connections from taget neuron to source
            max_conn = int(max([self.indInG[i] - self.indInG[i - 1]
                                for i in range(len(self.indInG)) if i != 0]))
            self.pop.set_max_connections(max_conn)
        elif (self.is_dense) != 0:
            self.g_mask = [pre * self.trg.size + post
                           for (pre, post) in conns]
        else:
            raise Exception("Setting connections with type '{0}' is not "
                            "currently supported".format(self._matrix_type))

        if not self.global_weights:
            self.vars["g"].set_values(g)
            self.pop.set_wuvar_mode("g", VarMode_LOC_HOST_DEVICE_INIT_HOST)

        self.connections_set = True

    def set_connected_populations(self, source, target):
        """Set two groups of neurons connected by this SynapseGroup

        Args:
        source   -- string name of the presynaptic neuron group
        target   -- string name of the postsynaptic neuron group
        """
        self.src = source
        self.trg = target

    def add_to(self, nn_model, delay_steps):
        """Add this SynapseGroup to the GeNN NNmodel

        Args:
        nn_model -- GeNN NNmodel
        """
        add_fct = getattr(
            nn_model,
            ("add_synapse_population_" + self.wu_type + "_" + self.ps_type))

        wu_var_ini = model_preprocessor.var_space_to_vals(
            self.w_update, {vn: self.vars[vn]
                            for vn in self.wu_var_names})

        wu_pre_var_ini = model_preprocessor.pre_var_space_to_vals(
            self.w_update, {vn: self.pre_vars[vn]
                            for vn in self.wu_pre_var_names})

        wu_post_var_ini = model_preprocessor.post_var_space_to_vals(
            self.w_update, {vn: self.post_vars[vn]
                            for vn in self.wu_post_var_names})

        ps_var_ini = model_preprocessor.var_space_to_vals(
            self.postsyn, {vn: self.vars[vn]
                           for vn in self.ps_var_names})

        # Use unitialised connectivity initialiser if none has been set
        connect_init = (genn_wrapper.uninitialised_connectivity()
                        if self.connectivity_initialiser is None
                        else self.connectivity_initialiser)
        self.pop = add_fct(self.name, self.matrix_type, delay_steps,
                           self.src.name, self.trg.name, self.w_update,
                           self.wu_params, wu_var_ini, wu_pre_var_ini,
                           wu_post_var_ini, self.postsyn, self.ps_params,
                           ps_var_ini, connect_init)

        for var_name, var in iteritems(self.vars):
            if var.init_required:
                if var_name in self.wu_var_names:
                    self.pop.set_wuvar_mode(
                        var_name, VarMode_LOC_HOST_DEVICE_INIT_HOST)
                if var_name in self.wu_pre_var_names:
                    self.pop.set_wupre_var_mode(
                        var_name, VarMode_LOC_HOST_DEVICE_INIT_HOST)
                if var_name in self.wu_post_var_names:
                    self.pop.set_wupost_var_mode(
                        var_name, VarMode_LOC_HOST_DEVICE_INIT_HOST)
                if var_name in self.ps_var_names:
                    self.pop.set_psvar_mode(
                        var_name, VarMode_LOC_HOST_DEVICE_INIT_HOST)

    def add_extra_global_param(self, param_name, param_values):
        """Add extra global parameter

        Args:
        param_name   -- string with the name of the extra global parameter
        param_values -- iterable or a single value
        """
        self._add_extra_global_param(param_name, param_values, self.w_update)


class CurrentSource(Group):

    """Class representing a current injection into a group of neurons"""

    def __init__(self, name):
        """Init CurrentSource

        Args:
        name -- string name of the current source
        """
        super(CurrentSource, self).__init__(name)
        self.current_source_model = None
        self.target_pop = None

    @property
    def size(self):
        """Number of neuron in the injected population"""
        return self.target_pop.size

    @size.setter
    def size(self, _):
        pass

    def set_current_source_model(self, model, param_space, var_space):
        """Set curront source model, its parameters and initial variables

        Args:
        model       --  type as string of intance of the model
        param_space --  dict with model parameters
        var_space   --  dict with model variables
        """
        (self.current_source_model, self.type, self.param_names, self.params,
         self.var_names, self.vars) = model_preprocessor.prepare_model(
             model, param_space, var_space,
             model_family=genn_wrapper.CurrentSourceModels)

    def add_to(self, nn_model, pop):
        """Inject this CurrentSource into population and
        add it to the GeNN NNmodel

        Args:
        pop         --  instance of NeuronGroup into which this CurrentSource
                        should be injected
        nn_model    --  GeNN NNmodel
        """
        add_fct = getattr(nn_model, "add_current_source_" + self.type)
        self.target_pop = pop

        var_ini = model_preprocessor.varSpaceToVarValues(
            self.current_source_model, self.vars)
        self.pop = add_fct(self.name, self.current_source_model, pop.name,
                           self.params, var_ini)

        for var_name, var in iteritems(self.vars):
            if var.init_required:
                self.pop.set_var_mode(var_name,
                                      VarMode_LOC_HOST_DEVICE_INIT_HOST)

    def add_extra_global_param(self, param_name, param_values):
        """Add extra global parameter

        Args:
        param_name   -- string with the name of the extra global parameter
        param_values -- iterable or a single value
        """
        self._add_extra_global_param(param_name, param_values,
                                     self.current_source_model)

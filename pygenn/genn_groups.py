"""GeNNGroups
This module provides classes which automatize model checks and parameter
convesions for GeNN Groups
"""
try:
    xrange
except NameError:  # Python 3
    xrange = range

from weakref import proxy
from deprecated import deprecated
from six import iteritems
import numpy as np
from . import genn_wrapper
from . import model_preprocessor
from .model_preprocessor import ExtraGlobalVariable, Variable, genn_types
from .genn_wrapper import (SynapseMatrixConnectivity_SPARSE,
                           SynapseMatrixConnectivity_BITMASK,
                           SynapseMatrixConnectivity_DENSE,
                           SynapseMatrixWeight_INDIVIDUAL,
                           SynapseMatrixWeight_INDIVIDUAL_PSM,
                           VarLocation_HOST,
                           SynapseMatrixConnectivity_PROCEDURAL)


class Group(object):

    """Parent class of NeuronGroup, SynapseGroup and CurrentSource"""

    def __init__(self, name, model):
        """Init Group

        Args:
        name    --  string name of the Group
        """
        self.name = name
        self._model = proxy(model)
        self.vars = {}
        self.extra_global_params = {}

    def set_var(self, var_name, values):
        """Set values for a Variable

        Args:
        var_name    --  string with the name of the variable
        values      --  iterable or a single value
        """
        self.vars[var_name].set_values(values)

    def pull_state_from_device(self):
        """Wrapper around GeNNModel.pull_state_from_device"""
        self._model.pull_state_from_device(self.name)

    def pull_var_from_device(self, var_name):
        """Wrapper around GeNNModel.pull_var_from_device

        Args:
        var_name    --  string with the name of the variable
        """
        self._model.pull_var_from_device(self.name, var_name)

    def pull_extra_global_param_from_device(self, egp_name, size=1):
        """Wrapper around GeNNModel.pull_extra_global_param_from_device

        Args:
        var_name    --  string with the name of the variable
        size        --  number of entries in EGP array
        """
        self._model.pull_extra_global_param_from_device(self.name, egp_name, size)

    def push_state_to_device(self):
        """Wrapper around GeNNModel.push_state_to_device"""
        self._model.push_state_to_device(self.name)

    def push_var_to_device(self, var_name):
        """Wrapper around GeNNModel.push_var_to_device

        Args:
        var_name    --  string with the name of the variable
        """
        self._model.push_var_to_device(self.name, var_name)

    def push_extra_global_param_to_device(self, egp_name, size=1):
        """Wrapper around GeNNModel.push_extra_global_param_to_device

        Args:
        var_name    --  string with the name of the variable
        size        --  number of entries in EGP array
        """
        self._model.push_extra_global_param_to_device(self.name, egp_name, size)

    def _set_extra_global_param(self, param_name, param_values, model, egp_dict=None):
        """Set extra global parameter

        Args:
        param_name      --  string with the name of the extra global parameter
        param_values    --  iterable
        model           --  instance of the model
        """
        # If no EGP dictionary is specified, use standard one
        if egp_dict is None:
            egp_dict = self.extra_global_params

        param_type = None
        for p in model.get_extra_global_params():
            if p.name == param_name:
                param_type = p.type
                break

        assert param_type is not None
        egp_dict[param_name] = ExtraGlobalVariable(param_name, param_type,
                                                   param_values)

    def _assign_ext_ptr_array(self, var_name, var_size, var_type):
        """Assign a variable to an external numpy array

        Args:
        var_name    --  string a fully qualified name of the variable to assign
        var_size    --  int the size of the variable
        var_type    --  string type of the variable. The supported types are
                        char, unsigned char, short, unsigned short, int,
                        unsigned int, long, unsigned long, long long,
                        unsigned long long, float, double, long double
                        and scalar.

        Returns numpy array of type var_type

        Raises ValueError if variable type is not supported
        """

        internal_var_name = var_name + self.name

        if var_type == "scalar":
            var_type = self._model._scalar

        return genn_types[var_type].assign_ext_ptr_array(self._model._slm,
                                                         internal_var_name,
                                                         var_size)

    def _assign_ext_ptr_single(self, var_name, var_type):
        """Assign a variable to an external scalar value containing one element

        Args:
        var_name    --  string a fully qualified name of the variable to assign
        var_type    --  string type of the variable. The supported types are
                        char, unsigned char, short, unsigned short, int,
                        unsigned int, long, unsigned long, long long,
                        unsigned long long, float, double, long double
                        and scalar.

        Returns numpy array of type var_type

        Raises ValueError if variable type is not supported
        """

        internal_var_name = var_name + self.name

        if var_type == "scalar":
            var_type = self._model._scalar

        return genn_types[var_type].assign_ext_ptr_single(self._model._slm,
                                                          internal_var_name)

    def _load_vars(self, size=None, var_dict=None, get_location_fn=None):
        # If no size is specified, use standard size
        if size is None:
            size = self.size

        # If no variable dictionary is specified, use standard one
        if var_dict is None:
            var_dict = self.vars

        # If no location getter function is specified, use standard one
        if get_location_fn is None:
            get_location_fn = self.pop.get_var_location

        # Loop through variables
        for var_name, var_data in iteritems(var_dict):
            # If variable is located on host
            var_loc = get_location_fn(var_name) 
            if (var_loc & VarLocation_HOST) != 0:
                # Get view
                var_data.view = self._assign_ext_ptr_array(var_name, size,
                                                           var_data.type)

                # If manual initialisation is required, copy over variables
                if var_data.init_required:
                    var_data.view[:] = var_data.values
            else:
                assert not var_data.init_required
                var_data.view = None

    def _reinitialise_vars(self, size=None, var_dict=None):
        # If no size is specified, use standard size
        if size is None:
            size = self.size

        # If no variable dictionary is specified, use standard one
        if var_dict is None:
            var_dict = self.vars

        # Loop through variables
        for var_name, var_data in iteritems(var_dict):
            # If manual initialisation is required, copy over variables
            if var_data.init_required:
                var_data.view[:] = var_data.values

    def _load_egp(self, egp_dict=None, egp_suffix=""):
        # If no EGP dictionary is specified, use standard one
        if egp_dict is None:
            egp_dict = self.extra_global_params

        # Loop through extra global params
        for egp_name, egp_data in iteritems(egp_dict):
            if egp_data.is_scalar:
                # Assign view
                egp_data.view = self._assign_ext_ptr_single(egp_name + egp_suffix,
                                                            egp_data.type)
                # Copy values
                egp_data.view[:] = egp_data.values
            else:
                # Allocate memory
                self._model._slm.allocate_extra_global_param(
                    self.name, egp_name + egp_suffix, len(egp_data.values))

                # Assign view
                egp_data.view = self._assign_ext_ptr_array(egp_name + egp_suffix,
                                                           len(egp_data.values), 
                                                           egp_data.type)

                # Copy values
                egp_data.view[:] = egp_data.values

                # Push egp_data
                self._model._slm.push_extra_global_param(
                    self.name, egp_name + egp_suffix, len(egp_data.values))
                    
    def _load_var_init_egps(self, var_dict=None):
        # If no variable dictionary is specified, use standard one
        if var_dict is None:
            var_dict = self.vars
        
        # Loop through variables and load any associated initialisation egps
        for var_name, var_data in iteritems(var_dict):
            self._load_egp(var_data.extra_global_params, var_name)


class NeuronGroup(Group):

    """Class representing a group of neurons"""

    def __init__(self, name, model):
        """Init NeuronGroup

        Args:
        name    --  string name of the group
        """
        super(NeuronGroup, self).__init__(name, model)
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
             model, self, param_space, var_space, self,
             model_family=genn_wrapper.NeuronModels)

        if self.type == "SpikeSourceArray":
            self.is_spike_source_array = True

    def add_to(self, model_spec, num_neurons):
        """Add this NeuronGroup to a model

        Args:
        model_spec  --  ``pygenn.genn_model.GeNNModel`` to add to
        num_neurons --  int number of neurons
        """
        add_fct = getattr(model_spec, "add_neuron_population_" + self.type)

        var_ini = model_preprocessor.var_space_to_vals(self.neuron, self.vars)
        self.pop = add_fct(self.name, num_neurons, self.neuron,
                           self.params, var_ini)

    @deprecated("This function was poorly named, use 'set_extra_global_param' instead")
    def add_extra_global_param(self, param_name, param_values):
        """Set extra global parameter

        Args:
        param_name      --  string with the name of the extra global parameter
        param_values    --  iterable or a single value
        """
        self.set_extra_global_param(param_name, param_values)

    def set_extra_global_param(self, param_name, param_values):
        """Set extra global parameter

        Args:
        param_name      --  string with the name of the extra global parameter
        param_values    --  iterable or a single value
        """
        self._set_extra_global_param(param_name, param_values, self.neuron)

    def pull_spikes_from_device(self):
        """Wrapper around GeNNModel.pull_spikes_from_device"""
        self._model.pull_spikes_from_device(self.name)

    def pull_current_spikes_from_device(self):
        """Wrapper around GeNNModel.pull_current_spikes_from_device"""
        self._model.pull_current_spikes_from_device(self.name)

    def push_spikes_to_device(self):
        """Wrapper around GeNNModel.push_spikes_to_device"""
        self._model.push_spikes_to_device(self.name)

    def push_current_spikes_to_device(self):
        """Wrapper around GeNNModel.push_current_spikes_to_device"""
        self._model.push_current_spikes_to_device(self.name)

    def load(self):
        """Loads neuron group"""
        self.spikes = self._assign_ext_ptr_array("glbSpk", 
                                                 self.size * self.delay_slots,
                                                 "unsigned int")
        self.spike_count = self._assign_ext_ptr_array("glbSpkCnt", 
                                                      self.delay_slots, 
                                                      "unsigned int")
        if self.delay_slots > 1:
            self.spike_que_ptr = self._model._slm.assign_external_pointer_single_ui(
                "spkQuePtr" + self.name)

        # Load neuron state variables
        self._load_vars()

        # Load neuron extra global params
        self._load_egp()

    def load_init_egps(self):
        # Load any egps used for variable initialisation
        self._load_var_init_egps()

    def reinitialise(self):
        """Reinitialise neuron group"""

        # Reinitialise neuron state variables
        self._reinitialise_vars()

class SynapseGroup(Group):

    """Class representing synaptic connection between two groups of neurons"""

    def __init__(self, name, model, weight_sharing_master=None):
        """Init SynapseGroup

        Args:
        name    --  string name of the group
        """
        self.connections_set = False
        super(SynapseGroup, self).__init__(name, model)
        self.w_update = None
        self.postsyn = None
        self.src = None
        self.trg = None
        self.psm_vars = {}
        self.pre_vars = {}
        self.post_vars = {}
        self.psm_extra_global_params = {}
        self.connectivity_extra_global_params = {}
        self.connectivity_initialiser = None
        self.weight_sharing_master = weight_sharing_master

    @property
    def num_synapses(self):
        """Number of synapses in group"""
        if self.is_dense:
            return self.trg.size * self.src.size
        elif self.is_ragged:
            return self._num_synapses

    @property
    def weight_update_var_size(self):
        """Size of each weight update variable"""
        if self.is_dense:
            return self.trg.size * self.src.size
        elif self.is_ragged:
            return self.max_row_length * self.src.size

    @property
    def max_row_length(self):
        return self.pop.get_max_connections()

    def set_psm_var(self, var_name, values):
        """Set values for a postsynaptic model variable

        Args:
        var_name    --  string with the name of the
                        postsynaptic model variable
        values      --  iterable or a single value
        """
        self.psm_vars[var_name].set_values(values)

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
        if self.weight_sharing_master is not None:
            raise Exception("when weight sharing is used, set_weight_update"
                            "can only be used on the 'master' population")
        else:
            (self.w_update, self.wu_type, self.wu_param_names, self.wu_params,
             self.wu_var_names, var_dict, self.wu_pre_var_names, pre_var_dict,
             self.wu_post_var_names, post_var_dict) =\
                 model_preprocessor.prepare_model(
                     model, self, param_space, var_space, pre_var_space,
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
             model, self, param_space, var_space,
             model_family=genn_wrapper.PostsynapticModels)

        self.psm_vars.update(var_dict)

    def get_var_values(self, var_name):
        if self.weight_sharing_master is not None:
            raise Exception("when weight sharing is used, get_var_values"
                            "can only be used on the 'master' population")
        else:
            var_view = self.vars[var_name].view

            if self.is_dense:
                return np.copy(var_view)
            elif self.is_ragged:
                max_rl = self.max_row_length
                row_ls = self.row_lengths if self.connectivity_initialiser is None else self._row_lengths

                # Create range containing the index where each row starts in ind
                row_start_idx = xrange(0, self.weight_update_var_size, max_rl)

                # Build list of subviews representing each row
                rows = [var_view[i:i + r] for i, r in zip(row_start_idx, row_ls)]

                # Stack all rows together into single array
                return np.hstack(rows)
            else:
                raise Exception("Matrix format not supported")

    @property
    def is_connectivity_init_required(self):
        return (self.weight_sharing_master is None 
                and self.connectivity_initialiser is None)

    @property
    def matrix_type(self):
        """Type of the projection matrix"""
        if self.weight_sharing_master is None:
            return self._matrix_type
        else:
            return self.weight_sharing_master.matrix_type

    @matrix_type.setter
    def matrix_type(self, matrix_type):
        if self.weight_sharing_master is None:
            self._matrix_type = getattr(genn_wrapper,
                                        "SynapseMatrixType_" + matrix_type)
        else:
            raise Exception("when weight sharing is used, matrix_type"
                            "can only be set on the 'master' population")

    @property
    def has_procedural_connectivity(self):
        """Tests whether synaptic connectivity is procedural"""
        return (self.matrix_type & SynapseMatrixConnectivity_PROCEDURAL) != 0

    @property
    def has_procedural_weights(self):
        """Tests whether synaptic weights are procedural"""
        return (self.matrix_type & SynapseMatrixWeight_PROCEDURAL) != 0

    @property
    def is_ragged(self):
        """Tests whether synaptic connectivity uses Ragged format"""
        return (self.matrix_type & SynapseMatrixConnectivity_SPARSE) != 0

    @property
    def is_bitmask(self):
        """Tests whether synaptic connectivity uses Bitmask format"""
        return (self.matrix_type & SynapseMatrixConnectivity_BITMASK) != 0

    @property
    def is_dense(self):
        """Tests whether synaptic connectivity uses dense format"""
        return (self.matrix_type & SynapseMatrixConnectivity_DENSE) != 0

    @property
    def has_individual_synapse_vars(self):
        """Tests whether synaptic connectivity has individual weights"""
        return (self.weight_sharing_master is None 
                and (self.matrix_type & SynapseMatrixWeight_INDIVIDUAL) != 0)

    @property
    def has_individual_postsynaptic_vars(self):
        """Tests whether synaptic connectivity has
        individual postsynaptic model variables"""
        return (self.matrix_type & SynapseMatrixWeight_INDIVIDUAL_PSM) != 0

    def set_sparse_connections(self, pre_indices, post_indices):
        """Set ragged format connections between two groups of neurons

        Args:
        pre_indices     --  ndarray of presynaptic indices
        post_indices    --  ndarray of postsynaptic indices
        """
        if self.weight_sharing_master is not None:
            raise Exception("when weight sharing is used, set_sparse_connections"
                            "can only be used on the 'master' population")
        elif self.is_ragged:
            # Lexically sort indices
            self.synapse_order = np.lexsort((post_indices, pre_indices))

            # Count synapses
            self._num_synapses = len(post_indices)

            # Count the number of synapses in each row
            row_lengths = np.bincount(pre_indices, minlength=self.src.size)
            row_lengths = row_lengths.astype(np.uint32)

            # Use maximum for max connections
            max_row_length = int(np.amax(row_lengths))
            self.pop.set_max_connections(max_row_length)

            # Set ind to sorted postsynaptic indices
            self.ind = post_indices[self.synapse_order]

            # Cache the row lengths
            self.row_lengths = row_lengths

            assert len(self.row_lengths) == self.src.size
        else:
            raise Exception("set_sparse_connections only supports"
                            "ragged format sparse connectivity")

        self.connections_set = True

    def get_sparse_pre_inds(self):
        """Get presynaptic indices of synapse group connections

        Returns:
        ndarray of presynaptic indices
        """

        if self.weight_sharing_master is not None:
            raise Exception("when weight sharing is used, get_sparse_pre_inds"
                            "can only be used on the 'master' population")
        elif self.is_ragged:

            rl = self.row_lengths if self.connectivity_initialiser is None else self._row_lengths

            if rl is None:
                raise Exception("problem accessing connectivity ")

            # Expand row lengths into full array
            # of presynaptic indices and return
            return np.hstack([np.repeat(i, l) for i, l in enumerate(rl)])

        else:
            raise Exception("get_sparse_pre_inds only supports"
                            "ragged format sparse connectivity")

    def get_sparse_post_inds(self):
        """Get postsynaptic indices of synapse group connections

        Returns:
        ndarrays of postsynaptic indices
        """
        if self.weight_sharing_master is not None:
            raise Exception("when weight sharing is used, get_sparse_post_inds"
                            "can only be used on the 'master' population")
        elif self.is_ragged:
            if self.connectivity_initialiser is None:

                if self.ind is None or self.row_lengths is None:
                    raise Exception("problem accessing manually initialised connectivity ")
                # Return cached indices
                return self.ind

            else:
                if self._ind is None or self._row_lengths is None:
                    raise Exception("problem accessing on-device initialised connectivity ")

                # the _ind array view still has some non-valid data so we remove them
                # with the row_lengths
                return np.hstack([
                    self._ind[i * self.max_row_length: (i * self.max_row_length) + r]
                        for i, r in enumerate(self._row_lengths)])

        else:
            raise Exception("get_sparse_post_inds only supports"
                            "ragged format sparse connectivity")


    def set_connected_populations(self, source, target):
        """Set two groups of neurons connected by this SynapseGroup

        Args:
        source   -- string name of the presynaptic neuron group
        target   -- string name of the postsynaptic neuron group
        """
        self.src = source
        self.trg = target

    def add_to(self, model_spec, delay_steps):
        """Add this SynapseGroup to the a model

        Args:
        model_spec  -- ``pygenn.genn_model.GeNNModel`` to add to
        delay_steps -- number of axonal delay timesteps to simulate for this synapse group
        """
        ps_var_ini = model_preprocessor.var_space_to_vals(
                self.postsyn, {vn: self.psm_vars[vn]
                               for vn in self.ps_var_names})

        if self.weight_sharing_master is None:
            add_fct = getattr(
                model_spec,
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

            # Use unitialised connectivity initialiser if none has been set
            connect_init = (genn_wrapper.uninitialised_connectivity()
                            if self.connectivity_initialiser is None
                            else self.connectivity_initialiser)

            self.pop = add_fct(self.name, self.matrix_type, delay_steps,
                               self.src.name, self.trg.name, self.w_update,
                               self.wu_params, wu_var_ini, wu_pre_var_ini,
                               wu_post_var_ini, self.postsyn, self.ps_params,
                               ps_var_ini, connect_init)
        else:
            add_fct = getattr(
                model_spec,
                ("add_slave_synapse_population_" + self.ps_type))

            self.pop = add_fct(self.name, self.weight_sharing_master.name,
                               delay_steps,self.src.name, self.trg.name,
                               self.postsyn, self.ps_params, ps_var_ini)

    @deprecated("This function was poorly named, use 'set_extra_global_param' instead")
    def add_extra_global_param(self, param_name, param_values):
        """Set extra global parameter

        Args:
        param_name      --  string with the name of the extra global parameter
        param_values    --  iterable or a single value
        """
        self.set_extra_global_param(param_name, param_values)

    def set_extra_global_param(self, param_name, param_values):
        """Set extra global parameter to weight update model

        Args:
        param_name   -- string with the name of the extra global parameter
        param_values -- iterable or a single value
        """
        self._set_extra_global_param(param_name, param_values, self.w_update)

    def set_psm_extra_global_param(self, param_name, param_values):
        """Set extra global parameter to postsynaptic model

        Args:
        param_name   -- string with the name of the extra global parameter
        param_values -- iterable or a single value
        """
        self._set_extra_global_param(param_name, param_values,
                                     self.postsyn,
                                     self.psm_extra_global_params)

    def set_connectivity_extra_global_param(self, param_name, param_values):
        """Set extra global parameter to connectivity initialisation snippet

        Args:
        param_name   -- string with the name of the extra global parameter
        param_values -- iterable or a single value
        """
        assert self.weight_sharing_master is None
        self._set_extra_global_param(param_name, param_values,
                                     self.connectivity_initialiser.get_snippet(),
                                     self.connectivity_extra_global_params)

    def pull_connectivity_from_device(self):
        """Wrapper around GeNNModel.pull_connectivity_from_device"""
        self._model.pull_connectivity_from_device(self.name)

    def push_connectivity_to_device(self):
        """Wrapper around GeNNModel.push_connectivity_to_device"""
        self._model.push_connectivity_to_device(self.name)

    def load(self):
        # If synapse population has non-dense connectivity
        # which requires initialising manually
        if not self.is_dense and self.weight_sharing_master is None:
            if self.is_ragged:
                # If connectivity is located on host
                conn_loc = self.pop.get_sparse_connectivity_location()
                if (conn_loc & VarLocation_HOST) != 0:
                    # Get pointers to ragged data structure members
                    ind = self._assign_ext_ptr_array("ind",
                                                     self.weight_update_var_size,
                                                     "unsigned int")
                    row_length = self._assign_ext_ptr_array("rowLength",
                                                            self.src.size,
                                                            "unsigned int")
                    # add pointers to the object
                    self._ind = ind
                    self._row_lengths = row_length

                    # If data is available
                    if self.connections_set:
                        # Copy in row length
                        row_length[:] = self.row_lengths

                        # Create (x)range containing the index where each row starts in ind
                        row_start_idx = xrange(0, self.weight_update_var_size,
                                               self.max_row_length)

                        # Loop through ragged matrix rows
                        syn = 0
                        for i, r in zip(row_start_idx, self.row_lengths):
                            # Copy row from non-padded indices into correct location
                            ind[i:i + r] = self.ind[syn:syn + r]
                            syn += r
                    elif self.connectivity_initialiser is None:
                        raise Exception("For sparse projections, the connections"
                                        "must be set before loading a model")
                # Otherwise, if connectivity isn't located on host, 
                # give error if user tries to manually configure it
                elif self.connections_set:
                    raise Exception("If sparse connectivity is only located "
                                    "on device, it cannot be set with "
                                    "set_sparse_connections")
            elif self.connections_set:
                raise Exception("Matrix format not supported")

        # Loop through weight update model state variables
        for var_name, var_data in iteritems(self.vars):
            # If population has individual synapse variables
            if self.has_individual_synapse_vars:
                # If variable is located on host
                var_loc = self.pop.get_wuvar_location(var_name) 
                if (var_loc & VarLocation_HOST) != 0:
                    # Get view
                    var_data.view = self._assign_ext_ptr_array(
                        var_name, self.weight_update_var_size, var_data.type)

                    # Initialise variable if necessary
                    self._init_wum_var(var_data)
                else:
                    assert not var_data.init_required
                    var_data.view = None

            # Load any var initialisation egps associated with this variable
            self._load_egp(var_data.extra_global_params, var_name)

        # Load weight update model presynaptic variables
        self._load_vars(self.src.size, self.pre_vars,
                        self.pop.get_wupre_var_location)

        # Load weight update model postsynaptic variables
        self._load_vars(self.trg.size, self.post_vars,
                        self.pop.get_wupost_var_location)

        # Load postsynaptic update model variables
        if self.has_individual_postsynaptic_vars:
            self._load_vars(self.trg.size, self.psm_vars,
                            self.pop.get_psvar_location)

        # Load extra global parameters
        self._load_egp()
        self._load_egp(self.psm_extra_global_params)

    def load_init_egps(self):
        # If population isn't a weight-sharing slave
        if self.weight_sharing_master is None:
            # Load any egps used for connectivity initialisation
            self._load_egp(self.connectivity_extra_global_params)
            
            # Load any egps used for variable initialisation
            self._load_var_init_egps()

        # Load any egps used for postsynaptic model variable initialisation
        if self.has_individual_postsynaptic_vars:
            self._load_var_init_egps(self.psm_vars)
        
        # Load any egps used for pre and postsynaptic variable initialisation
        self._load_var_init_egps(self.pre_vars)
        self._load_var_init_egps(self.post_vars)

    def reinitialise(self):
        """Reinitialise synapse group"""
        # If population has individual synapse variables
        if self.has_individual_synapse_vars:
            # Loop through weight update model state variables
            # and initialise if necessary
            for var_name, var_data in iteritems(self.vars):
                self._init_wum_var(var_data)

        # Reinitialise weight update model presynaptic variables
        self._reinitialise_vars(self.src.size, self.pre_vars)

        # Reinitialise weight update model postsynaptic variables
        self._reinitialise_vars(self.trg.size, self.post_vars)

        # Reinitialise postsynaptic update model variables
        if self.has_individual_postsynaptic_vars:
            self._reinitialise_vars(self.trg.size, self.psm_vars)

    def _init_wum_var(self, var_data):
        # If initialisation is required
        if var_data.init_required:
            # If connectivity is dense,
            # copy variables  directly into view
            # **NOTE** we assume order is row-major
            if self.is_dense:
                var_data.view[:] = var_data.values
            elif self.is_ragged:
                # Sort variable to match GeNN order
                sorted_var = var_data.values[self.synapse_order]

                # Create (x)range containing the index
                # where each row starts in ind
                row_start_idx = xrange(0, self.weight_update_var_size,
                                       self.max_row_length)

                # Loop through ragged matrix rows
                syn = 0
                for i, r in zip(row_start_idx, self.row_lengths):
                    # Copy row from non-padded indices into correct location
                    var_data.view[i:i + r] = sorted_var[syn:syn + r]
                    syn += r
            else:
                raise Exception("Matrix format not supported")

class CurrentSource(Group):

    """Class representing a current injection into a group of neurons"""

    def __init__(self, name, model):
        """Init CurrentSource

        Args:
        name -- string name of the current source
        """
        super(CurrentSource, self).__init__(name, model)
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
             model, self, param_space, var_space,
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

        var_ini = model_preprocessor.var_space_to_vals(
            self.current_source_model, self.vars)
        self.pop = add_fct(self.name, self.current_source_model, pop.name,
                           self.params, var_ini)

    @deprecated("This function was poorly named, use 'set_extra_global_param' instead")
    def add_extra_global_param(self, param_name, param_values):
        """Set extra global parameter

        Args:
        param_name   -- string with the name of the extra global parameter
        param_values -- iterable or a single value
        """
        self.set_extra_global_param(param_name, param_values)

    def set_extra_global_param(self, param_name, param_values):
        """Set extra global parameter

        Args:
        param_name   -- string with the name of the extra global parameter
        param_values -- iterable or a single value
        """
        self._set_extra_global_param(param_name, param_values,
                                     self.current_source_model)

    def load(self):
        # Load current source variables
        self._load_vars()

        # Load current source extra global parameters
        self._load_egp()

    def load_init_egps(self):
        # Load any egps used for variable initialisation
        self._load_var_init_egps()

    def reinitialise(self):
        """Reinitialise current source"""
        # Reinitialise current source state variables
        self._reinitialise_vars()

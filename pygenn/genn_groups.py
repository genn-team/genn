""" @namespace pygenn.genn_groups
This module provides classes which automate model checks and parameter
conversions for GeNN Groups
"""
try:
    xrange
except NameError:  # Python 3
    xrange = range

from weakref import proxy
from deprecated import deprecated
from six import iteritems, iterkeys, itervalues
from warnings import warn
import numpy as np

from . import genn_wrapper
from . import model_preprocessor
from .model_preprocessor import ExtraGlobalParameter, Variable
from .genn_wrapper import (SynapseMatrixConnectivity_SPARSE,
                           SynapseMatrixConnectivity_BITMASK,
                           SynapseMatrixConnectivity_DENSE,
                           SynapseMatrixWeight_INDIVIDUAL,
                           SynapseMatrixWeight_INDIVIDUAL_PSM,
                           VarLocation_HOST,
                           SynapseMatrixConnectivity_PROCEDURAL)
from .genn_wrapper.Models import VarAccessDuplication_SHARED, WUVarReference

class Group(object):

    """Parent class of NeuronGroup, SynapseGroup and CurrentSource"""

    def __init__(self, name, model):
        """Init Group

        Args:
        name    -- string name of the Group
        model   -- pygenn.genn_model.GeNNModel this group is part of
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
        self.extra_global_params[param_name].set_values(param_values)

    def pull_state_from_device(self):
        """Wrapper around GeNNModel.pull_state_from_device"""
        self._model.pull_state_from_device(self.name)

    def pull_var_from_device(self, var_name):
        """Wrapper around GeNNModel.pull_var_from_device

        Args:
        var_name    --  string with the name of the variable
        """
        self._model.pull_var_from_device(self.name, var_name)

    def pull_extra_global_param_from_device(self, egp_name, size=None):
        """Wrapper around GeNNModel.pull_extra_global_param_from_device

        Args:
        egp_name    --  string with the name of the variable
        size        --  number of entries in EGP array
        """
        self._pull_extra_global_param_from_device(egp_name, size)

    def push_state_to_device(self):
        """Wrapper around GeNNModel.push_state_to_device"""
        self._model.push_state_to_device(self.name)

    def push_var_to_device(self, var_name):
        """Wrapper around GeNNModel.push_var_to_device

        Args:
        var_name    --  string with the name of the variable
        """
        self._model.push_var_to_device(self.name, var_name)

    def push_extra_global_param_to_device(self, egp_name, size=None):
        """Wrapper around GeNNModel.push_extra_global_param_to_device

        Args:
        egp_name    --  string with the name of the variable
        size        --  number of entries in EGP array
        """
        self._push_extra_global_param_to_device(egp_name, size)

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

        return self._model.genn_types[var_type].assign_ext_ptr_array(
            internal_var_name, var_size)

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

        return self._model.genn_types[var_type].assign_ext_ptr_single(
            internal_var_name)

    def _push_extra_global_param_to_device(self, egp_name, size=None,
                                           egp_dict=None):
        """Wrapper around GeNNModel.push_extra_global_param_to_device

        Args:
        egp_name    --  string with the name of the variable
        size        --  number of entries in EGP array
        """
        # If no extra global parameters dictionary
        # is specified, use standard one
        if egp_dict is None:
            egp_dict = self.extra_global_params

        # Retrieve EGP from dictionary
        egp = egp_dict[egp_name]

        # If EGP is scalar, give error
        if egp.is_scalar:
            raise Exception("Only pointer-type extra global parameters "
                            "need to be pushed")

        # If deprecated size parameter is passed, give warning and
        if size is not None:
            warn("The size parameter is no longer "
                 "required and will be removed", DeprecationWarning)
            if size != len(egp.values):
                raise ValueError("The size parameter doesn't match the "
                                 "size of the extra global parameter data")

        self._model.push_extra_global_param_to_device(self.name, egp_name,
                                                      len(egp.values))

    def _pull_extra_global_param_from_device(self, egp_name, size=None,
                                           egp_dict=None):
        """Wrapper around GeNNModel.pull_extra_global_param_from_device

        Args:
        egp_name    --  string with the name of the variable
        size        --  number of entries in EGP array
        """
        # If no extra global parameters dictionary
        # is specified, use standard one
        if egp_dict is None:
            egp_dict = self.extra_global_params

        # Retrieve EGP from dictionary
        egp = egp_dict[egp_name]

        # If EGP is scalar, give error
        if egp.is_scalar:
            raise Exception("Only pointer-type extra global parameters "
                            "need to be pulled")

        # If deprecated size parameter is passed, give warning and
        if size is not None:
            warn("The size parameter is no longer "
                 "required and will be removed", DeprecationWarning)
            if size != len(egp.values):
                raise ValueError("The size parameter doesn't match the "
                                 "size of the extra global parameter data")

        self._model.pull_extra_global_param_from_device(self.name, egp_name,
                                                        len(egp.values))

    def _load_vars(self, vars, size=None, var_dict=None, get_location_fn=None):
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
        for v in vars:
            # Get corresponding data from dictionary
            var_data = var_dict[v.name]

            # If variable is located on host
            var_loc = get_location_fn(v.name) 
            if (var_loc & VarLocation_HOST) != 0:
                # Determine how many copies of this variable are present
                num_copies = (1 if (v.access & VarAccessDuplication_SHARED) != 0
                              else self._model.batch_size)

                # Get view
                var_data.view = self._assign_ext_ptr_array(v.name, size * num_copies,
                                                           var_data.type)

                # If there is more than one copy, reshape view to 2D
                if num_copies > 1:
                    var_data.view = np.reshape(var_data.view, (num_copies, -1))

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
            elif egp_data.values is not None:
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
        name    -- string name of the group
        model   -- pygenn.genn_model.GeNNModel this neuron group is part of
        """
        super(NeuronGroup, self).__init__(name, model)
        self.neuron = None
        self.spikes = None
        self.spike_count = None
        self.spike_que_ptr = [0]
        self._max_delay_steps = 0

    @property
    def current_spikes(self):
        """Current spikes from GeNN"""
        # Get current spike queue pointer
        d = self.spike_que_ptr[0]
        
        # If batch size is zero, return single slice of spikes
        if self._model.batch_size == 1:
            return self.spikes[0, d, 0:self.spike_count[0, d]]
        # Otherwise, return list of slices
        else:
            return [self.spikes[b, d, 0:self.spike_count[b, d]]
                    for b in range(self._model.batch_size)]

    @property
    def spike_recording_data(self):
        # Get byte view of data
        data_bytes = self._spike_recording_data.view(dtype=np.uint8)

        # Reshape view into a tensor with time, batches and recording bytes
        spike_recording_bytes = self._spike_recording_words * 4
        data_bytes = np.reshape(data_bytes, (-1, self._model.batch_size, 
                                                spike_recording_bytes))

        # Calculate start time of recording
        start_time_ms = (self._model.timestep - data_bytes.shape[0]) * self._model.dT
        if start_time_ms < 0.0:
            raise Exception("spike_recording_data can only be "
                            "accessed once buffer is full.")

        # Unpack data (results in one byte per bit)
        # **THINK** is there a way to avoid this step?
        data_unpack = np.unpackbits(data_bytes, axis=2, 
                                    count=self.size,
                                    bitorder="little")

        # Loop through batches
        spike_data = []
        for b in range(self._model.batch_size):
            # Calculate indices where there are spikes
            spikes = np.where(data_unpack[:,b,:] == 1)

            # Convert spike times to ms
            spike_times = start_time_ms + (spikes[0] * self._model.dT)

            # Add to list
            spike_data.append((spike_times, spikes[1]))

        # If batch size is 1, return 1st population's spikes otherwise list
        return spike_data[0] if self._model.batch_size == 1 else spike_data

    @property
    def delay_slots(self):
        """Maximum delay steps needed for this group"""
        return self.pop.get_num_delay_slots()

    @property
    def size(self):
        return self.pop.get_num_neurons()

    @property
    def spike_recording_enabled(self):
        return self.pop.is_spike_recording_enabled()

    @spike_recording_enabled.setter
    def spike_recording_enabled(self, enabled):
        return self.pop.set_spike_recording_enabled(enabled)

    def set_neuron(self, model, param_space, var_space):
        """Set neuron, its parameters and initial variables

        Args:
        model       --  type as string of intance of the model
        param_space --  dict with model parameters
        var_space   --  dict with model variables
        """
        (self.neuron, self.type, self.param_names, self.params,
         self.var_names, self.vars, self.extra_global_params) =\
             model_preprocessor.prepare_model(
                model, self, param_space, var_space,
                model_family=genn_wrapper.NeuronModels)

    def add_to(self, num_neurons):
        """Add this NeuronGroup to a model

        Args:
        num_neurons --  int number of neurons
        """
        add_fct = getattr(self._model._model, "add_neuron_population_" + self.type)

        var_ini = model_preprocessor.var_space_to_vals(self.neuron, self.vars)
        self.pop = add_fct(self.name, num_neurons, self.neuron,
                           self.params, var_ini)

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

    def load(self, num_recording_timesteps):
        """Loads neuron group"""
        # If spike data is present on the host
        batch_size = self._model.batch_size
        if (self.pop.get_spike_location() & VarLocation_HOST) != 0:
            self.spikes = self._assign_ext_ptr_array(
                "glbSpk", self.size * self.delay_slots * batch_size,
                "unsigned int")
            self.spike_count = self._assign_ext_ptr_array(
                "glbSpkCnt", self.delay_slots * batch_size, "unsigned int")

            # Reshape to expose delay slots and batches
            self.spikes = np.reshape(self.spikes, (batch_size, 
                                                   self.delay_slots, 
                                                   self.size))
            self.spike_count = np.reshape(self.spike_count, (batch_size,
                                                             self.delay_slots))

        # If spike recording is enabled
        if self.spike_recording_enabled:
            # Calculate spike recording words
            recording_words = (self._spike_recording_words * num_recording_timesteps 
                               * batch_size)

            # Assign pointer to recording data
            self._spike_recording_data = self._assign_ext_ptr_array("recordSpk",
                                                                    recording_words,
                                                                    "uint32_t")
        if self.delay_slots > 1:
            self.spike_que_ptr = self._model._slm.assign_external_pointer_single_ui(
                "spkQuePtr" + self.name)

        # Load neuron state variables
        self._load_vars(self.neuron.get_vars())

        # Load neuron extra global params
        self._load_egp()

    def load_init_egps(self):
        # Load any egps used for variable initialisation
        self._load_var_init_egps()

    def reinitialise(self):
        """Reinitialise neuron group"""

        # Reinitialise neuron state variables
        self._reinitialise_vars()

    @property
    def _spike_recording_words(self):
        return ((self.size + 31) // 32)

class SynapseGroup(Group):

    """Class representing synaptic connection between two groups of neurons"""

    def __init__(self, name, model, weight_sharing_master=None):
        """Init SynapseGroup

        Args:
        name                    -- string name of the group
        model                   -- pygenn.genn_model.GeNNModel this synapse group is part of
        weight_sharing_master   -- SynapseGroup this synapse group is a slave of
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
            # Prepare standard model
            (self.w_update, self.wu_type, self.wu_param_names, self.wu_params,
             self.wu_var_names, self.vars, self.extra_global_params) =\
                model_preprocessor.prepare_model(
                    model, self, param_space, var_space, 
                    genn_wrapper.WeightUpdateModels)

            self.wu_pre_var_names = [vnt.name for vnt in self.w_update.get_pre_vars()]
            if pre_var_space is not None and set(iterkeys(pre_var_space)) != set(self.wu_pre_var_names):
                raise ValueError("Invalid presynaptic variable initializers "
                                 "for WeightUpdateModels")
            self.pre_vars = {
                vnt.name: Variable(vnt.name, vnt.type, pre_var_space[vnt.name], self)
                for vnt in self.w_update.get_pre_vars()}

            self.wu_post_var_names = [vnt.name for vnt in self.w_update.get_post_vars()]
            if post_var_space is not None and set(iterkeys(post_var_space)) != set(self.wu_post_var_names):
                raise ValueError("Invalid postsynaptic variable initializers "
                                 "for WeightUpdateModels")
            self.post_vars = {
                vnt.name: Variable(vnt.name, vnt.type, post_var_space[vnt.name], self)
                for vnt in self.w_update.get_post_vars()}

    def set_post_syn(self, model, param_space, var_space):
        """Set postsynaptic model, its parameters and initial variables

        Args:
        model       --  type as string of intance of the model
        param_space --  dict with model parameters
        var_space   --  dict with model variables
        """
        (self.postsyn, self.ps_type, self.ps_param_names, self.ps_params,
         self.ps_var_names, self.psm_vars, self.psm_extra_global_params) =\
             model_preprocessor.prepare_model(
                model, self, param_space, var_space,
                model_family=genn_wrapper.PostsynapticModels)

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

    def add_to(self, delay_steps):
        """Add this SynapseGroup to the a model

        Args:
        delay_steps -- number of axonal delay timesteps to simulate for this synapse group
        """
        ps_var_ini = model_preprocessor.var_space_to_vals(
                self.postsyn, {vn: self.psm_vars[vn]
                               for vn in self.ps_var_names})

        if self.weight_sharing_master is None:
            add_fct = getattr(
                self._model._model,
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

            if self.connectivity_initialiser is not None:
                snippet = self.connectivity_initialiser.get_snippet()
                self.connectivity_extra_global_params =\
                    {egp.name: ExtraGlobalParameter(egp.name, egp.type, self)
                     for egp in snippet.get_extra_global_params()}

            self.pop = add_fct(self.name, self.matrix_type, delay_steps,
                               self.src.name, self.trg.name, self.w_update,
                               self.wu_params, wu_var_ini, wu_pre_var_ini,
                               wu_post_var_ini, self.postsyn, self.ps_params,
                               ps_var_ini, connect_init)
        else:
            add_fct = getattr(
                self._model._model,
                ("add_slave_synapse_population_" + self.ps_type))

            self.pop = add_fct(self.name, self.weight_sharing_master.name,
                               delay_steps,self.src.name, self.trg.name,
                               self.postsyn, self.ps_params, ps_var_ini)

    def set_psm_extra_global_param(self, param_name, param_values):
        """Set extra global parameter to postsynaptic model

        Args:
        param_name      --  string with the name of the extra global parameter
        param_values    --  iterable or a single value
        """
        self.psm_extra_global_params[param_name].set_values(param_values)

    def set_connectivity_extra_global_param(self, param_name, param_values):
        """Set extra global parameter to connectivity initialisation snippet

        Args:
        param_name   -- string with the name of the extra global parameter
        param_values -- iterable or a single value
        """
        assert self.weight_sharing_master is None
        self.connectivity_extra_global_params[param_name].set_values(param_values)

    def pull_connectivity_from_device(self):
        """Wrapper around GeNNModel.pull_connectivity_from_device"""
        self._model.pull_connectivity_from_device(self.name)

    def push_connectivity_to_device(self):
        """Wrapper around GeNNModel.push_connectivity_to_device"""
        self._model.push_connectivity_to_device(self.name)

    def pull_psm_extra_global_param_from_device(self, egp_name):
        """Wrapper around GeNNModel.pull_extra_global_param_from_device

        Args:
        egp_name    --  string with the name of the variable
        """
        self._pull_extra_global_param_from_device(
            egp_name, size, egp_dict=self.psm_extra_global_params)

    def push_psm_extra_global_param_to_device(self, egp_name):
        """Wrapper around GeNNModel.push_extra_global_param_to_device

        Args:
        egp_name    --  string with the name of the variable
        """
        self._push_extra_global_param_to_device(
            egp_name, self.psm_extra_global_params)

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
        if self.weight_sharing_master is None:
            for v in self.w_update.get_vars():
                # Get corresponding data from dictionary
                var_data = self.vars[v.name]

                # If population has individual synapse variables
                if self.has_individual_synapse_vars:
                    # If variable is located on host
                    var_loc = self.pop.get_wuvar_location(v.name) 
                    if (var_loc & VarLocation_HOST) != 0:
                        # Determine how many copies of this variable are present
                        num_copies = (1 if (v.access & VarAccessDuplication_SHARED) != 0
                                      else self._model.batch_size)
                        # Get view
                        var_data.view = self._assign_ext_ptr_array(
                            v.name, self.weight_update_var_size * num_copies, 
                            var_data.type)

                        # If there is more than one copy, reshape view to 2D
                        if num_copies > 1:
                            var_data.view = np.reshape(var_data.view, 
                                                       (num_copies, -1))

                        # Initialise variable if necessary
                        self._init_wum_var(var_data, num_copies)
                    else:
                        assert not var_data.init_required
                        var_data.view = None

                # Load any var initialisation egps associated with this variable
                self._load_egp(var_data.extra_global_params, v.name)

            # Load weight update model presynaptic variables
            self._load_vars(self.w_update.get_pre_vars(), self.src.size,
                            self.pre_vars, self.pop.get_wupre_var_location)

            # Load weight update model postsynaptic variables
            self._load_vars(self.w_update.get_post_vars(), self.trg.size, 
                            self.post_vars, self.pop.get_wupost_var_location)

            # Load postsynaptic update model variables
            if self.has_individual_postsynaptic_vars:
                self._load_vars(self.postsyn.get_vars(), self.trg.size,
                                self.psm_vars, self.pop.get_psvar_location)

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
            for v in self.w_update.get_vars():
                # Get corresponding data from dictionary
                var_data = self.vars[v.name]

                # If variable is located on host
                var_loc = self.pop.get_wuvar_location(v.name) 
                if (var_loc & VarLocation_HOST) != 0:
                    # Determine how many copies of this variable are present
                    num_copies = (1 if (v.access & VarAccessDuplication_SHARED) != 0
                                  else self._model.batch_size)

                    # Initialise
                    self._init_wum_var(var_data, num_copies)

        # Reinitialise weight update model presynaptic variables
        self._reinitialise_vars(self.src.size, self.pre_vars)

        # Reinitialise weight update model postsynaptic variables
        self._reinitialise_vars(self.trg.size, self.post_vars)

        # Reinitialise postsynaptic update model variables
        if self.has_individual_postsynaptic_vars:
            self._reinitialise_vars(self.trg.size, self.psm_vars)

    def _init_wum_var(self, var_data, num_copies):
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
                    if num_copies == 1:
                        var_data.view[i:i + r] = sorted_var[syn:syn + r]
                    else:
                        var_data.view[i:i + r,:] = sorted_var[syn:syn + r,:]
                    syn += r
            else:
                raise Exception("Matrix format not supported")

class CurrentSource(Group):

    """Class representing a current injection into a group of neurons"""

    def __init__(self, name, model):
        """Init CurrentSource

        Args:
        name    -- string name of the current source
        model   -- pygenn.genn_model.GeNNModel this current source is part of
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
        """Set current source model, its parameters and initial variables

        Args:
        model       --  type as string of intance of the model
        param_space --  dict with model parameters
        var_space   --  dict with model variables
        """
        (self.current_source_model, self.type, self.param_names, self.params,
         self.var_names, self.vars, self.extra_global_params) =\
             model_preprocessor.prepare_model(
                model, self, param_space, var_space,
                model_family=genn_wrapper.CurrentSourceModels)

    def add_to(self, pop):
        """Attach this CurrentSource to NeuronGroup and
        add it to the pygenn.genn_model.GeNNModel

        Args:
        pop         --  instance of NeuronGroup into which this CurrentSource
                        should be injected
        """
        add_fct = getattr(self._model._model, "add_current_source_" + self.type)
        self.target_pop = pop

        var_ini = model_preprocessor.var_space_to_vals(
            self.current_source_model, self.vars)
        self.pop = add_fct(self.name, self.current_source_model, pop.name,
                           self.params, var_ini)

    def load(self):
        # Load current source variables
        self._load_vars(self.current_source_model.get_vars())

        # Load current source extra global parameters
        self._load_egp()

    def load_init_egps(self):
        # Load any egps used for variable initialisation
        self._load_var_init_egps()

    def reinitialise(self):
        """Reinitialise current source"""
        # Reinitialise current source state variables
        self._reinitialise_vars()

class CustomUpdate(Group):

    """Class representing a custom update"""

    def __init__(self, name, model):
        """Init CustomUpdate

        Args:
        name    -- string name of the custom update
        model   -- pygenn.genn_model.GeNNModel this custom update is part of
        """
        super(CustomUpdate, self).__init__(name, model)
        self.custom_update_model = None
        self.var_refs = {}
        self.custom_wu_update = False

    def set_custom_update_model(self, model, param_space, var_space, var_ref_space):
        """Set custom update model, its parameters, 
        initial variables and variable referneces

        Args:
        model           --  type as string or instance of the model
        param_space     --  dict with model parameters
        var_space       --  dict with model variables
        var_references  --  dict with model variables
        """

        # Prepare standard model
        (self.custom_update_model, self.type, self.param_names, self.params,
         self.var_names, self.vars, self.extra_global_params) =\
            model_preprocessor.prepare_model(
                model, self, param_space, var_space, 
                genn_wrapper.CustomUpdateModels)

        # Check variable references
        self.var_ref_names = [vnt.name for vnt in self.custom_update_model.get_var_refs()]
        if var_ref_space is not None and set(iterkeys(var_ref_space)) != set(self.var_ref_names):
            raise ValueError("Invalid variable reference initializers "
                             "for CustomUpdateModels")

        # Count wu var references in list
        num_wu_var_refs = sum(isinstance(v[0], WUVarReference)
                              for v in itervalues(var_ref_space))

        # If there's a mixture of references to weight 
        # update  model and other variables, give error
        if num_wu_var_refs != 0 and num_wu_var_refs != len(var_ref_space):
            raise ValueError("Custom updates cannot be created with "
                             "references pointing to a mixture of "
                             "weight update and other variables")

        # Set flag 
        self.custom_wu_update = (num_wu_var_refs != 0)

        # Store variable references in class
        self.var_refs = var_ref_space

    def add_to(self, group_name):
        """Attach this CurrentSource to NeuronGroup and
        add it to the pygenn.genn_model.GeNNModel

        Args:
        group_name  --  name of update group this update should be performed in
        """
        add_fct = getattr(self._model._model, "add_custom_update_" + self.type)

        var_ini = model_preprocessor.var_space_to_vals(self.custom_update_model,
                                                       self.vars)
        if self.custom_wu_update:
            var_refs = model_preprocessor.var_ref_space_to_wu_var_refs(
                self.custom_update_model, self.var_refs)
        else:
            var_refs = model_preprocessor.var_ref_space_to_var_refs(
                self.custom_update_model, self.var_refs)

        self.pop = add_fct(self.name, group_name, self.custom_update_model, 
                           self.params, var_ini, var_refs)

    def load(self):
        # If this is a custom weight update
        if self.custom_wu_update:
            # Assert that population has individual synapse variables
            assert self._synapse_group.has_individual_synapse_vars

            # Loop through state variables
            for v in self.custom_update_model.get_vars():
                # Get corresponding data from dictionary
                var_data = self.vars[v.name]

                # If variable is located on host
                var_loc = self.pop.get_var_location(v.name) 
                if (var_loc & VarLocation_HOST) != 0:
                    # Determine how many copies of this variable are present
                    #num_copies = (1 if (v.access & VarAccessDuplication_SHARED) != 0
                    #              else self._model.batch_size)
                    num_copies = 1

                    # Get view
                    size = self._synapse_group.weight_update_var_size * num_copies
                    var_data.view = self._assign_ext_ptr_array(
                        v.name, size, var_data.type)

                    # If there is more than one copy, reshape view to 2D
                    if num_copies > 1:
                        var_data.view = np.reshape(var_data.view, 
                                                   (num_copies, -1))

                    # Initialise variable if necessary
                    self._synapse_group._init_wum_var(var_data, num_copies)

                # Load any var initialisation egps associated with this variable
                self._load_egp(var_data.extra_global_params, v.name)
        # Otherwise, load variables 
        else:
            self._load_vars(self.custom_update_model.get_vars(),
                            size=self.pop.get_size())

        # Load custom update extra global parameters
        self._load_egp()

    def load_init_egps(self):
        # Load any egps used for variable initialisation
        self._load_var_init_egps()

    def reinitialise(self):
        """Reinitialise custom update"""
        # If this is a custom weight update
        if self.custom_wu_update:
            # Assert that population has individual synapse variables
            assert self._synapse_group.has_individual_synapse_vars

            # Loop through custom update state variables
            for v in self.custom_update_model.get_vars():
                # Get corresponding data from dictionary
                var_data = self.vars[v.name]

                # If variable is located on host
                var_loc = self.pop.get_var_location(v.name) 
                if (var_loc & VarLocation_HOST) != 0:
                    # Determine how many copies of this variable are present
                    #num_copies = (1 if (v.access & VarAccessDuplication_SHARED) != 0
                    #              else self._model.batch_size)
                    num_copies = 1

                    # Initialise
                    self._synapse_group._init_wum_var(var_data, num_copies)
        # Otherwise, reinitialise current source state variables
        else:
            self._reinitialise_vars(size=self.pop.get_size())

    @property
    def _synapse_group(self):
        """Get SynapseGroup associated with custom weight update"""
        assert self.custom_wu_update

        # Return Python synapse group reference from 
        # first (arbitrarily) variable reference
        return next(itervalues(self.var_refs))[1]

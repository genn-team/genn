""" @namespace pygenn.genn_groups
This module provides classes which automate model checks and parameter
conversions for GeNN Groups
"""
try:
    xrange
except NameError:  # Python 3
    xrange = range

from deprecated import deprecated
from six import iteritems, iterkeys, itervalues
from warnings import warn
from weakref import proxy
import numpy as np

from . import neuron_models, types
from .genn import (CustomUpdateWU, SynapseMatrixConnectivity,
                   SynapseMatrixWeight, VarAccessDuplication, VarLocation)
from .model_preprocessor import prepare_model, ExtraGlobalParameter, Variable


class GroupMixin(object):

    """Parent class of NeuronGroupMixin, 
    SynapseGroupMixin and CurrentSourceMixin"""

    def _init_group(self, model):
        """Init Group

        Args:
        model   -- pygenn.genn_model.GeNNModel this group is part of
        """
        self._model = proxy(model)
        self.vars = {}
        self.extra_global_params = {}

    def pull_state_from_device(self):
        """Pull state from the device for a given population"""
        if not self._model._loaded:
            raise Exception("GeNN model has to be loaded before pulling")

        self._model._slm.pull_state_from_device(self.name)

    def pull_var_from_device(self, var_name):
        """Pull variable from the device for a given population

        Args:
        var_name    --  string with the name of the variable
        """
        if not self._model._loaded:
            raise Exception("GeNN model has to be loaded before pulling")

        self._model._slm.pull_var_from_device(self.name, var_name)

    def pull_extra_global_param_from_device(self, egp_name):
        """Pull extra global parameter from device

        Args:
        egp_name    --  string with the name of the variable
        size        --  number of entries in EGP array
        """
        self._pull_extra_global_param_from_device(egp_name)

    def push_state_to_device(self):
        """Push all population state variables to the device"""
        if not self._model._loaded:
            raise Exception("GeNN model has to be loaded before pushing")
        
        self._model._slm.push_state_to_device(self.name)

    def push_var_to_device(self, var_name):
        """Push population state variable to the device

        Args:
        var_name    --  string with the name of the variable
        """
        self._model._slm.push_var_to_device(self.name, var_name)

    def push_extra_global_param_to_device(self, egp_name):
        """Push extra global parameter to device

        Args:
        egp_name    --  string with the name of the variable
        """
        self._push_extra_global_param_to_device(egp_name)
    
    def _assign_ext_ptr_array(self, var_name, var_size, var_type):
        """Assign a variable to an external numpy array

        Args:
        var_name    --  string a fully qualified name of the variable to assign
        var_size    --  int the size of the variable
        var_type    --  ResolvedType object

        Returns numpy array of type var_type

        Raises ValueError if variable type is not supported
        """

        internal_var_name = var_name + self.name

        # Get numpy data type corresponding to type
        dtype = self._model.genn_types[var_type]
        
        # Calculate bytes
        num_bytes = np.dtype(dtype).itemsize * var_size
        
        # Get dtype view of array memoryview
        array = np.asarray(self._model._slm.get_array(
            internal_var_name, num_bytes)).view(dtype)
        assert not array.flags["OWNDATA"]
        return array

    def _assign_ext_ptr_single(self, var_name, var_type):
        """Assign a variable to an external scalar value containing one element

        Args:
        var_name    --  string a fully qualified name of the variable to assign
        var_type    --  ResolvedType object

        Returns numpy array of type var_type

        Raises ValueError if variable type is not supported
        """

        internal_var_name = var_name + self.name

        # Get numpy data type corresponding to type
        dtype = self._model.genn_types[var_type]
        
        # Get dtype view of array memoryview
        array = np.asarray(self._model._slm.get_scalar(
            internal_var_name, np.dtype(dtype).itemsize)).view(dtype)
        assert not array.flags["OWNDATA"]
        return array

    def _push_extra_global_param_to_device(self, egp_name, egp_dict=None):
        """Wrapper around GeNNModel.push_extra_global_param_to_device

        Args:
        egp_name    --  string with the name of the variable
        """
        # If no extra global parameters dictionary
        # is specified, use standard one
        if egp_dict is None:
            egp_dict = self.extra_global_params

        # Retrieve EGP from dictionary
        egp = egp_dict[egp_name]

        self._model._slm.push_extra_global_param_to_device(self.name, egp_name,
                                                           len(egp.values))

    def _pull_extra_global_param_from_device(self, egp_name, egp_dict=None):
        """Wrapper around GeNNModel.pull_extra_global_param_from_device

        Args:
        egp_name    --  string with the name of the variable
        """
        # If no extra global parameters dictionary
        # is specified, use standard one
        if egp_dict is None:
            egp_dict = self.extra_global_params

        # Retrieve EGP from dictionary
        egp = egp_dict[egp_name]

        self._model._slm.pull_extra_global_param_from_device(self.name, egp_name,
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
            get_location_fn = self.get_var_location

        # Loop through variables
        for v in vars:
            # Get corresponding data from dictionary
            var_data = var_dict[v.name]

            # If variable is located on host
            var_loc = get_location_fn(v.name) 
            if var_loc & VarLocation.HOST:
                # Determine how many copies of this variable are present
                num_copies = (1 if v.access & VarAccessDuplication.SHARED
                              else self._model.batch_size)
                
                # Determine size of this variable
                var_size = (1 if v.access & VarAccessDuplication.SHARED_NEURON
                            else size)
                            
                # Get view
                resolved_type = var_data.type.resolve(self._model.type_context)
                var_data.view = self._assign_ext_ptr_array(v.name, var_size * num_copies,
                                                           resolved_type)

                # If there is more than one copy, reshape view to 2D
                if num_copies > 1:
                    var_data.view = np.reshape(var_data.view, (num_copies, -1))

                # If manual initialisation is required, copy over variables
                if var_data.init_required:
                    var_data.view[:] = var_data.values
            else:
                assert not var_data.init_required
                var_data.view = None

    def _load_egp(self, egp_dict=None, egp_suffix=""):
        # If no EGP dictionary is specified, use standard one
        if egp_dict is None:
            egp_dict = self.extra_global_params

        # Loop through extra global params
        for egp_name, egp_data in iteritems(egp_dict):
            resolved_type = egp_data.type.resolve(self._model.type_context)
            if egp_data.values is not None:
                # Allocate memory
                self._model._slm.allocate_extra_global_param(
                    self.name, egp_name + egp_suffix, len(egp_data.values))

                # Assign view
                egp_data.view = self._assign_ext_ptr_array(egp_name + egp_suffix,
                                                           len(egp_data.values), 
                                                           resolved_type)

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

    def _unload_vars(self, var_dict=None):
        # If no variable dictionary is specified, use standard one
        if var_dict is None:
            var_dict = self.vars

        # Loop through variables and clear views
        for v in itervalues(var_dict):
            v.view = None
            for e in itervalues(v.extra_global_params):
                e.view = None

    def _unload_egps(self, egp_dict=None):
        # If no EGP dictionary is specified, use standard one
        if egp_dict is None:
            egp_dict = self.extra_global_params

        # Loop through extra global params and clear views
        for e in itervalues(egp_dict):
            e.view = None

class NeuronGroupMixin(GroupMixin):

    """Class representing a group of neurons"""

    def _init_group(self, model, var_space):
        """Init NeuronGroupMixin

        Args:
        model   -- pygenn.genn_model.GeNNModel this neuron group is part of
        """
        super(NeuronGroupMixin, self)._init_group(model)
        self.spike_que_ptr = None
        self._spike_recording_data = None
        self._spike_event_recording_data = None

        self.vars, self.extra_global_params = prepare_model(
            self.neuron_model, self, var_space)

    @property
    def spike_recording_data(self):
        return self._get_event_recording_data(True)

    @property
    def spike_event_recording_data(self):
        return self._get_event_recording_data(False)

    @property
    def size(self):
        return self.num_neurons

    def load(self, num_recording_timesteps):
        """Loads neuron group"""
        # If spike data is present on the host
        batch_size = self._model.batch_size

        # If spike recording is enabled
        if self.spike_recording_enabled:
            # Calculate spike recording words
            recording_words = (self._event_recording_words * num_recording_timesteps 
                               * batch_size)

            # Assign pointer to recording data
            self._spike_recording_data = self._assign_ext_ptr_array(
                "recordSpk", recording_words, types.Uint32)

        # If spike-event recording is enabled
        if self.spike_event_recording_enabled:
            # Calculate spike recording words
            recording_words = (self._event_recording_words * num_recording_timesteps 
                               * batch_size)

            # Assign pointer to recording data
            self._spike_event_recording_data = self._assign_ext_ptr_array(
                "recordSpkEvent", recording_words, types.Uint32)

        if self.num_delay_slots > 1:
            self.spike_que_ptr = self._model._slm.assign_external_pointer_single_ui(
                "spkQuePtr" + self.name)

        # Load neuron state variables
        self._load_vars(self.neuron_model.get_vars())

        # Load neuron extra global params
        self._load_egp()

    def unload(self):
        self.spike_que_ptr = None
        self._spike_recording_data = None
        self._spike_event_recording_data = None

        self._unload_vars()
        self._unload_egps()

    def load_init_egps(self):
        # Load any egps used for variable initialisation
        self._load_var_init_egps()

    @property
    def _event_recording_words(self):
        return ((self.size + 31) // 32)
        
    def _get_event_time_view(self, name):
        # Get view
        batch_size = self._model.batch_size
        view = self._assign_ext_ptr_array(
            name, self.size * self.num_delay_slots * batch_size,
            self._model.time_precision)

        # Reshape to expose delay slots and batches
        view = np.reshape(view, (batch_size, self.num_delay_slots,
                                 self.size))
        return view

    def _get_event_recording_data(self, true_spike):
        # Get byte view of data
        recording_data = (self._spike_recording_data if true_spike 
                          else self._spike_event_recording_data)
        data_bytes = recording_data.view(dtype=np.uint8)

        # Reshape view into a tensor with time, batches and recording bytes
        event_recording_bytes = self._event_recording_words * 4
        data_bytes = np.reshape(data_bytes, (-1, self._model.batch_size, 
                                                event_recording_bytes))

        # Calculate start time of recording
        start_time_ms = (self._model.timestep - data_bytes.shape[0]) * self._model.dt
        if start_time_ms < 0.0:
            raise Exception("spike_recording_data can only be "
                            "accessed once buffer is full.")

        # Unpack data (results in one byte per bit)
        # **THINK** is there a way to avoid this step?
        data_unpack = np.unpackbits(data_bytes, axis=2, 
                                    count=self.size,
                                    bitorder="little")

        # Loop through batches
        event_data = []
        for b in range(self._model.batch_size):
            # Calculate indices where there are events
            events = np.where(data_unpack[:,b,:] == 1)

            # Convert event times to ms
            event_times = start_time_ms + (events[0] * self._model.dt)

            # Add to list
            event_data.append((event_times, events[1]))

        # If batch size is 1, return 1st population's events otherwise list
        # **TODO** API fiddling
        return event_data[0] if self._model.batch_size == 1 else event_data


class SynapseGroupMixin(GroupMixin):

    """Class representing synaptic connection between two groups of neurons"""
    
    def _init_group(self, model, ps_var_space, wu_var_space, wu_pre_var_space,
                    wu_post_var_space, source, target):
        """Init SynapseGroupMixin

        Args:
        model   -- pygenn.GeNNModel this neuron group is part of
        """
        super(SynapseGroupMixin, self)._init_group(model)
        
        self.src = source
        self.trg = target
        self.in_syn = None
        self.connections_set = False
        
        self.vars, self.extra_global_params = prepare_model(
            self.wu_model, self, wu_var_space)
        self.psm_vars, self.psm_extra_global_params = prepare_model(
            self.ps_model, self, ps_var_space)
        
        self.pre_vars = {vnt.name: Variable(vnt.name, vnt.type, 
                                            wu_pre_var_space[vnt.name], self)
                         for vnt in self.wu_model.get_pre_vars()}
        self.post_vars = {vnt.name: Variable(vnt.name, vnt.type, 
                                             wu_post_var_space[vnt.name], self)
                          for vnt in self.wu_model.get_post_vars()}
        
        if self.matrix_type & SynapseMatrixConnectivity.TOEPLITZ:
            connect_init = self.toeplitz_connectivity_initialiser
        else:
            connect_init = self.sparse_connectivity_initialiser
        self.connectivity_extra_global_params =\
            {egp.name: ExtraGlobalParameter(egp.name, egp.type, self)
            for egp in connect_init.snippet.get_extra_global_params()}

    @property
    def num_synapses(self):
        """Number of synapses in group"""
        if self.matrix_type & SynapseMatrixConnectivity.DENSE:
            return self.trg.size * self.src.size
        elif (self.matrix_type & SynapseMatrixConnectivity.SPARSE):
            return self._num_synapses
        else:
            raise Exception("Matrix format not supported")

    @property
    def weight_update_var_size(self):
        """Size of each weight update variable"""
        if self.matrix_type & SynapseMatrixConnectivity.DENSE:
            return self.trg.size * self.src.size
        elif self.matrix_type & SynapseMatrixConnectivity.SPARSE:
            return self.max_connections * self.src.size
        elif self.matrix_type & SynapseMatrixWeight.KERNEL:
            return int(np.product(self.kernel_size))
        else:
            raise Exception("Matrix format not supported")

    def get_var_values(self, var_name):
        var_view = self.vars[var_name].view

        if self.matrix_type & SynapseMatrixConnectivity.DENSE:
            return np.copy(var_view)
        elif self.matrix_type & SynapseMatrixConnectivity.KERNEL:
            return np.copy(var_view)
        elif self.matrix_type & SynapseMatrixConnectivity.SPARSE:
            max_rl = self.max_connections
            row_ls = self._row_lengths if self._connectivity_initialiser_provided else self.row_lengths

            # Create range containing the index where each row starts in ind
            row_start_idx = xrange(0, self.weight_update_var_size, max_rl)

            # Build list of subviews representing each row
            rows = [var_view[i:i + r] for i, r in zip(row_start_idx, row_ls)]

            # Stack all rows together into single array
            return np.hstack(rows)
        else:
            raise Exception("Matrix format not supported")

    def set_sparse_connections(self, pre_indices, post_indices):
        """Set ragged format connections between two groups of neurons

        Args:
        pre_indices     --  ndarray of presynaptic indices
        post_indices    --  ndarray of postsynaptic indices
        """
        if self.matrix_type & SynapseMatrixConnectivity.SPARSE:
            # Lexically sort indices
            self.synapse_order = np.lexsort((post_indices, pre_indices))

            # Count synapses
            self._num_synapses = len(post_indices)

            # Count the number of synapses in each row
            row_lengths = np.bincount(pre_indices, minlength=self.src.size)
            row_lengths = row_lengths.astype(np.uint32)

            # Use maximum for max connections
            max_row_length = int(np.amax(row_lengths))
            self.max_connections = max_row_length

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
        if self.matrix_type & SynapseMatrixConnectivity.SPARSE:
            rl = self._row_lengths if self._connectivity_initialiser_provided else self.row_lengths

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
        if (self.matrix_type & SynapseMatrixConnectivity.SPARSE):
            if not self._connectivity_initialiser_provided:
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
                    self._ind[i * self.max_connections: (i * self.max_connections) + r]
                        for i, r in enumerate(self._row_lengths)])

        else:
            raise Exception("get_sparse_post_inds only supports"
                            "ragged format sparse connectivity")

    def pull_connectivity_from_device(self):
        """Wrapper around GeNNModel.pull_connectivity_from_device"""
        self._model._slm.pull_connectivity_from_device(self.name)

    def push_connectivity_to_device(self):
        """Wrapper around GeNNModel.push_connectivity_to_device"""
        self._model._slm.push_connectivity_to_device(self.name)
    
    def pull_in_syn_from_device(self):
        """Pull synaptic input current from device"""
        self.pull_var_from_device("outPost")

    def push_in_syn_to_device(self):
        """Push synaptic input current to device"""
        self.push_var_to_device("outPost")
        
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
        if not (self.matrix_type & SynapseMatrixConnectivity.DENSE):
            if (self.matrix_type & SynapseMatrixConnectivity.SPARSE):
                # If connectivity is located on host
                conn_loc = self.sparse_connectivity_location
                if conn_loc & VarLocation.HOST:
                    # Get pointers to ragged data structure members
                    ind = self._assign_ext_ptr_array("ind",
                                                     self.weight_update_var_size,
                                                     self._sparse_ind_type)
                    row_length = self._assign_ext_ptr_array("rowLength",
                                                            self.src.size,
                                                            types.Uint32)
                    # add pointers to the object
                    self._ind = ind
                    self._row_lengths = row_length

                    # If data is available
                    if self.connections_set:
                        # Copy in row length
                        row_length[:] = self.row_lengths

                        # Create (x)range containing the index where each row starts in ind
                        row_start_idx = xrange(0, self.weight_update_var_size,
                                               self.max_connections)

                        # Loop through ragged matrix rows
                        syn = 0
                        for i, r in zip(row_start_idx, self.row_lengths):
                            # Copy row from non-padded indices into correct location
                            ind[i:i + r] = self.ind[syn:syn + r]
                            syn += r
                    elif not self._connectivity_initialiser_provided:
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
        for v in self.wu_model.get_vars():
            # Get corresponding data from dictionary
            var_data = self.vars[v.name]

            # If population has individual synapse variables
            if ((self.matrix_type & SynapseMatrixWeight.INDIVIDUAL) or 
                    (self.matrix_type & SynapseMatrixWeight.KERNEL)):
                # If variable is located on host
                var_loc = self.get_wu_var_location(v.name) 
                if var_loc & VarLocation.HOST:
                    # Determine how many copies of this variable are present
                    num_copies = (1 if (v.access & VarAccessDuplication.SHARED) != 0
                                    else self._model.batch_size)
                    # Get view
                    resolved_type = var_data.type.resolve(self._model.type_context)
                    var_data.view = self._assign_ext_ptr_array(
                        v.name, self.weight_update_var_size * num_copies, 
                        resolved_type)

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

        # If population's presynaptic weight update hasn't been 
        # fused, load weight update model presynaptic variables
        if not self._wu_pre_model_fused:
            self._load_vars(self.wu_model.get_pre_vars(), self.src.size,
                            self.pre_vars, self.get_wu_pre_var_location)

        # If population's postsynaptic weight update hasn't been 
        # fused, load weight update model postsynaptic variables
        if not self._wu_post_model_fused:
            self._load_vars(self.wu_model.get_post_vars(), self.trg.size, 
                            self.post_vars, self.get_wu_post_var_location)
        
        # If this synapse group's postsynaptic model hasn't been fused
        if not self._ps_model_fused:
            # Load postsynaptic update model variables
            self._load_vars(self.ps_model.get_vars(), self.trg.size,
                            self.psm_vars, self.get_ps_var_location)
                
            # If it's inSyn is accessible on the host
            if self.in_syn_location & VarLocation.HOST:
                # Get view
                self.out_post = self._assign_ext_ptr_array(
                    "outPost", self.trg.size * self._model.batch_size,
                    self._model.precision)

                # Reshape to expose batches
                self.out_post = np.reshape(self.out_post, (self._model.batch_size,
                                                           self.trg.size))

        # Load extra global parameters
        self._load_egp()
        self._load_egp(self.psm_extra_global_params)

    def load_init_egps(self):
        # Load any egps used for connectivity initialisation
        self._load_egp(self.connectivity_extra_global_params)

        # Load any egps used for variable initialisation
        self._load_var_init_egps()

        # Load any egps used for postsynaptic model variable initialisation
        self._load_var_init_egps(self.psm_vars)

        # Load any egps used for pre and postsynaptic variable initialisation
        self._load_var_init_egps(self.pre_vars)
        self._load_var_init_egps(self.post_vars)

    @property
    def _connectivity_initialiser_provided(self):
        assert self.matrix_type & SynapseMatrixConnectivity.SPARSE
        
        snippet = self.sparse_connectivity_initialiser.snippet
        return (len(snippet.get_row_build_code()) > 0 
                or len(snippet.get_col_build_code()) > 0)

    def unload(self):
        self._ind = None
        self._row_lengths = None
        self.in_syn = None

        self._unload_vars()
        self._unload_vars(self.pre_vars)
        self._unload_vars(self.post_vars)
        self._unload_vars(self.psm_vars)
        self._unload_egps()
        self._unload_egps(self.psm_extra_global_params)
        self._unload_egps(self.connectivity_extra_global_params)

    def _init_wum_var(self, var_data, num_copies):
        # If initialisation is required
        if var_data.init_required:
            # If connectivity is dense,
            # copy variables  directly into view
            # **NOTE** we assume order is row-major
            if ((self.matrix_type & SynapseMatrixConnectivity.DENSE) or
                (self.matrix_type & SynapseMatrixWeight.KERNEL)):
                var_data.view[:] = var_data.values
            elif (self.matrix_type & SynapseMatrixConnectivity.SPARSE):
                # Sort variable to match GeNN order
                sorted_var = var_data.values[self.synapse_order]

                # Create (x)range containing the index
                # where each row starts in ind
                row_start_idx = xrange(0, self.weight_update_var_size,
                                       self.max_connections)

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

class CurrentSourceMixin(GroupMixin):

    """Class representing a current injection into a group of neurons"""

    def _init_group(self, model, var_space, target_pop):
        """Init NeuronGroup

        Args:
        name    -- string name of the group
        model   -- pygenn.genn_model.GeNNModel this neuron group is part of
        """
        super(CurrentSourceMixin, self)._init_group(model)
        self.target_pop = target_pop
        self.vars, self.extra_global_params = prepare_model(
            self.current_source_model, self, var_space)
        
    @property
    def size(self):
        """Number of neuron in the injected population"""
        return self.target_pop.size


    def load(self):
        # Load current source variables
        self._load_vars(self.current_source_model.get_vars())

        # Load current source extra global parameters
        self._load_egp()

    def load_init_egps(self):
        # Load any egps used for variable initialisation
        self._load_var_init_egps()

    def unload(self):
        self._unload_vars()
        self._unload_egps()

class CustomUpdateMixin(GroupMixin):
    """Class representing a custom update"""
    def _init_group(self, model, var_space):
        """Init CustomUpdate

        Args:
        name    -- string name of the group
        model   -- pygenn.genn_model.GeNNModel this neuron group is part of
        """
        super(CustomUpdateMixin, self)._init_group(model)
        self.vars, self.extra_global_params = prepare_model(
            self.custom_update_model, self, var_space)
    

    def load(self):
        # If this is a custom weight update
        if self._custom_wu_update:
            # Assert that population doesn't have procedural connectivity
            assert not (self._synapse_group.matrix_type 
                        & SynapseMatrixConnectivity.PROCEDURAL)

            # Loop through state variables
            for v in self.custom_update_model.get_vars():
                # Get corresponding data from dictionary
                var_data = self.vars[v.name]

                # If variable is located on host
                var_loc = self.get_var_location(v.name) 
                if var_loc & VarLocation.HOST:
                    # Determine how many copies of this variable are present
                    # **YUCK** this isn't quite right - really should look at is_batched()
                    #num_copies = (1 if (v.access & VarAccessDuplication_SHARED) != 0
                    #              else self._model.batch_size)
                    num_copies = 1

                    # Get view
                    size = self._synapse_group.weight_update_var_size * num_copies
                    resolved_type = var_data.type.resolve(self._model.type_context)
                    var_data.view = self._assign_ext_ptr_array(
                        v.name, size, resolved_type)

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
                            size=self.size)

        # Load custom update extra global parameters
        self._load_egp()

    def load_init_egps(self):
        # Load any egps used for variable initialisation
        self._load_var_init_egps()

    def unload(self):
        self._unload_vars()
        self._unload_egps()

    @property
    def _custom_wu_update(self):
        return isinstance(self, CustomUpdateWU)

    @property
    def _synapse_group(self):
        """Get SynapseGroup associated with custom weight update"""
        assert self._custom_wu_update

        # Return Python synapse group reference from 
        # first (arbitrarily) variable reference
        return next(itervalues(self.var_references)).synapse_group

class CustomConnectivityUpdateMixin(GroupMixin):
    """Class representing a custom connectivity update"""
    def _init_group(self, model, var_space, pre_var_space, 
                    post_var_space, synapse_group):
        """Init CustomConnectivityUpdateGroup

        Args:
        name    -- string name of the group
        model   -- pygenn.genn_model.GeNNModel this neuron group is part of
        """
        super(CustomConnectivityUpdateMixin, self)._init_group(model)
        self.synapse_group = synapse_group
        self.vars, self.extra_global_params = prepare_model(
            self.model, self, var_space)
        self.pre_vars = {vnt.name: Variable(vnt.name, vnt.type, 
                                            pre_var_space[vnt.name], self)
                         for vnt in self.model.get_pre_vars()}
        self.post_vars = {vnt.name: Variable(vnt.name, vnt.type, 
                                             post_var_space[vnt.name], self)
                          for vnt in self.model.get_post_vars()}

    def load(self):
        # Loop through state variables
        for v in self.model.get_vars():
            # Get corresponding data from dictionary
            var_data = self.vars[v.name]

            # If variable is located on host
            var_loc = self.get_var_location(v.name) 
            if var_loc & VarLocation.HOST:
                # Get view
                size = self._synapse_group.weight_update_var_size
                resolved_type = var_data.type.resolve(self._model.type_context)
                var_data.view = self._assign_ext_ptr_array(
                    v.name, size, resolved_type)

                # Initialise variable if necessary
                self._synapse_group._init_wum_var(var_data, 1)

            # Load any var initialisation egps associated with this variable
            self._load_egp(var_data.extra_global_params, v.name)
  
        # Load pre and postsynaptic variables
        self._load_vars(self.model.get_pre_vars(), 
                        self.synapse_group.src.size,
                        self.pre_vars, self.get_pre_var_location)
        self._load_vars(self.model.get_post_vars(), 
                        self.synapse_group.trg.size,
                        self.post_vars, self.get_post_var_location)

        # Load custom update extra global parameters
        self._load_egp()

    def load_init_egps(self):
        # Load any egps used for variable initialisation
        self._load_var_init_egps()
        
        # Load any egps used for pre and postsynaptic variable initialisation
        self._load_var_init_egps(self.pre_vars)
        self._load_var_init_egps(self.post_vars)

    def unload(self):
        self._unload_vars()
        self._unload_vars(self.pre_vars)
        self._unload_vars(self.post_vars)
        self._unload_egps()

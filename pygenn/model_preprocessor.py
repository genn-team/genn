from copy import copy
from deprecated import deprecated
from numbers import Number
from typing import Union
from weakref import proxy, ref, ProxyTypes
import numpy as np
from ._genn import (NumericValue, ResolvedType, SynapseMatrixConnectivity,
                    SynapseMatrixWeight, UnresolvedType, VarInit)
from .init_var_snippets import Uninitialised

# Type aliases
VarTypeType = Union[ResolvedType, UnresolvedType]

class ArrayBase:
    """Base class for classes which access
    arrays of memory in running model
    
    Args:
        variable_type:  data type of array elements
        group:          group array belongs to
        
    """
    def __init__(self, variable_type: VarTypeType, group):
        # Make copy of type as e.g. a reference to model.precision would result in circular dependency
        self.type = copy(variable_type)
        self.group = group if type(group) in ProxyTypes else proxy(group)
        self._view = None
        self._array = None
    
    def set_array(self, array, view_shape=None):
        """Assign an array obtained from runtime to object
        
        Args:
            array:      array object obtained from runtime
            view_shape: shape to reshape array with
        """
        self._array = array
        
        # Get numpy data type corresponding to type
        model = self.group._model
        resolved_type = (self.type if isinstance(self.type, ResolvedType)
                         else self.type.resolve(model._type_context))
        dtype = model.genn_types[resolved_type]
        
        # Get dtype view of host memoryview
        self._view = np.asarray(array.host_view).view(dtype)
        assert not self._view.flags["OWNDATA"]

        # Reshape view if shape is provided
        if view_shape is not None:
            self._view = np.reshape(self._view, view_shape)

    def push_to_device(self):
        """Copy array from host to device"""
        self._array.push_to_device()

    def pull_from_device(self):
        """Copy array device to host"""
        self._array.pull_from_device()

    def _unload(self):
        self._view = None
        self._array = None


class Array(ArrayBase):
    """Array class used for exposing internal GeNN state"""

    @property
    def view(self) -> np.ndarray:
        """Memory view of array"""
        return self._view

class VariableBase(ArrayBase):
    """Base class for arrays used to expose GeNN variables"""

    def __init__(self, variable_name: str, variable_type: VarTypeType,
                 init_values, group):
        """
        Args:
            variable_name:  name of the variable
            variable_type:  data type of the variable
            init_values:    values to initialise variable with
            group:          group variable belongs to
        """
        super(VariableBase, self).__init__(variable_type, group)
        
        self.name = variable_name
        self.set_init_values(init_values)
    
    def set_array(self, array, view_shape, delay_group):
        """Assign an array obtained from runtime to object
        
        Args:
            array:          array object obtained from runtime
            view_shape:     shape to reshape array with
            delay_group:    neuron group which defines this array's delays
        """
    
        super(VariableBase, self).set_array(array, view_shape)
        self._delay_group = (None if delay_group is None 
                             else ref(delay_group))

    @deprecated("Please use set_init_values method instead",
                category=FutureWarning)
    def set_values(self, values):
        self.set_init_values(values)

    def set_init_values(self, init_values):
        """Set values variable is initialised with

        Args:
            init_values:    values to initialise variable with

        """
        # By default variable doesn't need initialising
        self.init_required = False

        # If an var initialiser is specified
        if isinstance(init_values, VarInit):
            # Build extra global parameters dictionary from var init snippet
            self.extra_global_params =\
                {egp.name: ExtraGlobalParameter(egp.name, egp.type, self.group)
                 for egp in init_values.snippet.get_extra_global_params()}
        # If no values are specified - mark as uninitialised
        elif init_values is None:
            self.extra_global_params = {}
        # Otherwise
        else:
            # Try and iterate values - if they are iterable
            # they must be loaded at simulate time
            try:
                iter(init_values)
                self.init_values = np.asarray(init_values)
                self.init_required = True
                self.extra_global_params = {}
            # Otherwise - they can be initialised on device as a scalar
            except TypeError:
                self.extra_global_params = {}

    def _unload(self):
        super(VariableBase, self)._unload()
        for e in self.extra_global_params.values():
            e._unload()

class Variable(VariableBase):
    """Array class used for exposing per-neuron GeNN variables"""

    @property
    def view(self) -> np.ndarray:
        """Memory view of entire variable. If variable is 
        delayed this will contain multiple delayed values."""
        return self._view
    
    @property
    def current_view(self) -> np.ndarray:
        """Memory view of variable's values written in last timestep"""
        # If there's no delay group, return full view
        if self._delay_group is None:
            return self._view
        # Otherwise
        else:
            # Get delay pointer associated with delay group
            runtime = self.group._model._runtime
            delay_ptr = runtime.get_delay_pointer(self._delay_group())
            
            # Slice current delay slot from appropriate axis
            num_view_dims = len(self._view.shape)
            if num_view_dims == 2:
                return self._view[delay_ptr,:]
            else:
                assert num_view_dims == 3
                return self._view[:,delay_ptr,:]

    @property
    def values(self) -> np.ndarray:
        """Copy of entire variable. If variable is 
        delayed this will contain multiple delayed values."""
        return np.copy(self._view)
    
    @values.setter
    def values(self, vals: np.ndarray):
        self._view[:] = vals

    @property
    def current_values(self) -> np.ndarray:
        """Copy of variable's values written in last timestep"""
        return np.copy(self.current_view)

class SynapseVariable(VariableBase):
    """Array class used for exposing per-synapse GeNN variables"""

    @property
    def view(self) -> np.ndarray:
        """Memory view of variable. This operation is not supported for
        variables associated with :attr:`SynapseMatrixConnectivity.SPARSE`
        connectivity.
        """
        sg = self.group.synapse_group
        if ((sg.matrix_type & SynapseMatrixConnectivity.DENSE) or
            (sg.matrix_type & SynapseMatrixWeight.KERNEL)):
            return self._view
        else:
            raise Exception("Only variables associated with DENSE or KERNEL "
                            "connectivity can be accessed without copying "
                            "via 'view'. Please use 'values' instead.")

    @property
    def current_view(self) -> np.ndarray:
        """Memory view of variable. This operation is not supported for
        variables associated with :attr:`SynapseMatrixConnectivity.SPARSE`
        connectivity.
        """
        return self.view

    @property
    def values(self) -> np.ndarray:
        """Copy of variable's values. Variables associated with 
        :attr:`SynapseMatrixConnectivity.SPARSE`"""
        sg = self.group.synapse_group
        if sg.matrix_type & SynapseMatrixConnectivity.DENSE:
            return np.copy(self._view)
        elif sg.matrix_type & SynapseMatrixWeight.KERNEL:
            return np.copy(self._view)
        elif sg.matrix_type & SynapseMatrixConnectivity.SPARSE:
            max_rl = sg.max_connections
            row_ls = (sg._row_lengths._view 
                      if sg._connectivity_initialiser_provided
                      else sg.row_lengths)

            # Create range containing the index where each row starts in ind
            row_start_idx = range(0, sg.weight_update_var_size, max_rl)

            # Build list of subviews representing each row
            if len(self._view.shape) == 1:
                rows = [self._view[i:i + r] 
                        for i, r in zip(row_start_idx, row_ls)]
            else:
                rows = [self._view[:,i:i + r] 
                        for i, r in zip(row_start_idx, row_ls)]
            
            # Stack all rows together into single array
            return np.hstack(rows)
        else:
            raise Exception("Matrix format not supported")
    
    @values.setter
    def values(self, vals: np.ndarray):
        # If connectivity is dense,
        # copy variables  directly into view
        # **NOTE** we assume order is row-major
        sg = self.group.synapse_group
        if ((sg.matrix_type & SynapseMatrixConnectivity.DENSE) or
            (sg.matrix_type & SynapseMatrixWeight.KERNEL)):
            self._view[:] = vals
        elif (sg.matrix_type & SynapseMatrixConnectivity.SPARSE):
            # Sort variable to match GeNN order
            if len(self._view.shape) == 1:
                sorted_var = vals[sg.synapse_order]
            else:
                sorted_var = vals[:,sg.synapse_order]

            # Create range containing the index
            # where each row starts in ind
            row_start_idx = range(0, sg.weight_update_var_size,
                                  sg.max_connections)

            # Loop through ragged matrix rows
            syn = 0
            for i, r in zip(row_start_idx, sg.row_lengths):
                # Copy row from non-padded indices into correct location
                if len(self._view.shape) == 1:
                    self._view[i:i + r] = sorted_var[syn:syn + r]
                else:
                    self._view[:,i:i + r] = sorted_var[:,syn:syn + r]
                syn += r
        else:
            raise Exception("Matrix format not supported")

    @property
    def current_values(self) -> np.ndarray:
        return self.values

class ExtraGlobalParameter(Array):
    """Array class used for exposing GeNN extra global parameters
    
    Args:
        variable_name:  name of the extra global parameter
        variable_type:  data type of the extra global parameter
        group:          group extra global parameter belongs to
        init_values:    values to initialise extra global parameter with
    """
    def __init__(self, variable_name: str, variable_type: VarTypeType,
                 group, init_values=None):
        super(ExtraGlobalParameter, self).__init__(variable_type, group)
        self.name = variable_name
        self.set_init_values(init_values)

    @deprecated("Please use set_init_values method instead",
                category=FutureWarning)
    def set_values(self, values):
        self.set_init_values(values)

    def set_init_values(self, init_values):
        """Set values extra global parameter is initialised with

        Args:
            init_values:    values to initialise extra global parameter with

        """
        if init_values is None:
            self.init_values = None
        else:
            # Try and iterate values
            try:
                iter(init_values)
                self.init_values = np.asarray(init_values)
            # Otherwise give an error
            except TypeError:
                raise ValueError("extra global variables can only be "
                                 "initialised with iterables")
    
    @property
    def view(self) -> np.ndarray:
        """Memory view of extra global parameter"""
        return self._view
    
    @property
    def values(self) -> np.ndarray:
        """Copy of extra global parameter values"""
        return np.copy(self._view)
    
    @values.setter
    def values(self, vals: np.ndarray):
        self._view[:] = vals

def _prepare_param_vals(params):
    return {n: NumericValue(v) for n, v in params.items()}

def _prepare_vars(vars, var_space, group, var_type=Variable):
    return {v.name: var_type(v.name, v.type, var_space[v.name], group)
            for v in vars}

def _prepare_egps(egps, group):
    return {e.name: ExtraGlobalParameter(e.name, e.type, group)
            for e in egps}

def _get_var_init(var_space):
    """ Build a dictionary of VarInit objects to specify if and how
    variables should be initialsed by GeNN
    
    Args:
    var_space       --  dict with model variables
    
    Returns:
    dict mapping variable names to VarInit objects
    """
    var_init = {}
    for name, value in var_space.items():
        if isinstance(value, VarInit):
            var_init[name] = value
         # If no values are specified - mark as uninitialised
        elif value is None:
            var_init[name] = VarInit(Uninitialised(), {})
        else:
            # Try and iterate value - if they are iterable
            # they must be loaded at simulate time
            try:
                iter(value)
                var_init[name] = VarInit(Uninitialised(), {})
            # Otherwise - they can be initialised on device as a scalar
            except TypeError:
                var_init[name] = VarInit(value)
    return var_init
            
    
def _get_snippet(snippet, snippet_base_class, built_in_snippet_module):
    """Check whether the model is valid, i.e is native or derived
    from model_family.Custom

    Args:
    model                   -- string or instance of pygenn.genn.SnippetBase
    snippet_base_class      -- if model is an instance, base class it SHOULD have 
    built_in_snippet_module -- if model is a string, module which should be searched
                               for built in snippet

    Returns:
    instance of the snippet and its type as string

    Raises:
    AttributeError  -- if snippet specified by name doesn't exist
    Exception       -- if something other than a string or object derived 
                       from snippet_base_class is provided
    """
    
    # If model is a string, get function with 
    # this name from module and call it
    if isinstance(snippet, str):
        return getattr(built_in_snippet_module, snippet)()
    # Otherwise, if model is derived off correct 
    # base class, return it directly
    elif isinstance(snippet, snippet_base_class):
        return snippet
    else:
        raise Exception("Invalid snippet")
## @namespace pygenn.model_preprocessor
"""
This module provides functions for model validation, parameter type conversions
and defines class Variable
"""
from copy import copy
from numbers import Number
from weakref import proxy, ProxyTypes
import numpy as np
from six import iteritems, iterkeys, itervalues, string_types

from .genn import ResolvedType, VarInit
from .init_var_snippets import Uninitialised

def prepare_model(model, group, var_space):
    """Prepare a model by checking its validity and extracting information
    about variables and parameters

    Args:
    model           --  instance of a class derived from pygenn.genn.ModelBase
    group           --  group model will belong to
    var_space       --  dict with model variables
    model_module    --  Module which should contain base class for models and functions to get built in models

    Returns:
    tuple consisting of (dict mapping names of variables to 
    instances of class Variable, dict mapping names of egps to class ExtraGlobalParameter)

    """
    vars = {vnt.name: Variable(vnt.name, vnt.type, var_space[vnt.name], group)
            for vnt in model.get_vars()}

    egps = {egp.name: ExtraGlobalParameter(egp.name, egp.type, group)
            for egp in model.get_extra_global_params()}

    return vars, egps

def get_var_init(var_space):
    """ Build a dictionary of VarInit objects to specify if and how
    variables should be initialsed by GeNN
    
    Args:
    var_space       --  dict with model variables
    
    Returns:
    dict mapping variable names to VarInit objects
    """
    var_init = {}
    for name, value in iteritems(var_space):
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
            
    
def get_snippet(snippet, snippet_base_class, built_in_snippet_module):
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
    if isinstance(snippet, string_types):
        return getattr(built_in_snippet_module, snippet)()
    # Otherwise, if model is derived off correct 
    # base class, return it directly
    elif isinstance(snippet, snippet_base_class):
        return snippet
    else:
        raise Exception("Invalid snippet")

class Array:
    def __init__(self, variable_type, group):
        # Make copy of type as e.g. a reference to model.precision would result in circular dependency
        self.type = copy(variable_type)
        self.group = group if type(group) in ProxyTypes else proxy(group)
        self._view = None
        self._array = None
    
    def set_array(self, array, view_shape=None):
        self._array = array
        
        # Get numpy data type corresponding to type
        model = self.group._model
        resolved_type = (self.type if isinstance(self.type, ResolvedType)
                         else self.type.resolve(model.type_context))
        dtype = model.genn_types[resolved_type]
        
        # Get dtype view of host memoryview
        self._view = np.asarray(array.host_view).view(dtype)
        assert not self._view.flags["OWNDATA"]

        # Reshape view if shape is provided
        if view_shape is not None:
            self._view = np.reshape(self._view, view_shape)

    def push_to_device(self):
        self._array.push_to_device()

    def pull_from_device(self):
        self._array.pull_from_device()

    def _unload(self):
        self._view = None
        self._array = None

    @property
    def view(self):
        return self._view

class Variable(Array):

    """Class holding information about GeNN variables"""

    def __init__(self, variable_name, variable_type, values, group):
        """Init Variable

        Args:
        variable_name   -- string name of the variable
        variable_type   -- string type of the variable
        values          -- iterable, single value or VarInit instance
        group           -- pygenn.genn_groups.Group this  
                           variable is associated with
        """
        super(Variable, self).__init__(variable_type, group)
        
        self.name = variable_name
        self.set_values(values)

    def set_extra_global_init_param(self, param_name, param_values):
        """Set values of extra global parameter associated with
        variable initialisation snippet

        Args
        param_name      -- string, name of parameter
        param_values    -- iterable or single value
        """
        self.extra_global_params[param_name].set_values(param_values)

    def set_values(self, values):
        """Set Variable's values

        Args:
        values -- iterable, single value or VarInit instance

        """
        # By default variable doesn't need initialising
        self.init_required = False

        # If an var initialiser is specified
        if isinstance(values, VarInit):
            # Build extra global parameters dictionary from var init snippet
            self.extra_global_params =\
                {egp.name: ExtraGlobalParameter(egp.name, egp.type, self.group)
                 for egp in values.snippet.get_extra_global_params()}
        # If no values are specified - mark as uninitialised
        elif values is None:
            self.extra_global_params = {}
        # Otherwise
        else:
            # Try and iterate values - if they are iterable
            # they must be loaded at simulate time
            try:
                iter(values)
                self.values = np.asarray(values)
                self.init_required = True
                self.extra_global_params = {}
            # Otherwise - they can be initialised on device as a scalar
            except TypeError:
                self.extra_global_params = {}

    def _unload(self):
        super(Variable, self)._unload()
        for e in itervalues(self.extra_global_params):
            e._unload()


class ExtraGlobalParameter(Array):

    """Class holding information about GeNN extra global parameter"""

    def __init__(self, variable_name, variable_type, group, values=None):
        """Init Variable

        Args:
        variable_name   --  string name of the variable
        variable_type   --  string type of the variable
        group           --  pygenn.genn_groups.Group this
                            variable is associated with

        Keyword args:
        values          --  iterable
        """
        super(ExtraGlobalParameter, self).__init__(variable_type, group)
        self.name = variable_name
        self.set_values(values)

    def set_values(self, values):
        """Set Variable's values

        Args:
        values -- iterable or single value
        """
        if values is None:
            self.values = None
        else:
            # Try and iterate values
            try:
                iter(values)
                self.values = np.asarray(values)
            # Otherwise give an error
            except TypeError:
                raise ValueError("extra global variables can only be "
                                 "initialised with iterables")

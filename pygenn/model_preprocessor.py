## @namespace pygenn.model_preprocessor
"""
This module provides functions for model validation, parameter type conversions
and defines class Variable
"""
from numbers import Number
from weakref import proxy, ProxyTypes
import numpy as np
from six import iterkeys, itervalues
from . import genn_wrapper
from .genn_wrapper.Models import (VarInit, VarReference, WUVarReference,
                                  VarInitVector, VarRefVector, 
                                  VarReferenceVector, WUVarReferenceVector)
from .genn_wrapper.StlContainers import DoubleVector

def prepare_model(model, group, param_space, var_space, model_family):
    """Prepare a model by checking its validity and extracting information
    about variables and parameters

    Args:
    model           --  string or instance of a class derived from pygenn.genn_wrapper.NeuronModels.Custom or pygenn.genn_wrapper.WeightUpdateModels.Custom or pygenn.genn_wrapper.CurrentSourceModels.Custom
    group           --  group model will belong to
    param_space     --  dict with model parameters
    var_space       --  dict with model variables
    var_ref_space   --  optional dict with (custom update) model
                        variable references
    model_family    --  pygenn.genn_wrapper.NeuronModels or pygenn.genn_wrapper.WeightUpdateModels or pygenn.genn_wrapper.CurrentSourceModels

    Returns:
    tuple consisting of (model instance, model type, model parameter names,
                         model parameters, list of variable names,
                         dict mapping names of variables to instances of class Variable)

    """
    m_instance, m_type = is_model_valid(model, model_family)
    param_names = list(m_instance.get_param_names())
    if set(iterkeys(param_space)) != set(param_names):
        raise ValueError("Invalid parameter values for {0}".format(
            model_family.__name__))
    params = param_space_to_vals(m_instance, param_space)

    var_names = [vnt.name for vnt in m_instance.get_vars()]
    if set(iterkeys(var_space)) != set(var_names):
        raise ValueError("Invalid variable initializers for {0}".format(
            model_family.__name__))
    vars = {vnt.name: Variable(vnt.name, vnt.type, var_space[vnt.name], group)
            for vnt in m_instance.get_vars()}

    egps = {egp.name: ExtraGlobalParameter(egp.name, egp.type, group)
            for egp in m_instance.get_extra_global_params()}

    return (m_instance, m_type, param_names, params,
            var_names, vars, egps)

def prepare_snippet(snippet, param_space, snippet_family):
    """Prepare a snippet by checking its validity and extracting
    information about parameters

    Args:
    snippet         --  string or instance of a class derived from pygenn.genn_wrapper.InitVarSnippet.Custom or pygenn.genn_wrapper.InitSparseConnectivitySnippet.Custom
    param_space     --  dict with model parameters
    snippet_family  --  pygenn.genn_wrapper.InitVarSnippet or pygenn.genn_wrapper.InitSparseConnectivitySnippet

    Returns:
    tuple consisting of (snippet instance, snippet type,
                         snippet parameter names, snippet parameters)
    """
    s_instance, s_type = is_model_valid(snippet, snippet_family)
    param_names = list(s_instance.get_param_names())
    if set(iterkeys(param_space)) != set(param_names):
        raise ValueError("Invalid parameter initializers for {0}".format(
            snippet_family.__name__))
    params = param_space_to_val_vec(s_instance, param_space)

    return (s_instance, s_type, param_names, params)


def is_model_valid(model, model_family):
    """Check whether the model is valid, i.e is native or derived
    from model_family.Custom

    Args:
    model           --  string or instance of model_family.Custom
    model_family    --  model family (NeuronModels, WeightUpdateModels or
                        PostsynapticModels) to which model should belong to

    Returns:
    instance of the model and its type as string

    Raises ValueError if model is not valid (i.e. is not custom and is
    not natively available)
    """

    if not isinstance(model, str):
        if not isinstance(model, model_family.Custom):
            model_type = type(model).__name__
            if not hasattr(model_family, model_type):
                raise ValueError("model '{0}' is not "
                                 "supported".format(model_type))
        else:
            model_type = "Custom"
    else:
        model_type = model
        if not hasattr(model_family, model_type):
            raise ValueError("model '{0}' is not supported".format(model_type))
        else:
            model = getattr(model_family, model_type).get_instance()
    return model, model_type


def param_space_to_vals(model, param_space):
    """Convert a param_space dict to ParamValues

    Args:
    model       --  instance of the model
    param_space --  dict with parameters

    Returns:
    native model's ParamValues
    """
    return model.make_param_values(param_space_to_val_vec(model, param_space))


def param_space_to_val_vec(model, param_space):
    """Convert a param_space dict to a std::vector<double>

    Args:
    model     -- instance of the model
    param_space -- dict with parameters

    Returns:
    native vector of parameters
    """
    if not all(isinstance(p, Number) for p in itervalues(param_space)):
        raise ValueError("non-numeric parameters are not supported")

    return DoubleVector([param_space[pn] for pn in model.get_param_names()])


def var_space_to_vals(model, var_space):
    """Convert a var_space dict to VarValues

    Args:
    model       -- instance of the model
    var_space   -- dict with Variables

    Returns:
    native model's VarValues
    """
    return model.make_var_values(VarInitVector([var_space[vnt.name].init_val
                                                for vnt in model.get_vars()]))

def var_ref_space_to_var_refs(model, var_ref_space):
    """Convert a var_ref_space dict to VarReferences

    Args:
    model           -- instance of the model
    var_ref_space   -- dict with variable references

    Returns:
    native model's VarValues
    """
    return model.make_var_references(
        VarReferenceVector([var_ref_space[v.name][0]
                            for v in model.get_var_refs()]))
                                                
def var_ref_space_to_wu_var_refs(model, var_ref_space):
    """Convert a var_ref_space dict to WUVarReferences

    Args:
    model       -- instance of the model
    var_space   -- dict with Variables

    Returns:
    native model's VarValues
    """
    return model.make_wuvar_references(
        WUVarReferenceVector([var_ref_space[v.name][0]
                              for v in model.get_var_refs()]))

def pre_var_space_to_vals(model, var_space):
    """Convert a var_space dict to PreVarValues

    Args:
    model       -- instance of the weight update model
    var_space   -- dict with Variables

    Returns:
    native model's VarValues
    """
    return model.make_pre_var_values(
        VarInitVector([var_space[vnt.name].init_val
                       for vnt in model.get_pre_vars()]))


def post_var_space_to_vals(model, var_space):
    """Convert a var_space dict to PostVarValues

    Args:
    model       -- instance of the weight update model
    var_space   -- dict with Variables

    Returns:
    native model's VarValues
    """
    return model.make_post_var_values(
        VarInitVector([var_space[vnt.name].init_val
                       for vnt in model.get_post_vars()]))


class Variable(object):

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
        self.name = variable_name
        self.type = variable_type
        self.group = proxy(group)
        self.view = None
        self.needs_allocation = False
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
            # Use it as initial value
            self.init_val = values

            # Build extra global parameters dictionary from var init snippet
            self.extra_global_params =\
                {egp.name: ExtraGlobalParameter(egp.name, egp.type, self.group)
                 for egp in self.init_val.get_snippet().get_extra_global_params()}
        # If no values are specified - mark as uninitialised
        elif values is None:
            self.init_val = genn_wrapper.uninitialised_var()
            self.extra_global_params = {}
        # Otherwise
        else:
            # Try and iterate values - if they are iterable
            # they must be loaded at simulate time
            try:
                iter(values)
                self.init_val = genn_wrapper.uninitialised_var()
                self.values = np.asarray(
                    values, dtype=self.group._model.genn_types[self.type].np_dtype)
                self.init_required = True
                self.extra_global_params = {}
            # Otherwise - they can be initialised on device as a scalar
            except TypeError:
                self.init_val = VarInit(values)
                self.extra_global_params = {}

class ExtraGlobalParameter(object):

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
        if variable_type[-1] == "*":
            self.is_scalar = False
            self.type = variable_type[:-1]
        else:
            self.is_scalar = True
            self.type = variable_type

        self.group = group if type(group) in ProxyTypes else proxy(group)
        self.name = variable_name
        self.view = None
        self.set_values(values)

    def set_values(self, values):
        """Set Variable's values

        Args:
        values -- iterable or single value
        """
        if values is None:
            self.values = None
        elif self.is_scalar:
            if isinstance(values, Number):
                self.values = values
            else:
                raise ValueError("scalar extra global variables can only be "
                                 "initialised with a number")
        else:
            # Try and iterate values
            try:
                iter(values)
                self.values = np.asarray(
                    values, dtype=self.group._model.genn_types[self.type].np_dtype)
            # Otherwise give an error
            except TypeError:
                raise ValueError("extra global variables can only be "
                                 "initialised with iterables")

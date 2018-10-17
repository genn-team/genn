"""Model preprocessor
This module provides functions for model validation and parameter type conversions
and defines class Variable
"""

import genn_wrapper
from genn_wrapper.NewModels import VarInit, VarInitVector
from genn_wrapper.StlContainers import DoubleVector

def prepare_model(model, param_space, var_space, pre_var_space=None, post_var_space=None,
                 model_family=None):
    """Prepare a model by checking its validity and extracting information about variables and parameters

    Args:
    model           -- string or instance of a class derived from model_family.Custom
    param_space      -- dict with model parameters
    var_space        -- dict with model variables
    pre_var_space     -- optional dict with (weight update) model presynaptic variables
    post_var_space    -- optional dict with (weight update) model postsynaptic variables
    model_family     -- genn_wrapper.NeuronModels or genn_wrapper.WeightUpdateModels or genn_wrapper.CurrentSourceModels

    Return: tuple consisting of
            0. model instance,
            1. model type,
            2. model parameter names,
            3. model parameters,
            5. list of variable names
            4. dict mapping names of variables to instances of class Variable.

    """
    m_instance, m_type = is_model_valid(model, model_family)
    param_names = list(m_instance.getParamNames())
    params = parameter_space_to_param_values(m_instance, param_space)
    var_names = [vnt[0] for vnt in m_instance.getVars()]
    var_dict = {vnt[0] : Variable(vnt[0], vnt[1], var_space[vnt[0]])
              for vnt in m_instance.getVars()}
    
    if model_family == genn_wrapper.WeightUpdateModels:
        pre_var_names = [vnt[0] for vnt in m_instance.getPreVars()]
        pre_var_dict = {vnt[0] : Variable(vnt[0], vnt[1], varSpace[vnt[0]])
                      for vnt in m_instance.getPreVars()}
        post_var_names = [vnt[0] for vnt in m_instance.getPostVars()]
        post_var_dict = {vnt[0] : Variable(vnt[0], vnt[1], varSpace[vnt[0]])
                       for vnt in m_instance.getPostVars()}
        return (m_instance, m_type, param_names, params, var_names, var_dict,
                pre_var_names, pre_var_dict, post_var_names, post_var_dict)
    else:
        return (m_instance, m_type, param_names, params, var_names, var_dict)


def prepare_snippet(snippet, param_space, snippet_family):
    """Prepare a snippet by checking its validity and extracting information about parameters

    Args:
    snippet         -- string or instance of a class derived from snippet_family.Custom
    param_space      -- dict with model parameters
    snippet_family   -- genn_wrapper.InitVarSnippet or genn_wrapper.InitSparseConnectivitySnippet

    Return: tuple consisting of
            0. snippet instance,
            1. snippet type,
            2. snippet parameter names,
            3. snippet parameters
    """
    s_instance, s_type = is_model_valid(snippet, snippet_family)
    param_names = list(s_instance.getParamNames())
    params = parameter_space_to_double_vector(s_instance, param_space)

    return (s_instance, s_type, param_names, params)


def is_model_valid(model, model_family):
    """Check whether the model is valid, i.e is native or derived from model_family.Custom
    Args:
    model -- string or instance of model_family.Custom
    model_family -- model family (NeuronModels, WeightUpdateModels or PostsynapticModels) to which model should belong to

    Return:
    instance of the model and its type as string

    Raises ValueError if model is not valid (i.e. is not custom and is not natively available)
    """

    if not isinstance(model, str):
        if not isinstance(model, model_family.Custom):
            model_type = type(model).__name__
            if not hasattr(model_family, model_type):
                raise ValueError("model '{0}' is not supported".format(model_type))
        else:
            model_type = "Custom"
    else:
        model_type = model
        if not hasattr(model_family, model_type):
            raise ValueError("model '{0}' is not supported".format(model_type))
        else:
            model = getattr(model_family, model_type).getInstance()
    return model, model_type

def parameter_space_to_param_values(model, param_space):
    """Convert a param_space dict to ParamValues

    Args:
    model     -- instance of the model
    param_space -- dict with parameters

    Return:
    native model's ParamValues
    """
    return model.makeParamValues(parameter_space_to_double_vector(model, param_space))

def parameter_space_to_double_vector(model, param_space):
    """Convert a param_space dict to a std::vector<double>

    Args:
    model     -- instance of the model
    param_space -- dict with parameters

    Return:
    native vector of parameters
    """
    return DoubleVector([param_space[pn] for pn in model.getParamNames()])

def var_space_to_var_values(model, var_space):
    """Convert a var_space dict to VarValues

    Args:
    model     -- instance of the model
    var_space  -- dict with Variables

    Return:
    native model's VarValues
    """
    return model.makeVarValues(
        VarInitVector([var_space[vnt[0]].init_val for vnt in model.getVars()]))

def pre_var_space_to_var_values(model, var_space):
    """Convert a var_space dict to PreVarValues

    Args:
    model     -- instance of the weight update model
    var_space  -- dict with Variables

    Return:
    native model's VarValues
    """
    return model.makePreVarValues(
        VarInitVector([var_space[vnt[0]].init_val for vnt in model.getPreVars()]))

def post_var_space_to_var_values(model, var_space):
    """Convert a var_space dict to PostVarValues

    Args:
    model     -- instance of the weight update model
    var_space  -- dict with Variables

    Return:
    native model's VarValues
    """
    return model.makePostVarValues(
        VarInitVector([var_space[vnt[0]].init_val for vnt in model.getPostVars()]))


class Variable(object):

    """Class holding information about GeNN variables"""

    def __init__(self, variable_name, variable_type, values=None):
        """Init Variable

        Args:
        variable_name -- string name of the variable
        variable_type -- string type of the variable

        Keyword args:
        values       -- iterable, single value or VarInit instance
        """
        self.name = variable_name
        self.type = variable_type
        self.view = None
        self.needs_allocation = False
        self.set_values(values)

    def set_values(self, values):
        """Set Variable's values

        Args:
        values -- iterable, single value or VarInit instance

        """
        # By default variable doesn't need initialising
        self.init_required = False

        # If an var initialiser is specified, set it directly
        if isinstance(values, VarInit):
            self.init_val = values
        # If no values are specified - mark as uninitialised
        elif values is None:
            self.init_val = genn_wrapper.uninitialisedVar()
        # Otherwise
        else:
            # Try and iterate values - if they are iterable they must be loaded at simulate time
            try:
                iter(values)
                self.init_val = genn_wrapper.uninitialisedVar()
                self.values = list(values)
                self.init_required = True
            # Otherwise - they can be initialised on device as a scalar
            except TypeError:
                self.init_val = VarInit(values)

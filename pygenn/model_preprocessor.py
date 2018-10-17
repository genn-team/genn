"""Model preprocessor
This module provides functions for model validation and parameter type conversions
and defines class Variable
"""

import genn_wrapper
from genn_wrapper.NewModels import VarInit, VarInitVector
from genn_wrapper.StlContainers import DoubleVector

def prepareModel(model, paramSpace, varSpace, preVarSpace=None, postVarSpace=None,
                 modelFamily=None):
    """Prepare a model by checking its validity and extracting information about variables and parameters

    Args:
    model           -- string or instance of a class derived from modelFamily.Custom
    paramSpace      -- dict with model parameters
    varSpace        -- dict with model variables
    preVarSpace     -- optional dict with (weight update) model presynaptic variables
    postVarSpace    -- optional dict with (weight update) model postsynaptic variables
    modelFamily     -- genn_wrapper.NeuronModels or genn_wrapper.WeightUpdateModels or genn_wrapper.CurrentSourceModels

    Return: tuple consisting of
            0. model instance,
            1. model type,
            2. model parameter names,
            3. model parameters,
            5. list of variable names
            4. dict mapping names of variables to instances of class Variable.

    """
    mInstance, mType = isModelValid(model, modelFamily)
    paramNames = list(mInstance.getParamNames())
    params = parameterSpaceToParamValues(mInstance, paramSpace)
    varNames = [vnt[0] for vnt in mInstance.getVars()]
    varDict = { vnt[0] : Variable(vnt[0], vnt[1], varSpace[vnt[0]])
              for vnt in mInstance.getVars() }
    
    if modelFamily == genn_wrapper.WeightUpdateModels:
        preVarNames = [vnt[0] for vnt in mInstance.getPreVars()]
        preVarDict = {vnt[0] : Variable(vnt[0], vnt[1], varSpace[vnt[0]])
                      for vnt in mInstance.getPreVars()}
        postVarNames = [vnt[0] for vnt in mInstance.getPostVars()]
        postVarDict = {vnt[0] : Variable(vnt[0], vnt[1], varSpace[vnt[0]])
                       for vnt in mInstance.getPostVars()}
        return (mInstance, mType, paramNames, params, varNames, varDict,
                preVarNames, preVarDict, postVarNames, postVarDict)
    else:
        return (mInstance, mType, paramNames, params, varNames, varDict)


def prepareSnippet(snippet, paramSpace, snippetFamily):
    """Prepare a snippet by checking its validity and extracting information about parameters

    Args:
    snippet         -- string or instance of a class derived from snippetFamily.Custom
    paramSpace      -- dict with model parameters
    snippetFamily   -- genn_wrapper.InitVarSnippet or genn_wrapper.InitSparseConnectivitySnippet

    Return: tuple consisting of
            0. snippet instance,
            1. snippet type,
            2. snippet parameter names,
            3. snippet parameters
    """
    sInstance, sType = isModelValid(snippet, snippetFamily)
    paramNames = list(sInstance.getParamNames())
    params = parameterSpaceToDoubleVector(sInstance, paramSpace)

    return (sInstance, sType, paramNames, params)


def isModelValid(model, modelFamily):
    """Check whether the model is valid, i.e is native or derived from modelFamily.Custom
    Args:
    model -- string or instance of modelFamily.Custom
    modelFamily -- model family (NeuronModels, WeightUpdateModels or PostsynapticModels) to which model should belong to

    Return:
    instance of the model and its type as string

    Raises ValueError if model is not valid (i.e. is not custom and is not natively available)
    """

    if not isinstance(model, str):
        if not isinstance(model, modelFamily.Custom):
            modelType = type(model).__name__
            if not hasattr(modelFamily, modelType):
                raise ValueError("model '{0}' is not supported".format(modelType))
        else:
            modelType = "Custom"
    else:
        modelType = model
        if not hasattr(modelFamily, modelType):
            raise ValueError("model '{0}' is not supported".format(modelType))
        else:
            model = getattr(modelFamily, modelType).getInstance()
    return model, modelType

def parameterSpaceToParamValues(model, paramSpace):
    """Convert a paramSpace dict to ParamValues

    Args:
    model     -- instance of the model
    paramSpace -- dict with parameters

    Return:
    native model's ParamValues
    """
    return model.makeParamValues(parameterSpaceToDoubleVector(model, paramSpace))

def parameterSpaceToDoubleVector(model, paramSpace):
    """Convert a paramSpace dict to a std::vector<double>

    Args:
    model     -- instance of the model
    paramSpace -- dict with parameters

    Return:
    native vector of parameters
    """
    return DoubleVector([paramSpace[pn] for pn in model.getParamNames()])

def varSpaceToVarValues(model, varSpace):
    """Convert a varSpace dict to VarValues

    Args:
    model     -- instance of the model
    varSpace  -- dict with Variables

    Return:
    native model's VarValues
    """
    return model.makeVarValues(
        VarInitVector([varSpace[vnt[0]].initVal for vnt in model.getVars()]))

def preVarSpaceToVarValues(model, varSpace):
    """Convert a varSpace dict to PreVarValues

    Args:
    model     -- instance of the weight update model
    varSpace  -- dict with Variables

    Return:
    native model's VarValues
    """
    return model.makePreVarValues(
        VarInitVector([varSpace[vnt[0]].initVal for vnt in model.getPreVars()]))

def postVarSpaceToVarValues(model, varSpace):
    """Convert a varSpace dict to PostVarValues

    Args:
    model     -- instance of the weight update model
    varSpace  -- dict with Variables

    Return:
    native model's VarValues
    """
    return model.makePostVarValues(
        VarInitVector([varSpace[vnt[0]].initVal for vnt in model.getPostVars()]))


class Variable(object):

    """Class holding information about GeNN variables"""

    def __init__(self, variableName, variableType, values=None):
        """Init Variable

        Args:
        variableName -- string name of the variable
        variableType -- string type of the variable

        Keyword args:
        values       -- iterable, single value or VarInit instance
        """
        self.name = variableName
        self.type = variableType
        self.view = None
        self.needsAllocation = False
        self.setValues(values)

    def setValues(self, values):
        """Set Variable's values

        Args:
        values -- iterable, single value or VarInit instance

        """
        # By default variable doesn't need initialising
        self.initRequired = False

        # If an var initialiser is specified, set it directly
        if isinstance(values, VarInit):
            self.initVal = values
        # If no values are specified - mark as uninitialised
        elif values is None:
            self.initVal = genn_wrapper.uninitialisedVar()
        # Otherwise
        else:
            # Try and iterate values - if they are iterable they must be loaded at simulate time
            try:
                iter(values)
                self.initVal = genn_wrapper.uninitialisedVar()
                self.values = list(values)
                self.initRequired = True
            # Otherwise - they can be initialised on device as a scalar
            except TypeError:
                self.initVal = VarInit(values)

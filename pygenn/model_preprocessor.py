"""Model preprocessor
This module provides functions for model validation and parameter type conversions
and defines class Variable
"""

import genn_wrapper
from NewModels import VarInit, VarInitVector
from StlContainers import DoubleVector

def prepareModel( model, paramSpace, varSpace, modelFamily ):
    """Prepare a model by checking its validity and extracting information about variables and parameters

    Args:
    model -- string or instance of a class derived from modelFamily.Custom
    paramSpace  -- dict with model parameters
    varSpace    -- dict with model variables
    modelFamily -- pygenn.NeuronModels or pygenn.WeightUpdateModels or pygenn.CurrentSourceModels

    Return: tuple consisting of
            0. model instance,
            1. model type,
            2. model parameter names,
            3. model parameters,
            4. dict mapping names of variables to instances of class Variable.

    """
    mInstance, mType = isModelValid( model, modelFamily )
    paramNames = list( mInstance.getParamNames() )
    params = parameterSpaceToParamValues( mInstance, paramSpace )
    varNames = [vnt[0] for vnt in mInstance.getVars()]
    varDict = { vnt[0] : Variable( vnt[0], vnt[1], varSpace[vnt[0]] )
              for vnt in mInstance.getVars() }
    return ( mInstance, mType, paramNames, params, varNames, varDict )

def isModelValid( model, modelFamily ):
    """Check whether the model is valid, i.e is native or derived from modelFamily.Custom
    Args:
    model -- string or instance of modelFamily.Custom
    modelFamily -- model family (NeuronModels, WeightUpdateModels or PostsynapticModels) to which model should belong to

    Return:
    instance of the model and its type as string

    Raises ValueError if model is not valid (i.e. is not custom and is not natively available)
    """

    if not isinstance( model, str ):
        if not isinstance( model, modelFamily.Custom ):
            modelType = type( model ).__name__
            if not hasattr( modelFamily, modelType ):
                raise ValueError( 'model "{0}" is not supported'.format( modelType ) )
        else:
            modelType = 'Custom'
    else:
        modelType = model
        if not hasattr( modelFamily, modelType ):
            raise ValueError( 'model "{0}" is not supported'.format( modelType ) )
        else:
            model = getattr( modelFamily, modelType ).getInstance()
    return model, modelType

def parameterSpaceToParamValues( model, paramSpace ):
    """Convert a paramSpace dict to ParamValues

    Args:
    model     -- instance of the model
    paramSpace -- dict with parameters

    Return:
    native model's ParamValues
    """
    paramVals = [paramSpace[pn] for pn in model.getParamNames()]

    return model.makeParamValues( DoubleVector( paramVals ) )

def varSpaceToVarValues( model, varSpace ):
    """Convert a varSpace dict to VarValues

    Args:
    model     -- instance of the model
    varSpace  -- dict with Variables

    Return:
    native model's VarValues
    """
    varVals = [varSpace[vnt[0]].initVal for vnt in model.getVars()]

    return model.makeVarValues( VarInitVector( varVals ) )


class Variable(object):

    """Class holding information about GeNN variables"""

    def __init__(self, variableName, variableType, values=None):
        """Init Variable

        Args:
        variableName -- string name of the variable
        variableType -- string type of the variable

        Keyword args:
        values       -- iterable or sigle value.
        """
        self.name = variableName
        self.type = variableType
        self.view = None
        self.needsAllocation = False
        self.setValues( values )

    def setInitVar( self, initVarSnippet, paramSpace ):
        """Set variable initialization using InitVarSnippet

        Args:
        initVarSnippet -- type as string or instance of a class derived from InitVarSnippet.Custom
        paramSpace     -- dict mapping parameter names to their values for InitVarSnippet
        """
        ivsInst, ivsType = isModelValid( initVarSnippet, genn_wrapper.InitVarSnippet )
        params = parameterSpaceToParamValues( ivsInst, paramSpace )
        initFct = getattr( genn_wrapper, 'initVar_' + ivsType )
        self.initVal = initFct( params )

    def setValues( self, values, initVar=None ):
        """Set Variable's values

        Args:
        values -- iterable or single value or parameter space is initVar is specified

        Keyword args:
        initVar -- type as string or instance of a class derived from InitVarSnippet.Custom
        """
        self.initRequired = False
        if initVar is not None:
            self.setInitVar( initVar, values )
        elif values is None:
            self.initVal = genn_wrapper.uninitialisedVar()
        else:
            try:
                iter( values )
                self.initVal = genn_wrapper.uninitialisedVar()
                self.values = list( values )
                self.initRequired = True
            except TypeError:
                self.initVal = VarInit( values )

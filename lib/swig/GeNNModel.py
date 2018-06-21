import libgenn as lg
import SharedLibraryModel as slm
from os import path
from subprocess import check_call
import numpy as np
import json

class GeNNModel( object ):

    """GeNNModel class
    This class helps to define, build and run a GeNN model from python
    """

    def __init__(self, scalar, modelName = None):
        """Init GeNNModel
        Args:
        scalar    -- type of scalars as string ("float" or "double")
        Keyword args:
        modelName -- name of the model
        """
        self._scalar = scalar
        if scalar == 'float':
            self._slm = slm.SharedLibraryModel_f()
            self._npType = np.float32
        elif scalar == 'double':
            self._slm = slm.SharedLibraryModel_d()
            self._npType = np.float64
        else:
            raise ValueError( 'unknown scalar type "{0}"'.format( scalar ) )
        
        self._built = False
        self._localhost = lg.init_cuda_mpi()
        
        lg.setDefaultVarMode( lg.VarMode_LOC_HOST_DEVICE_INIT_DEVICE )
        lg.initGeNN()
        self._model = lg.NNmodel()
        self._modelName = None
        if modelName is not None:
            self.modelName = modelName
        self.neuronPopulations = {}
        self.synapsePopulations = {}
        self._supportedNeurons = list(lg.NeuronModels.getSupportedNeurons())
        self._supportedPostsyn = list(lg.PostsynapticModels.getSupportedPostsyn())
        self._supportedWUpdate = list(lg.WeightUpdateModels.getSupportedWUpdate())
        self._dT = 1.0

    @property
    def modelName( self ):
        """Name of the model"""
        return self._modelName

    @modelName.setter
    def modelName( self, modelName ):
        if self._built:
            raise Exception("GeNN model already built")
        self._modelName = modelName
        self._model.setName( modelName )

    @property
    def dT( self ):
        """Step sise"""
        return self._dT

    @dT.setter 
    def dT( self, dt ):
        if self._built:
            raise Exception("GeNN model already built")
        self._dT = dt
        self._model.setDT( dt )

    
    def addNeuronPopulation(self, popName, numNeurons, neuron, paramSpace, varSpace, customNeuron=False):
        """Add a neuron population to the GeNN model

        Args:
        popName     -- name of the new population
        numNeurons  -- number of neurons in the new population
        neuron      -- type of the NeuronModels class as string or instance of neuron class
                        derived from NeuronModels::Custom class if customNeuron is True;
                        sa createCustomNeuronClass
        paramSpace  -- dict with param values for the NeuronModels class
        varSpace    -- dict with initial variable values for the NeuronModels class

        Keyword args:
        customNeuron -- boolean which indicates whether a custom neuron model is used. False by default
        """
        if self._built:
            raise Exception("GeNN model already built")
        if popName in self.neuronPopulations:
            raise ValueError( 'neuron population "{0}" already exists'.format( popName ) )
        
        
        # check whether the neuron is valid
        if not isinstance( neuron, str ):
            if not isinstance( neuron, lg.NeuronModels.Custom ):
                neuronType = type( neuron ).__name__
                if neuronType not in self._supportedNeurons:
                    raise ValueError( 'neuron model "{0}" is not supported'.format( neuronType ) )
            else:
                neuronType = 'Custom'
        else:
            neuronType = neuron
            if neuronType not in self._supportedNeurons:
                raise ValueError( 'neuron model "{0}" is not supported'.format( neuronType ) )
            else:
                neuron = eval( 'lg.NeuronModels.' + neuron + '()' )

        params = parameterSpaceToParamValues( neuron, paramSpace )
        varIni = varSpaceToVarValues( neuron, varSpace )

        # add neuron population to the GeNN model
        eval( 'self._model.addNeuronPopulation_' + \
                neuronType + \
                '( popName, numNeurons, neuron, params, varIni )' ) 

        # save neuron population info
        tmpPopDescr = {
                'NI' : neuron,
                'nN' : numNeurons,
                'nParams' : len( paramSpace ),
                'vars' : {},
                'spk' : None,
                'spkCnt' : None
        }

        nameTypes = tmpPopDescr['NI'].getVars()
        for i in range( nameTypes.size() ):
            tmpPopDescr['vars'][nameTypes[i][0]] = None

        self.neuronPopulations[popName] = tmpPopDescr

    
    def addSynapsePopulation(self, popName, matrixType, delaySteps, source, target,
                wUpdateModel, wuParamSpace, wuVarSpace,
                postsynModel, postsynParamSpace, postsynVarSpace,
                customWeightUpdate=False, customPostsynaptic=False):
        if self._built:
            raise Exception("GeNN model already built")

        if popName in self.synapsePopulations:
            raise ValueError( 'synapse population "{0}" already exists'.format( popName ) )

        # check whether the wUpdateModel is valid
        if not isinstance( wUpdateModel, str ):
            if not isinstance( wUpdateModel, lg.WeightUpdateModels.Custom ):
                wUpdateType = type( wUpdateModel ).__name__
                if wUpdateType not in self._supportedWUpdate:
                    raise ValueError( 'weightUpdate model "{0}" is not supported'.format( wUpdateType ) )
            else:
                wUpdateType = 'Custom'
        else:
            wUpdateType = wUpdateModel
            if wUpdateType not in self._supportedWUpdate:
                raise ValueError( 'weightUpdate model "{0}" is not supported'.format( wUpdateType ) )
            else:
                wUpdateModel = eval( 'lg.WeightUpdateModels.' + wUpdateType + '()' )

        # check whether the postsynModel is valid
        if not isinstance( postsynModel, str ):
            if not isinstance( postsynModel, lg.PostsynapticModels.Custom ):
                postsynType = type( postsynModel ).__name__
                if postsynType not in self._supportedPostsyn:
                    raise ValueError( 'postsynaptic model "{0}" is not supported'.format( postsynType ) )
            else:
                postsynType = 'Custom'
        else:
            postsynType = postsynModel
            if postsynType not in self._supportedPostsyn:
                raise ValueError( 'Postsynaptic model "{0}" is not supported'.format( postsynType ) )
            else:
                postsynModel = eval( 'lg.PostsynapticModels.' + postsynType + '()' )
        
        """Add a synapse population to the GeNN model

        Args:
        popName      -- name of the new population
        matrixType   -- type of the matrix as string
        delaySteps   -- delay in number of steps
        source       -- name of the source population
        target       -- name of the target population
        wUptateModel -- type of the WeightUpdateModels class as string or instance of weight update model class
                            derived from WeightUpdateModels::Custom class if customWeightUpdate is True;
                            sa createCustomWeightUpdateClass
        wuParamValues -- list with param values for the WeightUpdateModels class
        wuInitVarValues -- list with initial variable values for the WeightUpdateModels class
        postsynModel -- type of the PostsynapticModels class as string or instance of postsynaptic model class
                            derived from PostsynapticModels::Custom class if customPostsynaptic is True;
                            sa createCustomPostsynapticClass
        postsynParamValues -- list with param values for the PostsynapticModels class
        postsynInitVarValues -- list with initial variable values for the PostsynapticModels class

        Keyword args:
        customWeightUpdate -- boolean which indicates whether a custom weight update model is used. False by default
        customPostsynaptic -- boolean which indicates whether a custom postsynaptic model is used. False by default
        """

        ps_params = parameterSpaceToParamValues( postsynModel, postsynParamSpace )
        ps_varIni = varSpaceToVarValues( postsynModel, postsynVarSpace )

        wu_params = parameterSpaceToParamValues( wUpdateModel, wuParamSpace )
        wu_varIni = varSpaceToVarValues( wUpdateModel, wuVarSpace )

        # convert matrix type
        mType = eval( "lg.SynapseMatrixType_" + matrixType )

        
        # add synaptic population to the GeNN model
        eval( 'self._model.addSynapsePopulation_' + \
                wUpdateType + '_' + postsynType + \
                '( popName, mType, delaySteps, source, target, ' + \
                'wUpdateModel, wu_params, wu_varIni, ' + \
                'postsynModel, ps_params, ps_varIni )' )

        # save synaptic population info
        tmpPopDescr = {
                'WUI' : wUpdateModel,
                'PSI' : postsynModel,
                'nN'  : self.neuronPopulations[source]['nN'] * \
                    self.neuronPopulations[target]['nN'],
                'src' : source,
                'trg' : target,
                'delay' : delaySteps,
                'mType' : matrixType,
                'vars' : {}
        }
        
        nameTypes = tmpPopDescr['WUI'].getVars()
        for i in range( nameTypes.size() ):
            tmpPopDescr['vars'][nameTypes[i][0]] = None
        
        nameTypes = tmpPopDescr['PSI'].getVars()
        for i in range( nameTypes.size() ):
            tmpPopDescr['vars'][nameTypes[i][0]] = None
        
        self.synapsePopulations[popName] = tmpPopDescr


    def initializeVarOnDevice( self, popName, varName, mask, vals ):
        """Set values for the given variable and population and push them to the device
        
        Note: shapes of mask and vals must be the same

        Args:
        popName -- name of the population for which the values must be set
        varName -- name of the variable for which the values must be set
        mask    -- list with neurons ids
        vals    -- list with variable values
        """
        if popName not in self.neuronPopulations:
            if popName not in self.synapsePopulations:
                raise ValueError( 'Failed to initialize variable "{0}": \
                        population "{1}" does not exist'.format( varName, popName ) )
            else:
                var = self.synapsePopulations[popName]['vars'][varName]
        else:
            var = self.neuronPopulations[popName]['vars'][varName]

        var[mask] = vals
        self._slm.pushPopulationStateToDevice( popName )


    def initializeSpikesOnDevice( self, popName, mask, targets, counts ):
        """Set spike counts and targets for the given population and push them to the device
        
        Note: shapes of mask, targets and counts must be the same

        Args:
        popName -- name of the population for which the spikes must be set
        mask    -- list with source neurons ids
        targets -- list with target neuron ids
        counts  -- list with number of spikes for source neurons
        """
        if popName not in self.neuronPopulations:
            raise ValueError( 'Failed to initialize variable "{0}": \
                    population "{1}" does not exist'.format( varName, popName ) )
        self.neuronPopulations[popName]['spk'][mask] = targets
        self.neuronPopulations[popName]['spkCnt'][mask] = counts
        self._slm.pushPopulationSpikesToDevice( popName )


    def build( self, pathToModel = "./" ):

        """Finalize and build a GeNN model
        
        Keyword args:
        pathToModel -- path where to place the generated model code. Defaults to the local directory.
        """

        if self._built:
            raise Exception("GeNN model already built")
        self._pathToModel = pathToModel 
        self._model.finalize()
        lg.chooseDevice(self._model, pathToModel, self._localhost)
        lg.finalize_model_runner_generation(self._model, self._pathToModel, self._localhost)

        check_call( ['make', '-C', path.join( pathToModel, self.modelName + '_CODE' ) ] )
        
        self._built = True

    def loadExistingModel( self, pathToModel, neuronPop, synapsePop ):
        """import the existing model as shared library and initialize it
        Args:
        pathToModel -- path to the model
        neuronPop   -- dictionary with neuron populations. Each population must have:
                        'nN' - number of neurons
                        'vars' - dictionary with variable names. The values of the dictionary will
                                be overwritten with actuals variable views when the model is loaded
                        'spk' - view into internal spikes variable. Can be None. This value will be
                                overwritten with actual variable view when the model is loaded.
                        'spkCnt' - view into internal spike count variable. Can be None. This value will be
                                   overwritten with actual variable view when the model is loaded.

                        Example: {'Pop1' : {'nN' : 10,
                                            'vars' : { 'varName1': None },
                                            'spk' : None,
                                            'spkCnt' : None }
                                 }
                        
        synapsePop  -- dictionary with synapse populations. Each population must have: 'nN', 'vars'. sa neuronPop.
        """

        self.neuronPopulations = neuronPop
        self.synapsePopulations = synapsePop
        self._pathToModel = pathToModel
        self._built = True
        self.load()
       
    def load( self ):
        """import the model as shared library and initialize it"""
        if not self._built:
            raise Exception( "GeNN model has to be built before running" )
        
        self._slm.open( self._pathToModel, self.modelName )

        self._slm.allocateMem()
        self._slm.initialize()

        self.timestep = self._slm.assignExternalPointerToTimestep()
        self.T = self._slm.assignExternalPointerToT()
        
        for popName, popData in self.neuronPopulations.items():
            self._slm.initNeuronPopIO( popName, popData['nN'] )
            popData['spk'] = self._slm.assignExternalPointerToSpikes( popName, popData['nN'], False )
            popData['spkCnt'] = self._slm.assignExternalPointerToSpikes( popName, popData['nN'], True )
            for varName, varData in popData['vars'].items():
                popData['vars'][varName] = self._slm.assignExternalPointerToVar( popName, popData['nN'], varName )

        for popName, popData in self.synapsePopulations.items():
            
            self._slm.initSynapsePopIO( popName, popData['nN'] )
            #  popData['spk'] = self._slm.assignExternalPointerToSpikes( popName, popData['nN'], False )
            #  popData['spkCnt'] = self._slm.assignExternalPointerToSpikes( popName, popData['nN'], True )
            for varName, varData in popData['vars'].items():
                popData['vars'][varName] = self._slm.assignExternalPointerToVar( popName, popData['nN'], varName )


    def exportModelSpec( self, path='./' ):
        def serialize( obj ):
            if not isinstance( obj, str ) and not isinstance(obj, dict):
                return obj.__dict__
            return obj

        print( json.dumps( {'nPop':self.neuronPopulations, 'sPop':self.synapsePopulations},
            default=serialize) )



    def stepTimeGPU( self ):
        """Make one simulation step"""
        if not self._built:
            raise Exception( "GeNN model has to be built before running" )
        self._slm.stepTimeGPU()
    
    def stepTimeCPU( self ):
        """Make one simulation step"""
        if not self._built:
            raise Exception( "GeNN model has to be built before running" )
        self._slm.stepTimeCPU()

    def pullPopulationStateFromDevice( self, popName ):
        """Pull state from the device for a given population"""
        if not self._built:
            raise Exception( "GeNN model has to be built before running" )
        self._slm.pullPopulationStateFromDevice( popName )
    
    def pullPopulationSpikesFromDevice( self, popName ):
        """Pull spikes from the device for a given population"""
        if not self._built:
            raise Exception( "GeNN model has to be built before running" )
        self._slm.pullPopulationSpikesFromDevice( popName )

def makeCustomParamValues( params ):
    """This helper function converts list with params values to CustomParamValues class"""
    return lg.NewModels.CustomParamValues( lg.DoubleVector( params ) )

def makeCustomVarValues( varVals ):
    """This helper function converts list with variable values to CustomVarValues class"""
    return lg.NewModels.CustomVarValues( lg.DoubleVector( varVals ) )

def parameterSpaceToParamValues( model, paramSpace ):
    """Convert a paramSpace dict to ParamValues
    
    Args:
    mode       -- GeNN Neuron-, Postsynaptic-, or WeightUpdateModel
    paramSpace -- dict with parameters
    """
    paramNames = list( model.getParamNames() )
    paramVals = []
    for pn in paramNames:
        paramVals.append( paramSpace[pn] )

    return model.make_ParamValues( lg.DoubleVector( paramVals ) )

def varSpaceToVarValues( model, varSpace ):
    """Convert a iniSpace dict to VarValues
    
    Args:
    model     -- GeNN Neuron-, Postsynaptic-, or WeightUpdateModel
    varSpace  -- dict with initial values
    """
    varNameTypes = list( model.getVars() )
    varVals = []
    for vnt in varNameTypes:
        varVals.append( varSpace[vnt[0]] )

    return model.make_VarValues( lg.DoubleVector( varVals ) )


def createCustomNeuronClass( 
        className,
        paramNames=None,
        varNameTypes=None,
        derivedParams=None,
        simCode=None,
        thresholdConditionCode=None,
        resetCode=None,
        supportCode=None,
        extraGlobalParams=None,
        additionalInputVars=None,
        isPoisson=None,
        custom_body=None ):

    """This helper function creates a custom NeuronModel class.
    
    sa createCustomNeuronClass
    sa createCustomWeightUpdateClass

    Args:
    className     -- name of the new class

    Keyword args:
    paramNames    -- list of strings with param names of the model
    varNameTypes  -- list of pairs of strings with varible names and types of the model
    derivedParams -- list of pairs, where the first member is string with name of
                        the derived parameter and the second MUST be an instance of the class
                        which inherits from libgenn.Snippet.DerivedParamFunc
    simCode       -- string with the simulation code
    thresholdConditionCode -- string with the threshold condition code
    resetCode     -- string with the reset code
    supportCode   -- string with the support code
    extraGlobalParams -- list of pairs of strings with names and types of additional parameters
    additionalInputVars -- list of tuples with names and types as strings and
                            initial values of additional local input variables
    isPoisson     -- boolean, is this neuron model the internal Poisson model?

    custom_body   -- dictionary with additional attributes and methods of the new class
    """
    if not isinstance( custom_body, dict ) and custom_body is not None:
        raise ValueError( "custom_body must be an isinstance of dict or None" )

    body = {}

    if simCode is not None:
        body['getSimCode'] = lambda self: simCode
    
    if thresholdConditionCode is not None:
        body['getThresholdConditionCode'] = lambda self: thresholdConditionCode

    if resetCode is not None:
        body['getResetCode'] = lambda self: resetCode

    if supportCode is not None:
        body['getSupportCode'] = lambda self: supportCode
    
    if extraGlobalParams is not None:
        body['getxtraGlobalParams'] = lambda self: lg.StringPairVector(
                [lg.StringPair( egp[0], egp[1] ) for egp in extraGlobalParams] )
 
    if additionalInputVars:
        body['getAdditionalInputVars'] = lambda self: lg.StringStringDoublePairPairVector(
                [lg.StringStringDoublePairPair( aiv[0], lg.StringDoublePair( aiv[1], aiv[2] ) ) \
                        for aiv in additionalInputVars] )

    if isPoisson is not None:
        body['isPoisson'] = lambda self: isPoisson
    
    if custom_body is not None:
        body.update( custom_body )

    return createCustomModelClass(
            className,
            lg.NeuronModels.Custom,
            paramNames,
            varNameTypes,
            derivedParams,
            body )

def createCustomPostsynapticClass( 
        className,
        paramNames=None,
        varNameTypes=None,
        derivedParams=None,
        decayCode=None,
        applyInputCode=None,
        supportCode=None,
        custom_body=None ):

    """This helper function creates a custom PostsynapticModel class.
    
    sa createCustomNeuronClass
    sa createCustomWeightUpdateClass

    Args:
    className      -- name of the new class

    Keyword args:
    paramNames     -- list of strings with param names of the model
    varNameTypes   -- list of pairs of strings with varible names and types of the model
    derivedParams  -- list of pairs, where the first member is string with name of
                        the derived parameter and the second MUST be an instance of the class
                        which inherits from libgenn.Snippet.DerivedParamFunc
    decayCode      -- string with the decay code
    applyInputCode -- string with the apply input code
    supportCode    -- string with the support code

    custom_body    -- dictionary with additional attributes and methods of the new class
    """
    if not isinstance( custom_body, dict ) and custom_body is not None:
        raise ValueError()

    body = {}

    if decayCode is not None:
        body['getDecayCode'] = lambda self: decayCode
    
    if applyInputCode is not None:
        body['getApplyInputCode'] = lambda self: applyInputCode

    if supportCode is not None:
        body['getSupportCode'] = lambda self: supportCode
    
    if custom_body is not None:
        body.update( custom_body )

    return createCustomModelClass(
            className,
            lg.PostsynapticModels.Custom,
            paramNames,
            varNameTypes,
            derivedParams,
            body )


def createCustomWeightUpdateClass( 
        className,
        paramNames=None,
        varNameTypes=None,
        derivedParams=None,
        simCode=None,
        eventCode=None,
        learnPostCode=None,
        synapseDynamicsCode=None,
        eventThresholdConditionCode=None,
        simSupportCode=None,
        learnPostSupportCode=None,
        synapseDynamicsSuppportCode=None,
        extraGlobalParams=None,
        isPreSpikeTimeRequired=None,
        isPostSpikeTimeRequired=None,
        custom_body=None ):

    """This helper function creates a custom WeightUpdateModel class.
    
    sa createCustomNeuronClass
    sa createCustomPostsynapticClass

    Args:
    className     -- name of the new class

    Keyword args:
    paramNames    -- list of strings with param names of the model
    varNameTypes  -- list of pairs of strings with varible names and types of the model
    derivedParams -- list of pairs, where the first member is string with name of
                        the derived parameter and the second MUST be an instance of the class
                        which inherits from libgenn.Snippet.DerivedParamFunc
    simCode       -- string with the simulation code
    eventCode     -- string with the event code
    learnPostCode -- string with the code to include in learnSynapsePost kernel/function
    synapseDynamicsCode -- string with the synapse dynamics code
    eventThresholdConditionCode -- string with the event threshold condition code
    simSupportCode -- string with simulation support code
    learnPostSupportCode -- string with support code for learnSynapsePost kernel/function
    synapseDynamicsSuppportCode -- string with synapse dynamics support code
    extraGlobalParams -- list of pairs of strings with names and types of additional parameters
    isPreSpikeTimeRequired -- boolean, is presynaptic spike time required?
    isPostSpikeTimeRequired -- boolean, is postsynaptic spike time required?

    custom_body   -- dictionary with additional attributes and methods of the new class
    """
    if not isinstance( custom_body, dict ) and custom_body is not None:
        raise ValueError( "custom_body must be an isinstance of dict or None" )

    body = {}

    if simCode is not None:
        body['getSimCode'] = lambda self: simCode

    if eventCode is not None:
        body['getEventCode'] = lambda self: eventCode
    
    if learnPostCode is not None:
        body['getLearnPostCode'] = lambda self: learnPostCode

    if synapseDynamicsCode is not None:
        body['getSynapseDynamicsCode'] = lambda self: synapseDynamicsCode

    if eventThresholdConditionCode is not None:
        body['getEventThresholdConditionCode'] = lambda self: eventThresholdConditionCode

    if simSupportCode is not None:
        body['getSimSupportCode'] = lambda self: simSupportCode

    if learnPostSupportCode is not None:
        body['getLearnPostSupportCode'] = lambda self: learnPostSupportCode

    if synapseDynamicsSuppportCode is not None:
        body['getSynapseDynamicsSuppportCode'] = lambda self: synapseDynamicsSuppportCode

    if extraGlobalParams is not None:
        body['getExtraGlobalParams'] = lambda self: lg.StringPairVector(
                [lg.StringPair( egp[0], egp[1] ) for egp in extraGlobalParams] )

    if isPreSpikeTimeRequired is not None:
        body['isPreSpikeTimeRequired'] = lambda self: isPreSpikeTimeRequired

    if isPostSpikeTimeRequired is not None:
        body['isPostSpikeTimeRequired'] = lambda self: isPostSpikeTimeRequired
    
    if custom_body is not None:
        body.update( custom_body )
    
    return createCustomModelClass(
            className,
            lg.WeightUpdateModels.Custom,
            paramNames,
            varNameTypes,
            derivedParams,
            body )

def createDPFClass( dpfunc ):

    """Helper function to create derived parameter function class

    Args:
    dpfunc -- a function which computes the derived parameter and takes 
                two args "pars" (vector of double) and "dt" (double)
    """

    def ctor( self ):
        lg.Snippet.DerivedParamFunc.__init__( self )

    def call( self, pars, dt ):
        return dpfunc( pars, dt )

    return type( '', ( lg.Snippet.DerivedParamFunc, ), {'__init__' : ctor, '__call__' : call} )

def createCustomModelClass(
        className,
        base,
        paramNames,
        varNameTypes,
        derivedParams,
        custom_body ):

    """This helper function completes a custom model class creation.
    This part is common for all model classes and is nearly useless on its own
    unless you specify custom_body.
    sa createCustomNeuronClass
    sa createCustomWeightUpdateClass
    sa createCustomPostsynapticClass

    Args:
    className     -- name of the new class
    base          -- base class
    paramNames    -- list of strings with param names of the model
    varNameTypes  -- list of pairs of strings with varible names and types of the model
    derivedParams -- list of pairs, where the first member is string with name of
                        the derived parameter and the second MUST be an instance of the class
                        which inherits from libgenn.Snippet.DerivedParamFunc
    custom_body   -- dictionary with attributes and methods of the new class
    """

    def ctor( self ):
        base.__init__(self)

    body = {
            '__init__' : ctor,
    }
    
    if paramNames is not None:
        body['getParamNames'] = lambda self: lg.StringVector( paramNames )

    if varNameTypes is not None:
        body['getVars'] = lambda self: lg.StringPairVector( [lg.StringPair( vn[0], vn[1] ) for vn in varNameTypes] )

    if derivedParams is not None:
        
        body['getDerivedParams'] = lambda self: lg.Snippet.StringDPFPairVector(
                [lg.Snippet.StringDPFPair( dp[0], lg.Snippet.makeDPF( dp[1] ) )
                    for dp in derivedParams] )

    if custom_body is not None:
        body.update( custom_body )

    return type( className, (base,), body )

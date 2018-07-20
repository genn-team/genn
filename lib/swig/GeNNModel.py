from os import path
from subprocess import check_call
import json
import numpy as np
import libgenn as lg
import SharedLibraryModel as slm
from GeNNGroups import NeuronGroup, SynapseGroup

class GeNNModel( object ):

    """GeNNModel class
    This class helps to define, build and run a GeNN model from python
    """

    def __init__(self, precision=None, modelName=None, enableDebug=False,
                 autoInitSparseVars=True):
        """Init GeNNModel
        Keyword args:
        scalar    -- precision as string ("float" or "double" or "long double")
                     Defaults to float.
        modelName -- name of the model
        """
        self._scalar = precision
        if precision is None or precision == 'float':
            gennFloatType = 'GENN_FLOAT'
            self._scalar = 'float'
            self._slm = slm.SharedLibraryModel_f()
            self._npType = np.float32
        elif precision == 'double':
            gennFloatType = 'GENN_DOUBLE'
            self._slm = slm.SharedLibraryModel_d()
            self._npType = np.float64
        elif precision == 'long double':
            gennFloatType = 'GENN_LONG_DOUBLE'
            self._slm = slm.SharedLibraryModel_ld()
            self._npType = np.float128
        else:
            raise ValueError( 'Supported precisions are "{0}", '
                              'but {1} was given'.format( 
                                  'float, double and long double',
                                  precision ) )
        
        self._built = False
        self._localhost = lg.init_cuda_mpi()
        
        lg.setDefaultVarMode( lg.VarMode_LOC_HOST_DEVICE_INIT_DEVICE )
        lg.GeNNPreferences.debugCode = enableDebug
        lg.GeNNPreferences.autoInitSparseVars = autoInitSparseVars
        lg.initGeNN()
        self._model = lg.NNmodel()
        self._model.setPrecision( getattr(lg, gennFloatType ) )
        self._modelName = None
        if modelName is not None:
            self.modelName = modelName
        self.neuronPopulations = {}
        self.synapsePopulations = {}
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

    
    def addNeuronPopulation( self, popName, numNeurons, neuron,
                             paramSpace, varSpace ):
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
       
        nGroup = NeuronGroup( popName )
        nGroup.setSize( numNeurons )
        nGroup.setNeuron( neuron, paramSpace, varSpace )
        nGroup.addTo( self._model )

        self.neuronPopulations[popName] = nGroup

        return nGroup
        
    def addSynapsePopulation( self, popName, matrixType, delaySteps,
                              source, target,
                              wUpdateModel, wuParamSpace, wuVarSpace,
                              postsynModel, psParamSpace, psVarSpace ):
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
        if self._built:
            raise Exception("GeNN model already built")

        if popName in self.synapsePopulations:
            raise ValueError( 'synapse population "{0}" already exists'.format( popName ) )

        sGroup = SynapseGroup( popName )
        sGroup.setDelaySteps( delaySteps )
        sGroup.matrixType = matrixType
        sGroup.setConnectedPopulations(
                source, self.neuronPopulations[source].size,
                target, self.neuronPopulations[target].size )
        sGroup.setWUpdate( wUpdateModel, wuParamSpace, wuVarSpace )
        sGroup.setPostsyn( postsynModel, psParamSpace, psVarSpace )
        sGroup.addTo( self._model )

        self.synapsePopulations[popName] = sGroup

        self.neuronPopulations[source].maxDelaySteps = delaySteps

        return sGroup


    def setConnections( self, popName, conns, g ):
        self.synapsePopulations[popName].setConnections( conns, g )


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
                var = self.synapsePopulations[popName].vars[varName]
        else:
            var = self.neuronPopulations[popName].vars[varName]

        var.view[mask] = vals
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
        self.neuronPopulations[popName].spikes[mask] = targets
        self.neuronPopulations[popName].spikeCount[mask] = counts
        self._slm.pushPopulationSpikesToDevice( popName )


    def build( self, pathToModel = "./" ):

        """Finalize and build a GeNN model
        
        Keyword args:
        pathToModel -- path where to place the generated model code. Defaults to the local directory.
        """

        if self._built:
            raise Exception("GeNN model already built")
        self._pathToModel = pathToModel

        for popName, popData in self.synapsePopulations.items():
            if popData.sparse:
                popData.pop.setMaxConnections( popData.size )

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
                        'size' - number of neurons
                        'vars' - dictionary with variable names. The values of the dictionary will
                                be overwritten with actuals variable views when the model is loaded
                        'spk' - view into internal spikes variable. Can be None. This value will be
                                overwritten with actual variable view when the model is loaded.
                        'spkCnt' - view into internal spike count variable. Can be None. This value will be
                                   overwritten with actual variable view when the model is loaded.

                        Example: {'Pop1' : {'size' : 10,
                                            'vars' : { 'varName1': None },
                                            'spk' : None,
                                            'spkCnt' : None }
                                 }
                        
        synapsePop  -- dictionary with synapse populations. Each population must have: 'size', 'vars'. sa neuronPop.
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
        
        if self._scalar == 'float':
            self.timestep = self._slm.assignExternalPointerSingle_f('t')
        if self._scalar == 'double':
            self.timestep = self._slm.assignExternalPointerSingle_d('t')
        if self._scalar == 'long double':
            self.timestep = self._slm.assignExternalPointerSingle_ld('t')
        self.T = self._slm.assignExternalPointerSingle_ull('iT')

        for popName, popData in self.neuronPopulations.items():
            self._slm.initNeuronPopIO( popName, popData.size )
            self._slm.pullPopulationStateFromDevice( popName )
            popData.spikes = self.assignExternalPointerPop( popName, 'glbSpk',
                    popData.size * (popData.maxDelaySteps + 1), 'unsigned int' )
            popData.spikeCount = self.assignExternalPointerPop( popName, 'glbSpkCnt',
                    popData.size * (popData.maxDelaySteps + 1), 'unsigned int' )
            if popData.maxDelaySteps > 0:
                popData.spikeQuePtr = self._slm.assignExternalPointerSingle_ui(
                      'spkQuePtr' + popName )
            else:
                popData.spikeQuePtr = [0]

            for varName, varData in popData.vars.items():
                varData.view = self.assignExternalPointerPop(
                        popName, varName, popData.size, varData.type )
                if varData.initRequired:
                    varData.view[:] = varData.values 

            self._slm.pushPopulationStateToDevice( popName )

            for egpName, egpData in popData.extraGlobalParams.items():
                # if auto allocation is not enabled, let the user care about
                # allocation and initialization of the EGP
                if egpData.needsAllocation:
                    self._slm.allocateExtraGlobalParam( popName, egpName,
                            len( egpData.values ) )
                    egpData.view = self.assignExternalPointerPop( popName,
                        egpName, len( egpData.values ), egpData.type[:-1] )
                    if egpData.initRequired:
                        egpData.view[:] = egpData.values



        for popName, popData in self.synapsePopulations.items():

            self._slm.initSynapsePopIO( popName, popData.size )
            self._slm.pullPopulationStateFromDevice( popName )
            
            if popData.sparse:
                if popData.connectionsSet:
                    pre = self.neuronPopulations[popData.src]
                    self._slm.allocateSparsePop( popName, popData.size )
                    self._slm.initializeSparsePop( popName, popData.ind,
                                                   popData.indInG, popData.g )

                else:
                    raise Exception( 'For sparse projections, the connections must be set before loading a model' )

            for varName, varData in popData.vars.items():
                size = popData.size
                if varName in [vnt[0] for vnt in popData.postsyn.getVars()]:
                    size = self.neuronPopulations[popData.trg].size
                varData.view = self.assignExternalPointerPop(
                        popName, varName, size, varData.type )
                if varName == 'g' and popData.connectionsSet:
                    continue
                if varData.initRequired:
                    varData.view[:] = varData.values 
                    #  self.initializeVarOnDevice( popName, varName,
                    #          list( range( len( varData.values ) ) ), varData.values )

            if not popData.sparse:
                if popData.connectionsSet:
                    self.initializeVarOnDevice( popName, 'g', popData.mask, popData.g )

            self._slm.pushPopulationStateToDevice( popName )

        self._slm.initializeModel()


    def assignExternalPointerPop( self, popName, varName, varSize, varType ):
        """Assign a variable to an external numpy array
        
        Args:
        popName -- population name
        varName -- a name of the variable to assing, without population name
        varSize -- the size of the variable
        varType -- type of the variable as string. The supported types are
                   scalar, float, double, long double, int, unsigned int.

        Returns numpy array of type varType

        Raises ValueError if variable type is not supported
        """

        return self.assignExternalPointerArray( varName + popName, varSize, varType )


    def assignExternalPointerArray( self, varName, varSize, varType ):
        """Assign a variable to an external numpy array
        
        Args:
        varName -- a fully qualified name of the variable to assign
        varSize -- the size of the variable
        varType -- type of the variable as string. The supported types are
                   scalar, float, double, long double, int, unsigned int.

        Returns numpy array of type varType

        Raises ValueError if variable type is not supported
        """

        if varType == 'scalar':
            if self._scalar == 'float':
                return self._slm.assignExternalPointerArray_f( varName, varSize )
            elif self._scalar == 'double':
                return self._slm.assignExternalPointerArray_d( varName, varSize )
            elif self._scalar == 'long double':
                return self._slm.assignExternalPointerArray_ld( varName, varSize )

        elif varType == 'float':
            return self._slm.assignExternalPointerArray_f( varName, varSize )
        elif varType == 'double':
            return self._slm.assignExternalPointerArray_d( varName, varSize )
        elif varType == 'long double':
            return self._slm.assignExternalPointerArray_ld( varName, varSize )
        elif varType == 'int':
            return self._slm.assignExternalPointerArray_i( varName, varSize )
        elif varType == 'unsigned int':
            return self._slm.assignExternalPointerArray_ui( varName, varSize )
        else:
            raise ValueError( 'unsupported varType' )


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

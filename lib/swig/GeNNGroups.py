import pygenn as pg

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
        if values is not None:
            self.setValues( values )

    def setInitVar( self, initVarSnippet, paramSpace ):
        """Set variable initialization using InitVarSnippet

        Args:
        initVarSnippet -- type as string or instance of a class derived from InitVarSnippet.Custom
        paramSpace     -- dict mapping parameter names to their values for InitVarSnippet
        """
        ivsInst, ivsType = ModelPreprocessor.isModelValid( initVarSnippet, pg.InitVarSnippet )
        params = ModelPreprocessor.parameterSpaceToParamValues( ivsInst, paramSpace )
        initFct = getattr( pg, 'initVar_' + ivsType )
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
            setRandomInit( initVar, values )
        else:
            try:
                iter( values )
                self.initVal = pg.uninitialisedVar()
                self.values = list( values )
                self.initRequired = True
            except TypeError:
                self.initVal = pg.NewModels.VarInit( values )

class ModelPreprocessor(object):
    @staticmethod
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
        mInstance, mType = ModelPreprocessor.isModelValid( model, modelFamily )
        paramNames = list( mInstance.getParamNames() )
        params = ModelPreprocessor.parameterSpaceToParamValues( mInstance, paramSpace )
        var_dict = { vnt[0] : Variable( vnt[0], vnt[1], varSpace[vnt[0]] )
                  for vnt in mInstance.getVars() }
        return ( mInstance, mType, paramNames, params, var_dict )

    @staticmethod
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
    
    @staticmethod
    def parameterSpaceToParamValues( model, paramSpace ):
        """Convert a paramSpace dict to ParamValues
    
        Args:
        model     -- instance of the model
        paramSpace -- dict with parameters

        Return:
        native model's ParamValues
        """
        paramVals = [paramSpace[pn] for pn in model.getParamNames()]

        return model.makeParamValues( pg.StlContainers.DoubleVector( paramVals ) )

    @staticmethod
    def varSpaceToVarValues( model, varSpace ):
        """Convert a varSpace dict to VarValues

        Args:
        model     -- instance of the model
        varSpace  -- dict with Variables

        Return:
        native model's VarValues
        """
        varVals = [v.initVal for v in varSpace.values()]

        return model.makeVarValues( pg.NewModels.VarInitVector( varVals ) )



class Group( object ):

    """Parent class of NeuronGroup, SynapseGroup and CurrentSource"""

    def __init__( self, name ):
        """Init Group

        Args:
        name -- string name of the Group
        """
        self.name = name
        self.vars = {}
        self.extraGlobalParams = {}
        self.size = None


    def setVar( self, varName, values ):
        """Set values for a Variable

        Args:
        varName -- string with the name of the variable
        values  -- iterable or a single value
        """
        self.vars[varName].setValues( values )

    def _addExtraGlobalParam( self, paramName, paramValues, model, autoAlloc=True ):
        """Add extra global parameter

        Args:
        paramName   -- string with the name of the extra global parameter
        paramValues -- iterable or a single value
        model       -- instance of the model
        autoAlloc   -- boolean whether the extra global parameter should be allocated. Defaults to true.
        """
        pnt = list( model.getExtraGlobalParams() )
        paramType = None
        for pn, pt in pnt:
            if pn == paramName:
                paramType = pt
                break

        egp = Variable( paramName, paramType, paramValues )
        egp.needsAllocation = autoAlloc

        self.extraGlobalParams[paramName] = egp


class NeuronGroup( Group ):

    """Class representing a group of neurons"""

    def __init__( self, name ):
        """Init NeuronGroup

        Args:
        name -- string name of the group
        """
        super( NeuronGroup, self ).__init__( name )
        self.neuron = None
        self.spikes = None
        self.spikeCount = None
        self.spikeQuePtr = [0]
        self.isSpikeSourceArray = False
        self._maxDelaySteps = 0

    @property
    def currentSpikes( self ):
        """Current spikes from GeNN"""
        return self.spikes[self.spikeQuePtr[0] * self.size :
                self.spikeQuePtr[0] * self.size + self.spikeCount[self.spikeQuePtr[0]]]

    @property
    def maxDelaySteps( self ):
        """Maximum delay steps needed for this group"""
        return self._maxDelaySteps

    @maxDelaySteps.setter
    def maxDelaySteps( self, delaySteps ):
        self._maxDelaySteps = int(max( self._maxDelaySteps, delaySteps ))


    def setSize( self, size ):
        """Set number of neurons in the group"""
        self.size = size

    def setNeuron( self, model, paramSpace, varSpace ):
        """Set neuron, its parameters and initial variables

        Args:
        model      -- type as string of intance of the model
        paramSpace -- dict with model parameters
        varSpace   -- dict with model variables
        """
        ( self.neuron, self.type, self.paramNames,
          self.params, self.vars ) = ModelPreprocessor.prepareModel( model, paramSpace,
                                                        varSpace,
                                                        pg.NeuronModels )
        if self.type == 'SpikeSourceArray':
            self.isSpikeSourceArray = True

    def addTo( self, nnModel ):
        """Add this NeuronGroup to the GeNN NNmodel

        Args:
        nnModel -- GeNN NNmodel
        """
        addFct = getattr( nnModel, 'addNeuronPopulation_' + self.type )

        varIni = ModelPreprocessor.varSpaceToVarValues( self.neuron, self.vars )
        self.pop = addFct( self.name, self.size, self.neuron,
                           self.params, varIni )

    def addExtraGlobalParam( self, paramName, paramValues ):
        """Add extra global parameter

        Args:
        paramName   -- string with the name of the extra global parameter
        paramValues -- iterable or a single value
        """
        self._addExtraGlobalParam( paramName, paramValues, self.neuron )


class SynapseGroup( Group ):

    """Class representing synaptic connection between two groups of neurons"""

    def __init__( self, name ):
        """Init SynapseGroup

        Args:
        name -- string name of the group
        """
        self.sparse = False
        self.connectionsSet = False
        super( SynapseGroup, self ).__init__( name )
        self.wUpdate = None
        self.postsyn = None
        self.vars['g'] = None
        self.src = None
        self.trg = None
    
    @property
    def size( self ):
        """Number of synaptic connections"""
        if not self.sparse:
            return self.trg_size * self.src_size
        else:
            return self._size

    @size.setter
    def size( self, size ):
        if self.sparse:
            self._size = size

    def setWUpdate( self, model, paramSpace, varSpace ):
        """Set weight update model, its parameters and initial variables

        Args:
        model      -- type as string of intance of the model
        paramSpace -- dict with model parameters
        varSpace   -- dict with model variables
        """
        ( self.wUpdate, self.wuType, self.wuParamNames,
          self.wuParams, varrs ) = ModelPreprocessor.prepareModel( model, paramSpace,
                                                      varSpace,
                                                      pg.WeightUpdateModels )
        self.wuVarNames = varrs.keys()
        self.vars.update( varrs )

    def setPostsyn( self, model, paramSpace, varSpace ):
        """Set postsynaptic model, its parameters and initial variables

        Args:
        model      -- type as string of intance of the model
        paramSpace -- dict with model parameters
        varSpace   -- dict with model variables
        """
        ( self.postsyn, self.psType, self.psParamNames,
          self.psParams, varrs ) = ModelPreprocessor.prepareModel( model, paramSpace,
                                                      varSpace,
                                                      pg.PostsynapticModels )
        self.psVarNames = varrs.keys()
        self.vars.update( varrs )
    
    def setDelaySteps( self, delaySteps ):
        """Set number delay steps"""
        self.delaySteps = delaySteps
    
    @property
    def matrixType( self ):
        """Type of the projection matrix"""
        return self._matrixType

    @matrixType.setter
    def matrixType( self, matrixType ):
        self._matrixType = getattr( pg, 'SynapseMatrixType_' + matrixType )
        if matrixType.split('_')[0] == 'SPARSE':
            self.sparse = True
            self.ind = None
            self.indInG = None
        else:
            self.sparse = False

    def setConnections( self, conns, g ):
        """Set connections between two groups of neurons

        Args:
        conns -- connections as tuples (pre, post)
        g     -- strength of the connection
        """
        if self.sparse:
            conns.sort()
            self.size = len( conns )
            self.ind = [ post for (_, post) in conns ]
            self.indInG = []
            self.indInG.append( 0 )
            curPre = 0
            for i, (pre, _) in enumerate( conns ):
                if pre > curPre:
                    self.indInG.append( i )
                    while pre != curPre:
                        curPre += 1
            while len(self.indInG) < self.src_size + 1:
                self.indInG.append( len(conns) )
            self.g = g
        else:
            self.mask = [ pre * self.trg_size + post for (pre, post) in conns ]
            self.g = g
            self.size = self.trg_size * self.src_size

        self.connectionsSet = True
        

    def setConnectedPopulations( self, source, src_size, target, trg_size ):
        """Set two groups of neurons connected by this SynapseGroup

        Args:
        source   -- string name of the presynaptic neuron group
        src_size -- number of neurons in the presynaptic group
        target   -- string name of the postsynaptic neuron group
        trg_size -- number of neurons in the presynaptic group
        """
        self.src = source
        self.trg = target
        self.src_size = src_size
        self.trg_size = trg_size

    def addTo( self, nnModel ):
        """Add this SynapseGroup to the GeNN NNmodel

        Args:
        nnModel -- GeNN NNmodel
        """
        addFct = getattr( nnModel,
                          ( 'addSynapsePopulation_' + 
                            self.wuType + '_' + self.psType ) )

        wuVarIni = ModelPreprocessor.varSpaceToVarValues( self.wUpdate,
                { vn : self.vars[vn] for vn in self.wuVarNames } )
        psVarIni = ModelPreprocessor.varSpaceToVarValues( self.postsyn,
                { vn : self.vars[vn] for vn in self.psVarNames } )

        self.pop = addFct( self.name, self.matrixType, self.delaySteps,
                           self.src, self.trg,
                           self.wUpdate, self.wuParams, wuVarIni,
                           self.postsyn, self.psParams, psVarIni )
    
    def addExtraGlobalParam( self, paramName, paramValues ):
        """Add extra global parameter

        Args:
        paramName   -- string with the name of the extra global parameter
        paramValues -- iterable or a single value
        """
        self._addExtraGlobalParam( paramName, paramValues, self.wUpdate )


class CurrentSource( Group ):

    """Class representing a current injection into a group of neurons"""

    def __init__( self, name ):
        """Init CurrentSource

        Args:
        name -- string name of the current source
        """
        super( CurrentSource, self ).__init__( name )
        self.currentSourceModel = None
        self.targetPop = None

    @property
    def size( self ):
        """Number of neuron in the injected population"""
        return self.targetPop.size

    @size.setter
    def size( self, _ ):
        pass

    def setCurrentSourceModel( self, model, paramSpace, varSpace ):
        """Set curront source model, its parameters and initial variables

        Args:
        model      -- type as string of intance of the model
        paramSpace -- dict with model parameters
        varSpace   -- dict with model variables
        """
        ( self.currentSourceModel, self.type, self.paramNames,
          self.params, self.vars ) = ModelPreprocessor.prepareModel( model, paramSpace,
                                                        varSpace,
                                                        pg.CurrentSourceModels )

    def addTo( self, nnModel, pop ):
        """Inject this CurrentSource into population and add add it to the GeNN NNmodel

        Args:
        pop     -- instance of NeuronGroup into which this CurrentSource should be injected
        nnModel -- GeNN NNmodel
        """
        addFct = getattr( nnModel, 'addCurrentSource_' + self.type )
        self.targetPop = pop

        varIni = ModelPreprocessor.varSpaceToVarValues( self.currentSourceModel, self.vars )
        self.pop = addFct( self.name, self.currentSourceModel, pop.name,
                           self.params, varIni )

    def addExtraGlobalParam( self, paramName, paramValues ):
        """Add extra global parameter

        Args:
        paramName   -- string with the name of the extra global parameter
        paramValues -- iterable or a single value
        """
        self._addExtraGlobalParam( paramName, paramValues, self.currentSourceModel )

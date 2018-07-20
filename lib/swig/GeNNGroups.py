
import libgenn as lg

class Variable(object):

    def __init__(self, variableName, variableType, values=None):
        self.name = variableName
        self.type = variableType
        self.view = None
        self.needsAllocation = False
        if values is not None:
            self.setValues( values )
        
    def setValues( self, values ):
        self.initRequired = False
        try:
            iter( values )
            self.initVal = values[0]
            self.values = list( values )
            self.initRequired = True
        except TypeError:
            self.initVal = values


class Group( object ):
    
    def __init__( self, name ):
        self.name = name
        self.vars = {}
        self.extraGlobalParams = {}
        self.size = None

    def prepareModel( self, model, paramSpace, varSpace, modelFamily ):

        mInstance, mType = self.isModelValid( model, modelFamily )
        paramNames = list( mInstance.getParamNames() )
        params = self.parameterSpaceToParamValues( mInstance, paramSpace )
        varrs = { vnt[0] : Variable( vnt[0], vnt[1], varSpace[vnt[0]] )
                  for vnt in mInstance.getVars() }
        return ( mInstance, mType, paramNames, params, varrs )

    def isModelValid( self, model, modelFamily ):
        """check whether the model is valid
        Args:
        model -- instance or type as string
        modelFamily -- model family (NeuronModels, WeightUpdateModels or PostsynapticModels) to which model should belong to

        Return:
        isinstance of the model and its type as string

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
                model = getattr( modelFamily, modelType )()
        return model, modelType
    
    def parameterSpaceToParamValues( self, model, paramSpace ):

        """Convert a paramSpace dict to ParamValues
    
        Args:
        paramSpace -- dict with parameters
        """
        paramVals = [paramSpace[pn] for pn in model.getParamNames()]

        return model.make_ParamValues( lg.DoubleVector( paramVals ) )

    def varSpaceToVarValues( self, model, varSpace ):
        """Convert a iniSpace dict to VarValues
    
        Args:
        varSpace  -- dict with initial values
        """
        varVals = [v.initVal for v in varSpace.values()]

        return model.make_VarValues( lg.DoubleVector( varVals ) )

    def setVar( self, varName, values ):
        self.vars[varName].setValues( values )

    def _addExtraGlobalParam( self, paramName, paramValues, model, autoAlloc=True ):

        pnt = list( model.getExtraGlobalParams() )
        paramType = None
        for pn, pt in pnt:
            if pn == paramName:
                paramType = pt

        egp = Variable( paramName, paramType, paramValues )
        egp.needsAllocation = autoAlloc

        self.extraGlobalParams[paramName] = egp


class NeuronGroup( Group ):

    def __init__( self, name ):
        super( NeuronGroup, self ).__init__( name )
        self.neuron = None
        self.spikes = None
        self.spikeCount = None
        self.isSpikeSourceArray = False
        self._maxDelaySteps = 0

    @property
    def maxDelaySteps( self ):
        return self._maxDelaySteps

    @maxDelaySteps.setter
    def maxDelaySteps( self, delaySteps ):
        self._maxDelaySteps = max( self._maxDelaySteps, delaySteps )


    def setSize( self, size ):
        self.size = size

    def setNeuron( self, model, paramSpace, varSpace ):
        ( self.neuron, self.type, self.paramNames,
          self.params, self.vars ) = self.prepareModel( model, paramSpace,
                                                        varSpace,
                                                        lg.NeuronModels )
        if self.type == 'SpikeSourceArray':
            self.isSpikeSourceArray = True

    def addTo( self, nnModel ):
        addFct = getattr( nnModel, 'addNeuronPopulation_' + self.type )

        varIni = self.varSpaceToVarValues( self.neuron, self.vars )
        self.pop = addFct( self.name, self.size, self.neuron,
                           self.params, varIni )

    def addExtraGlobalParam( self, paramName, paramValues ):
        self._addExtraGlobalParam( paramName, paramValues, self.neuron )


class SynapseGroup( Group ):

    def __init__( self, name ):
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
        if not self.sparse:
            return self.trg_size * self.src_size
        else:
            return self._size

    @size.setter
    def size( self, size ):
        if self.sparse:
            self._size = size

    def setWUpdate( self, model, paramSpace, varSpace ):
        ( self.wUpdate, self.wuType, self.wuParamNames,
          self.wuParams, varrs ) = self.prepareModel( model, paramSpace,
                                                      varSpace,
                                                      lg.WeightUpdateModels )
        self.wuVarNames = varrs.keys()
        self.vars.update( varrs )

    def setPostsyn( self, model, paramSpace, varSpace ):
        ( self.postsyn, self.psType, self.psParamNames,
          self.psParams, varrs ) = self.prepareModel( model, paramSpace,
                                                      varSpace,
                                                      lg.PostsynapticModels )
        self.psVarNames = varrs.keys()
        self.vars.update( varrs )
    
    def setDelaySteps( self, delaySteps ):
        self.delaySteps = delaySteps
    
    @property
    def matrixType( self ):
        return self._matrixType

    @matrixType.setter
    def matrixType( self, matrixType ):
        self._matrixType = getattr( lg, 'SynapseMatrixType_' + matrixType )
        if matrixType.split('_')[0] == 'SPARSE':
            self.sparse = True
            self.ind = None
            self.indInG = None
        else:
            self.sparse = False

    def setConnections( self, conns, g ):

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
        self.src = source
        self.trg = target
        self.src_size = src_size
        self.trg_size = trg_size

    def addTo( self, nnModel ):
        addFct = getattr( nnModel,
                          ( 'addSynapsePopulation_' + 
                            self.wuType + '_' + self.psType ) )

        wuVarIni = self.varSpaceToVarValues( self.wUpdate,
                { vn : self.vars[vn] for vn in self.wuVarNames } )
        psVarIni = self.varSpaceToVarValues( self.postsyn,
                { vn : self.vars[vn] for vn in self.psVarNames } )

        self.pop = addFct( self.name, self.matrixType, self.delaySteps,
                           self.src, self.trg,
                           self.wUpdate, self.wuParams, wuVarIni,
                           self.postsyn, self.psParams, psVarIni )
    
    def addExtraGlobalParam( self, paramName, paramValues ):
        self._addExtraGlobalParam( paramName, paramValues, self.wUpdate )

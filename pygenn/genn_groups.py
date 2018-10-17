"""GeNNGroups
This module provides classes which automatize model checks and parameter
convesions for GeNN Groups
"""

from six import iteritems
import genn_wrapper
import model_preprocessor
from model_preprocessor import Variable


class Group(object):

    """Parent class of NeuronGroup, SynapseGroup and CurrentSource"""

    def __init__(self, name):
        """Init Group

        Args:
        name -- string name of the Group
        """
        self.name = name
        self.vars = {}
        self.extraGlobalParams = {}

    def setVar(self, varName, values):
        """Set values for a Variable

        Args:
        varName -- string with the name of the variable
        values  -- iterable or a single value
        """
        self.vars[varName].setValues(values)

    def _addExtraGlobalParam(self, paramName, paramValues, model, autoAlloc=True):
        """Add extra global parameter

        Args:
        paramName   -- string with the name of the extra global parameter
        paramValues -- iterable or a single value
        model       -- instance of the model
        autoAlloc   -- boolean whether the extra global parameter should be allocated. Defaults to true.
        """
        pnt = list(model.getExtraGlobalParams())
        paramType = None
        for pn, pt in pnt:
            if pn == paramName:
                paramType = pt
                break

        egp = Variable(paramName, paramType, paramValues)
        egp.needsAllocation = autoAlloc

        self.extraGlobalParams[paramName] = egp


class NeuronGroup(Group):

    """Class representing a group of neurons"""

    def __init__(self, name):
        """Init NeuronGroup

        Args:
        name -- string name of the group
        """
        super(NeuronGroup, self).__init__(name)
        self.neuron = None
        self.spikes = None
        self.spikeCount = None
        self.spikeQuePtr = [0]
        self.isSpikeSourceArray = False
        self._maxDelaySteps = 0

    @property
    def currentSpikes(self):
        """Current spikes from GeNN"""
        return self.spikes[self.spikeQuePtr[0] * self.size :
                self.spikeQuePtr[0] * self.size + self.spikeCount[self.spikeQuePtr[0]]]

    @property
    def delaySlots(self):
        """Maximum delay steps needed for this group"""
        return self.pop.getNumDelaySlots()

    @property
    def size(self):
        return self.pop.getNumNeurons()

    def setNeuron(self, model, paramSpace, varSpace):
        """Set neuron, its parameters and initial variables

        Args:
        model      -- type as string of intance of the model
        paramSpace -- dict with model parameters
        varSpace   -- dict with model variables
        """
        ( self.neuron, self.type, self.paramNames, self.params, self.varNames,
            self.vars ) = model_preprocessor.prepareModel( model, paramSpace,
                                                           varSpace,
                                                           modelFamily=genn_wrapper.NeuronModels )
        if self.type == "SpikeSourceArray":
            self.isSpikeSourceArray = True

    def addTo(self, nnModel, numNeurons):
        """Add this NeuronGroup to the GeNN NNmodel

        Args:
        nnModel    -- GeNN NNmodel
        numNeurons -- int number of neurons
        """
        addFct = getattr(nnModel, "addNeuronPopulation_" + self.type)

        varIni = model_preprocessor.varSpaceToVarValues(self.neuron, self.vars)
        self.pop = addFct( self.name, numNeurons, self.neuron,
                           self.params, varIni )

        for varName, var in iteritems(self.vars):
            if var.initRequired:
                self.pop.setVarMode(varName, genn_wrapper.VarMode_LOC_HOST_DEVICE_INIT_HOST)

    def addExtraGlobalParam(self, paramName, paramValues):
        """Add extra global parameter

        Args:
        paramName   -- string with the name of the extra global parameter
        paramValues -- iterable or a single value
        """
        self._addExtraGlobalParam(paramName, paramValues, self.neuron)


class SynapseGroup(Group):

    """Class representing synaptic connection between two groups of neurons"""

    def __init__(self, name):
        """Init SynapseGroup

        Args:
        name -- string name of the group
        """
        self.sparse = False
        self.connectionsSet = False
        super(SynapseGroup, self).__init__(name)
        self.wUpdate = None
        self.postsyn = None
        self.src = None
        self.trg = None
        self.preVars = {}
        self.postVars = {}
    
    @property
    def size(self):
        """Size of connection matrix"""
        if not self.sparse:
            return self.trg_size * self.src_size
        else:
            return self._size

    @size.setter
    def size(self, size):
        if self.sparse:
            self._size = size

    def setPreVar(self, varName, values):
        """Set values for a presynaptic variable

        Args:
        varName -- string with the name of the presynaptic variable
        values  -- iterable or a single value
        """
        self.preVars[varName].setValues(values)

    def setPostVar(self, varName, values):
        """Set values for a postsynaptic variable

        Args:
        varName -- string with the name of the presynaptic variable
        values  -- iterable or a single value
        """
        self.postVars[varName].setValues(values)
        
    def setWUpdate(self, model, paramSpace, varSpace, preVarSpace, postVarSpace):
        """Set weight update model, its parameters and initial variables

        Args:
        model           -- type as string of intance of the model
        paramSpace      -- dict with model parameters
        varSpace        -- dict with model variables
        preVarSpace     -- dict with model presynaptic variables
        postVarSpace    -- dict with model postsynaptic variables
        """
        ( self.wUpdate, self.wuType, self.wuParamNames, self.wuParams, 
         self.wuVarNames, varDict, self.wuPreVarNames, preVarDict,
         self.wuPostVarNames, postVarDict) =\
             model_preprocessor.prepareModel( model, paramSpace,
                                             varSpace, preVarSpace, postVarSpace,
                                             modelFamily=genn_wrapper.WeightUpdateModels )
        self.vars.update(varDict)
        self.preVars.update(preVarDict)
        self.postVars.update(postVarDict)

    def setPostsyn(self, model, paramSpace, varSpace):
        """Set postsynaptic model, its parameters and initial variables

        Args:
        model      -- type as string of intance of the model
        paramSpace -- dict with model parameters
        varSpace   -- dict with model variables
        """
        ( self.postsyn, self.psType, self.psParamNames, self.psParams, self.psVarNames,
            varDict ) = model_preprocessor.prepareModel( model, paramSpace,
                                                         varSpace,
                                                         modelFamily=genn_wrapper.PostsynapticModels )
        self.vars.update(varDict)

    @property
    def matrixType(self):
        """Type of the projection matrix"""
        return self._matrixType

    @matrixType.setter
    def matrixType(self, matrixType):
        self._matrixType = getattr(genn_wrapper, "SynapseMatrixType_" + matrixType)
        if matrixType.startswith("SPARSE"):
            self.sparse = True
            self.ind = None
            self.indInG = None
        else:
            self.sparse = False

        if matrixType.endswith("GLOBALG"):
            self.globalG = True
        else:
            self.globalG = False

    def setConnections(self, conns, g):
        """Set connections between two groups of neurons

        Args:
        conns -- connections as tuples (pre, post)
        g     -- strength of the connection
        """
        if self.sparse:
            conns.sort()
            self.size = len(conns)
            self.ind = [ post for (_, post) in conns ]
            self.indInG = []
            self.indInG.append(0)
            curPre = 0
            self.maxConn = 0
            # convert connection tuples to indInG
            for i, (pre, _) in enumerate(conns):
                while pre != curPre:
                    self.indInG.append(i)
                    curPre += 1
            # if there are any "hanging" presynaptic neurons without connections,
            # they should all point to the end of indInG
            while len(self.indInG) < self.src_size + 1:
                self.indInG.append(len(conns))
            # compute max number of connections from taget neuron to source
            self.maxConn = int(max([ self.indInG[i] - self.indInG[i-1] for i in range(len(self.indInG)) if i != 0 ]))
        else:
            self.gMask = [ pre * self.trg_size + post for (pre, post) in conns ]
            self.size = self.trg_size * self.src_size

        if not self.globalG:
            self.vars["g"].setValues(g)
            self.pop.setWUVarMode("g", genn_wrapper.VarMode_LOC_HOST_DEVICE_INIT_HOST)

        self.connectionsSet = True
        

    def setConnectedPopulations(self, source, src_size, target, trg_size):
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

    def addTo(self, nnModel, delaySteps, connectivityInitialiser):
        """Add this SynapseGroup to the GeNN NNmodel

        Args:
        nnModel -- GeNN NNmodel
        """
        addFct = getattr( nnModel,
                          ( "addSynapsePopulation_" +
                            self.wuType + "_" + self.psType ) )

        wuVarIni = model_preprocessor.varSpaceToVarValues( self.wUpdate,
                { vn : self.vars[vn] for vn in self.wuVarNames } )
        wuPreVarIni = model_preprocessor.preVarSpaceToVarValues( self.wUpdate,
                { vn : self.preVars[vn] for vn in self.wuPreVarNames } )
        wuPostVarIni = model_preprocessor.postVarSpaceToVarValues( self.wUpdate,
                { vn : self.postVars[vn] for vn in self.wuPostVarNames } )
        psVarIni = model_preprocessor.varSpaceToVarValues( self.postsyn,
                { vn : self.vars[vn] for vn in self.psVarNames } )

        self.pop = addFct( self.name, self.matrixType, delaySteps,
                           self.src, self.trg,
                           self.wUpdate, self.wuParams, wuVarIni, wuPreVarIni, wuPostVarIni,
                           self.postsyn, self.psParams, psVarIni, connectivityInitialiser )

        for varName, var in iteritems(self.vars):
            if var.initRequired:
                if varName in self.wuVarNames:
                    self.pop.setWUVarMode(varName, genn_wrapper.VarMode_LOC_HOST_DEVICE_INIT_HOST)
                if varName in self.wuPreVarNames:
                    self.pop.setWUPreVarMode(varName, genn_wrapper.VarMode_LOC_HOST_DEVICE_INIT_HOST)
                if varName in self.wuPostVarNames:
                    self.pop.setWUPostVarMode(varName, genn_wrapper.VarMode_LOC_HOST_DEVICE_INIT_HOST)
                if varName in self.psVarNames:
                    self.pop.setPSVarMode(varName, genn_wrapper.VarMode_LOC_HOST_DEVICE_INIT_HOST)
    
    def addExtraGlobalParam(self, paramName, paramValues):
        """Add extra global parameter

        Args:
        paramName   -- string with the name of the extra global parameter
        paramValues -- iterable or a single value
        """
        self._addExtraGlobalParam(paramName, paramValues, self.wUpdate)


class CurrentSource(Group):

    """Class representing a current injection into a group of neurons"""

    def __init__(self, name):
        """Init CurrentSource

        Args:
        name -- string name of the current source
        """
        super(CurrentSource, self).__init__(name)
        self.currentSourceModel = None
        self.targetPop = None

    @property
    def size(self):
        """Number of neuron in the injected population"""
        return self.targetPop.size

    @size.setter
    def size(self, _):
        pass

    def setCurrentSourceModel(self, model, paramSpace, varSpace):
        """Set curront source model, its parameters and initial variables

        Args:
        model      -- type as string of intance of the model
        paramSpace -- dict with model parameters
        varSpace   -- dict with model variables
        """
        ( self.currentSourceModel, self.type, self.paramNames, self.params, self.varNames,
            self.vars ) = model_preprocessor.prepareModel( model, paramSpace,
                                                           varSpace,
                                                           modelFamily=genn_wrapper.CurrentSourceModels )

    def addTo(self, nnModel, pop):
        """Inject this CurrentSource into population and add add it to the GeNN NNmodel

        Args:
        pop     -- instance of NeuronGroup into which this CurrentSource should be injected
        nnModel -- GeNN NNmodel
        """
        addFct = getattr(nnModel, "addCurrentSource_" + self.type)
        self.targetPop = pop

        varIni = model_preprocessor.varSpaceToVarValues(self.currentSourceModel, self.vars)
        self.pop = addFct( self.name, self.currentSourceModel, pop.name,
                           self.params, varIni )

        for varName, var in iteritems(self.vars):
            if var.initRequired:
                self.pop.setVarMode(varName, genn_wrapper.VarMode_LOC_HOST_DEVICE_INIT_HOST)

    def addExtraGlobalParam(self, paramName, paramValues):
        """Add extra global parameter

        Args:
        paramName   -- string with the name of the extra global parameter
        paramValues -- iterable or a single value
        """
        self._addExtraGlobalParam(paramName, paramValues, self.currentSourceModel)

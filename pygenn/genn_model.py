"""GeNNModel

This module provides GeNNModel class to simplify working with pygenn module and
helper functions to derive custom model classes.

``GeNNModel`` can be (and should be) used to configure a model, build, load and
finally run it. Recording is done manually by pulling from the population of
interest and then copying the values from ``Variable.view`` attribute. Each
simulation step must be triggered manually by calling ``stepTime`` function.

Example:
    The following example shows in a (very) simplified manner how to build and
    run a simulation using GeNNModel::

        import GeNNModel
        gm = GeNNModel.GeNNModel()

        # add populations
        neuronPop = gm.addNeuronPopulation(_parameters_truncated_)
        synPop = gm.addSynapsePopulation(_parameters_truncated_)

        # build and load model
        gm.build(path_to_model)
        gm.load()

        Vs = numpy.empty((simulation_length, population_size))
        # Variable.view provides a view into a raw C array
        # here a Variable call V (voltage) will be recorded
        vView = neuronPop.vars['V'].view

        # run a simulation for 1000 steps
        for i in range 1000:
            # manually trigger one simulation step
            gm.stepTime()
            # when you pull state from device, views of all variables are updated
            # and show current simulated values
            gm.pullStateFromDevice(neuronPopName)
            # finally, record voltage by copying form view into array.
            Vs[i,:] = vView
"""
# python imports
from os import path
from subprocess import check_call # to call make
# 3rd party imports
import numpy as np
from six import iteritems
# pygenn imports
import genn_wrapper
import genn_wrapper.SharedLibraryModel as slm
from genn_wrapper.NewModels import VarInit
from genn_wrapper.InitSparseConnectivitySnippet import Init
from genn_groups import NeuronGroup, SynapseGroup, CurrentSource
from model_preprocessor import prepareSnippet

class GeNNModel(object):

    """GeNNModel class
    This class helps to define, build and run a GeNN model from python
    """

    def __init__(self, precision=None, modelName='GeNNModel', enableDebug=False,
                 cpuOnly=False):
        """Init GeNNModel
        Keyword args:
        precision    -- string precision as string ("float" or "double" or "long double")
                     Defaults to float.
        modelName    -- string name of the model. Defaults to "GeNNModel".
        enableDebug  -- boolean enable debug mode. Disabled by default.
        cpuOnly      -- boolean whether GeNN should run only on CPU. Disabled by default.
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
        self._cpuOnly = cpuOnly
        self._localhost = genn_wrapper.initMPI_pygenn()
        
        genn_wrapper.setDefaultVarMode(genn_wrapper.VarMode_LOC_HOST_DEVICE_INIT_DEVICE)
        genn_wrapper.GeNNPreferences.cvar.debugCode = enableDebug
        self._model = genn_wrapper.NNmodel()
        self._model.setPrecision(getattr(genn_wrapper, gennFloatType))
        self.modelName = modelName
        self.neuronPopulations = {}
        self.synapsePopulations = {}
        self.currentSources = {}
        self._dT = 0.1

    @property
    def modelName(self):
        """Name of the model"""
        return self._modelName

    @modelName.setter
    def modelName(self, modelName):
        if self._built:
            raise Exception("GeNN model already built")
        self._modelName = modelName
        self._model.setName(modelName)

    @property
    def t(self):
        """Simulation time in ms"""
        return self._T[0]

    def timestep(self):
        """Simulation time step"""
        return self._TS[0]

    @property
    def dT(self):
        """Step size"""
        return self._dT

    @dT.setter 
    def dT(self, dt):
        if self._built:
            raise Exception("GeNN model already built")
        self._dT = dt
        self._model.setDT(dt)
    
    def addNeuronPopulation( self, popName, numNeurons, neuron,
                             paramSpace, varSpace ):
        """Add a neuron population to the GeNN model

        Args:
        popName     -- name of the new population
        numNeurons  -- number of neurons in the new population
        neuron      -- type of the NeuronModels class as string or instance of neuron class
                        derived from NeuronModels::Custom class.
                        sa createCustomNeuronClass
        paramSpace  -- dict with param values for the NeuronModels class
        varSpace    -- dict with initial variable values for the NeuronModels class
        """
        if self._built:
            raise Exception("GeNN model already built")
        if popName in self.neuronPopulations:
            raise ValueError('neuron population "{0}" already exists'.format(popName))
       
        nGroup = NeuronGroup(popName)
        nGroup.setNeuron(neuron, paramSpace, varSpace)
        nGroup.addTo(self._model, int(numNeurons))

        self.neuronPopulations[popName] = nGroup

        return nGroup
        
    def addSynapsePopulation( self, popName, matrixType, delaySteps,
                              source, target,
                              wUpdateModel, wuParamSpace, wuVarSpace, wuPreVarSpace, wuPostVarSpace,
                              postsynModel, psParamSpace, psVarSpace,
                              connectivityInitialiser=genn_wrapper.uninitialisedConnectivity()):
        """Add a synapse population to the GeNN model

        Args:
        popName      -- name of the new population
        matrixType   -- type of the matrix as string
        delaySteps   -- delay in number of steps
        source       -- name of the source population
        target       -- name of the target population
        wUptateModel -- type of the WeightUpdateModels class as string or instance of weight update model class
                            derived from WeightUpdateModels::Custom class.
                            sa createCustomWeightUpdateClass
        wuParamValues   -- dict with param values for the WeightUpdateModels class
        wuInitVarValues -- dict with initial variable values for the WeightUpdateModels class
        postsynModel    -- type of the PostsynapticModels class as string or instance of postsynaptic model class
                            derived from PostsynapticModels::Custom class.
                            sa createCustomPostsynapticClass
        postsynParamValues   -- dict with param values for the PostsynapticModels class
        postsynInitVarValues -- dict with initial variable values for the PostsynapticModels class
        connectivityInitialiser -- InitSparseConnectivitySnippet::Init for connectivity
        """
        if self._built:
            raise Exception("GeNN model already built")

        if popName in self.synapsePopulations:
            raise ValueError('synapse population "{0}" already exists'.format(popName))

        sGroup = SynapseGroup(popName)
        sGroup.matrixType = matrixType
        sGroup.setConnectedPopulations(
                source, self.neuronPopulations[source].size,
                target, self.neuronPopulations[target].size )
        sGroup.setWUpdate(wUpdateModel, wuParamSpace, wuVarSpace, wuPreVarSpace, wuPostVarSpace)
        sGroup.setPostsyn(postsynModel, psParamSpace, psVarSpace)
        sGroup.addTo(self._model, delaySteps, connectivityInitialiser)

        self.synapsePopulations[popName] = sGroup

        return sGroup

    def addCurrentSource( self, csName, currentSourceModel, popName,
                             paramSpace, varSpace ):
        """Add a current source to the GeNN model

        Args:
        csName      -- name of the new current source
        currentSourceModel -- type of the CurrentSourceModels class as string or
                              instance of CurrentSourceModels class derived from
                              CurrentSourceModels::Custom class
                              sa createCustomCurrentSourceClass
        popName     -- name of the population into which the current source should be injected
        paramSpace  -- dict with param values for the CurrentSourceModels class
        varSpace    -- dict with initial variable values for the CurrentSourceModels class
        """
        if self._built:
            raise Exception("GeNN model already built")
        if popName not in self.neuronPopulations:
            raise ValueError('neuron population "{0}" does not exist'.format(popName))
        if csName in self.currentSources:
            raise ValueError('current source "{0}" already exists'.format(csName))

        cSource = CurrentSource(csName)
        cSource.setCurrentSourceModel(currentSourceModel, paramSpace, varSpace)
        cSource.addTo(self._model, self.neuronPopulations[popName])

        self.currentSources[csName] = cSource

        return cSource


    def setConnections(self, popName, conns, g):
        self.synapsePopulations[popName].setConnections(conns, g)


    def initializeVarOnDevice(self, popName, varName, mask, vals):
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
                        population "{1}" does not exist'.format(varName, popName))
            else:
                var = self.synapsePopulations[popName].vars[varName]
        else:
            var = self.neuronPopulations[popName].vars[varName]

        var.view[mask] = vals
        self.pushStateToDevice(popName)


    def initializeSpikesOnDevice(self, popName, mask, targets, counts):
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
                    population "{1}" does not exist'.format(varName, popName))
        self.neuronPopulations[popName].spikes[mask] = targets
        self.neuronPopulations[popName].spikeCount[mask] = counts
        self.pushSpikesToDevice(popName)


    def build(self, pathToModel = "./"):

        """Finalize and build a GeNN model
        
        Keyword args:
        pathToModel -- path where to place the generated model code. Defaults to the local directory.
        """

        if self._built:
            raise Exception("GeNN model already built")
        self._pathToModel = pathToModel

        for popName, popData in iteritems(self.synapsePopulations):
            if popData.sparse:
                popData.pop.setMaxConnections(popData.maxConn)

        self._model.finalize()
        genn_wrapper.generate_model_runner_pygenn(self._model, self._pathToModel, self._localhost)

        check_call(['make', '-C', path.join(pathToModel, self.modelName + '_CODE') ])
        
        self._built = True

    def load(self):
        """import the model as shared library and initialize it"""
        if not self._built:
            raise Exception("GeNN model has to be built before running")
        
        self._slm.open(self._pathToModel, self.modelName)


        self._slm.allocateMem()
        self._slm.initialize()
        
        if self._scalar == 'float':
            self._T = self._slm.assignExternalPointerSingle_f('t')
        if self._scalar == 'double':
            self.T = self._slm.assignExternalPointerSingle_d('t')
        if self._scalar == 'long double':
            self._T = self._slm.assignExternalPointerSingle_ld('t')
        self._TS = self._slm.assignExternalPointerSingle_ull('iT')

        for popName, popData in iteritems(self.neuronPopulations):
            self._slm.initNeuronPopIO(popName)
            popData.spikes = self.assignExternalPointerPop( popName, 'glbSpk',
                    popData.size * popData.delaySlots, 'unsigned int' )
            popData.spikeCount = self.assignExternalPointerPop( popName, 'glbSpkCnt',
                    popData.delaySlots, 'unsigned int' )
            if popData.delaySlots > 1:
                popData.spikeQuePtr = self._slm.assignExternalPointerSingle_ui(
                      'spkQuePtr' + popName )

            for varName, varData in iteritems(popData.vars):
                varData.view = self.assignExternalPointerPop(
                        popName, varName, popData.size, varData.type )
                if varData.initRequired:
                    varData.view[:] = varData.values 


            for egpName, egpData in iteritems(popData.extraGlobalParams):
                # if auto allocation is not enabled, let the user care about
                # allocation and initialization of the EGP
                if egpData.needsAllocation:
                    self._slm.allocateExtraGlobalParam( popName, egpName,
                            len(egpData.values))
                    egpData.view = self.assignExternalPointerPop( popName,
                        egpName, len(egpData.values), egpData.type[:-1])
                    if egpData.initRequired:
                        egpData.view[:] = egpData.values

        for popName, popData in iteritems(self.synapsePopulations):

            self._slm.initSynapsePopIO(popName)
            
            if popData.sparse:
                if popData.connectionsSet:
                    self._slm.allocateSparseProj(popName, len(popData.ind))
                    self._slm.initializeSparseProj( popName, popData.ind,
                                                   popData.indInG )

                else:
                    raise Exception('For sparse projections, the connections must be set before loading a model')

            for varName, varData in iteritems(popData.vars):
                size = popData.size
                if varName in [vnt[0] for vnt in popData.postsyn.getVars()]:
                    size = self.neuronPopulations[popData.trg].size
                if varName == 'g' and popData.globalG:
                    continue
                varData.view = self.assignExternalPointerPop(
                        popName, varName, size, varData.type )
                if varData.initRequired:
                    if varName == 'g' and popData.connectionsSet and not popData.sparse:
                        varData.view[:] = np.zeros((size,))
                        varData.view[popData.gMask] = varData.values
                    else:
                        varData.view[:] = varData.values

        for srcName, srcData in iteritems(self.currentSources):
            self._slm.initCurrentSourceIO(srcName)

            for varName, varData in iteritems(srcData.vars):
                varData.view = self.assignExternalPointerPop(
                        srcName, varName, srcData.size, varData.type )
                if varData.initRequired:
                    varData.view[:] = varData.values

            for egpName, egpData in iteritems(srcData.extraGlobalParams):
                # if auto allocation is not enabled, let the user care about
                # allocation and initialization of the EGP
                if egpData.needsAllocation:
                    self._slm.allocateExtraGlobalParam( srcName, egpName,
                            len(egpData.values))
                    egpData.view = self.assignExternalPointerPop( srcName,
                        egpName, len(egpData.values), egpData.type[:-1])
                    if egpData.initRequired:
                        egpData.view[:] = egpData.values

        self._slm.initializeModel()

        if self._cpuOnly:
            self.stepTime = self._slm.stepTimeCPU
        else:
            self.stepTime = self._slm.stepTimeGPU


    def assignExternalPointerPop(self, popName, varName, varSize, varType):
        """Assign a population variable to an external numpy array
        
        Args:
        popName -- string population name
        varName -- string a name of the variable to assing, without population name
        varSize -- int the size of the variable
        varType -- string type of the variable. The supported types are
                   char, unsigned char, short, unsigned short, int, unsigned int,
                   long, unsigned long, long long, unsigned long long,
                   float, double, long double and scalar.

        Returns numpy array of type varType

        Raises ValueError if variable type is not supported
        """

        return self.assignExternalPointerArray(varName + popName, varSize, varType)


    def assignExternalPointerArray(self, varName, varSize, varType):
        """Assign a variable to an external numpy array
        
        Args:
        varName -- string a fully qualified name of the variable to assign
        varSize -- int the size of the variable
        varType -- string type of the variable. The supported types are
                   char, unsigned char, short, unsigned short, int, unsigned int,
                   long, unsigned long, long long, unsigned long long,
                   float, double, long double and scalar.

        Returns numpy array of type varType

        Raises ValueError if variable type is not supported
        """

        if varType == 'scalar':
            if self._scalar == 'float':
                return self._slm.assignExternalPointerArray_f(varName, varSize)
            elif self._scalar == 'double':
                return self._slm.assignExternalPointerArray_d(varName, varSize)
            elif self._scalar == 'long double':
                return self._slm.assignExternalPointerArray_ld(varName, varSize)

        elif varType == 'char':
            return self._slm.assignExternalPointerArray_c(varName, varSize)
        elif varType == 'unsigned char':
            return self._slm.assignExternalPointerArray_uc(varName, varSize)
        elif varType == 'short':
            return self._slm.assignExternalPointerArray_s(varName, varSize)
        elif varType == 'unsigned short':
            return self._slm.assignExternalPointerArray_us(varName, varSize)
        elif varType == 'int':
            return self._slm.assignExternalPointerArray_i(varName, varSize)
        elif varType == 'unsigned int':
            return self._slm.assignExternalPointerArray_ui(varName, varSize)
        elif varType == 'long':
            return self._slm.assignExternalPointerArray_l(varName, varSize)
        elif varType == 'unsigned long':
            return self._slm.assignExternalPointerArray_ul(varName, varSize)
        elif varType == 'long long':
            return self._slm.assignExternalPointerArray_ll(varName, varSize)
        elif varType == 'unsigned long long':
            return self._slm.assignExternalPointerArray_ull(varName, varSize)
        elif varType == 'float':
            return self._slm.assignExternalPointerArray_f(varName, varSize)
        elif varType == 'double':
            return self._slm.assignExternalPointerArray_d(varName, varSize)
        elif varType == 'long double':
            return self._slm.assignExternalPointerArray_ld(varName, varSize)
        else:
            raise TypeError('unsupported varType "{}"'.format(varType))

    def _stepTimeGPU(self):
        """Make one simulation step (for library built for CPU)"""
        self._slm.stepTimeGPU()
    
    def _stepTimeCPU(self):
        """Make one simulation step (for library built for CPU)"""
        self._slm.stepTimeCPU()

    def stepTime(self):
        """Make one simulation step"""
        pass

    def pullStateFromDevice(self, popName):
        """Pull state from the device for a given population"""
        if not self._built:
            raise Exception("GeNN model has to be built before running")
        if not self._cpuOnly:
            self._slm.pullStateFromDevice(popName)
    
    def pullSpikesFromDevice(self, popName):
        """Pull spikes from the device for a given population"""
        if not self._built:
            raise Exception("GeNN model has to be built before running")
        if not self._cpuOnly:
            self._slm.pullSpikesFromDevice(popName)

    def pullCurrentSpikesFromDevice(self, popName):
        """Pull spikes from the device for a given population"""
        if not self._built:
            raise Exception("GeNN model has to be built before running")
        if not self._cpuOnly:
            self._slm.pullCurrentSpikesFromDevice(popName)

    def pushStateToDevice(self, popName):
        """Push state to the device for a given population"""
        if not self._built:
            raise Exception("GeNN model has to be built before running")
        if not self._cpuOnly:
            self._slm.pushStateToDevice(popName)

    def pushSpikesToDevice(self, popName):
        """Push spikes from the device for a given population"""
        if not self._built:
            raise Exception("GeNN model has to be built before running")
        if not self._cpuOnly:
            self._slm.pushSpikesToDevice(popName)

    def pushCurrentSpikesFromDevice(self, popName):
        """Push spikes from the device for a given population"""
        if not self._built:
            raise Exception("GeNN model has to be built before running")
        if not self._cpuOnly:
            self._slm.pushCurrentSpikesToDevice(popName)

    def end(self):
        """Free memory"""
        for group in [self.neuronPopulations, self.currentSources]:
            for groupName, groupData in iteritems(group):

                for egpName, egpData in iteritems(groupData.extraGlobalParams):
                    # if auto allocation is not enabled, let the user care about
                    # freeing of the EGP
                    if egpData.needsAllocation:
                        self._slm.freeExtraGlobalParam(groupName, egpName)
        # "normal" variables are freed when SharedLibraryModel is destoyed

def initVar(initVarSnippet, paramSpace):
    """This helper function creates a VarInit object
    to easily initialise a variable using a snippet.

    Args:
    initVarSnippet --   type of the InitVarSnippet class as string or instance of class
                        derived from InitVarSnippet::Custom class.
    paramSpace --       dict with param values for the InitVarSnippet class
    """
    # Prepare snippet
    (sInstance, sType, paramNames, params) =\
        prepareSnippet(initVarSnippet, paramSpace,
                       genn_wrapper.InitVarSnippet)

    # Use add function to create suitable VarInit
    return VarInit(sInstance, params)

def initConnectivity(initSparseConnectivitySnippet, paramSpace):
    """This helper function creates a InitSparseConnectivitySnippet::Init object
    to easily initialise connectivity using a snippet.

    Args:
    initSparseConnectivitySnippet --    type of the InitSparseConnectivitySnippet class as string or instance of class
                                        derived from InitSparseConnectivitySnippet::Custom class.
    paramSpace --                       dict with param values for the InitSparseConnectivitySnippet class
    """
    # Prepare snippet
    (sInstance, sType, paramNames, params) =\
        prepareSnippet(initSparseConnectivitySnippet, paramSpace,
                       genn_wrapper.InitSparseConnectivitySnippet)

    # Use add function to create suitable VarInit
    return Init(sInstance, params)

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
    if not isinstance(custom_body, dict) and custom_body is not None:
        raise ValueError("custom_body must be an isinstance of dict or None")

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
        body['getxtraGlobalParams'] = lambda self: genn_wrapper.StlContainers.StringPairVector(
                [genn_wrapper.StlContainers.StringPair(egp[0], egp[1]) for egp in extraGlobalParams])
 
    if additionalInputVars:
        body['getAdditionalInputVars'] = lambda self: genn_wrapper.StlContainers.StringStringDoublePairPairVector(
                [genn_wrapper.StlContainers.StringStringDoublePairPair(aiv[0], genn_wrapper.StlContainers.StringDoublePair(aiv[1], aiv[2])) \
                        for aiv in additionalInputVars] )

    if isPoisson is not None:
        body['isPoisson'] = lambda self: isPoisson
    
    if custom_body is not None:
        body.update(custom_body)

    return createCustomModelClass(
            className,
            genn_wrapper.NeuronModels.Custom,
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
    sa createCustomCurrentSourceClass
    sa createCustomInitVarSnippetClass

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
    if not isinstance(custom_body, dict) and custom_body is not None:
        raise ValueError()

    body = {}

    if decayCode is not None:
        body['getDecayCode'] = lambda self: decayCode
    
    if applyInputCode is not None:
        body['getApplyInputCode'] = lambda self: applyInputCode

    if supportCode is not None:
        body['getSupportCode'] = lambda self: supportCode
    
    if custom_body is not None:
        body.update(custom_body)

    return createCustomModelClass(
            className,
            genn_wrapper.PostsynapticModels.Custom,
            paramNames,
            varNameTypes,
            derivedParams,
            body )


def createCustomWeightUpdateClass( 
        className,
        paramNames=None,
        varNameTypes=None,
        preVarNameTypes=None,
        postVarNameTypes=None,
        derivedParams=None,
        simCode=None,
        eventCode=None,
        learnPostCode=None,
        synapseDynamicsCode=None,
        eventThresholdConditionCode=None,
        preSpikeCode=None,
        postSpikeCode=None,
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
    sa createCustomCurrentSourceClass
    sa createCustomInitVarSnippetClass

    Args:
    className           -- name of the new class

    Keyword args:
    paramNames          -- list of strings with param names of the model
    varNameTypes        -- list of pairs of strings with variable names and types of the model
    preVarNameTypes     -- list of pairs of strings with presynaptic variable names and types of the model
    postVarNameTypes    -- list of pairs of strings with postsynaptic variable names and types of the model
    derivedParams       -- list of pairs, where the first member is string with name of
                           the derived parameter and the second MUST be an instance of the class
                           which inherits from libgenn.Snippet.DerivedParamFunc
    simCode             -- string with the simulation code
    eventCode           -- string with the event code
    learnPostCode       -- string with the code to include in learnSynapsePost kernel/function
    synapseDynamicsCode -- string with the synapse dynamics code
    eventThresholdConditionCode -- string with the event threshold condition code
    preSpikeCode                -- string with the code run once per spiking presynaptic neuron
    postSpikeCode               -- string with the code run once per spiking postsynaptic neuron
    simSupportCode -- string with simulation support code
    learnPostSupportCode -- string with support code for learnSynapsePost kernel/function
    synapseDynamicsSuppportCode -- string with synapse dynamics support code
    extraGlobalParams -- list of pairs of strings with names and types of additional parameters
    isPreSpikeTimeRequired -- boolean, is presynaptic spike time required?
    isPostSpikeTimeRequired -- boolean, is postsynaptic spike time required?

    custom_body   -- dictionary with additional attributes and methods of the new class
    """
    if not isinstance(custom_body, dict) and custom_body is not None:
        raise ValueError("custom_body must be an instance of dict or None")

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

    if preSpikeCode is not None:
        body['getPreSpikeCode'] = lambda self: preSpikeCode
    
    if postSpikeCode is not None:
        body['getPostSpikeCode'] = lambda self: postSpikeCode
    
    if simSupportCode is not None:
        body['getSimSupportCode'] = lambda self: simSupportCode

    if learnPostSupportCode is not None:
        body['getLearnPostSupportCode'] = lambda self: learnPostSupportCode

    if synapseDynamicsSuppportCode is not None:
        body['getSynapseDynamicsSuppportCode'] = lambda self: synapseDynamicsSuppportCode

    if extraGlobalParams is not None:
        body['getExtraGlobalParams'] = lambda self: genn_wrapper.StlContainers.StringPairVector(
                [genn_wrapper.StlContainers.StringPair(egp[0], egp[1]) for egp in extraGlobalParams])

    if preVarNameTypes is not None:
        body['getPreVars'] = lambda self: genn_wrapper.StlContainers.StringPairVector([genn_wrapper.StlContainers.StringPair(vn[0], vn[1])
                                                                                        for vn in preVarNameTypes] )
    
    if postVarNameTypes is not None:
        body['getPostVars'] = lambda self: genn_wrapper.StlContainers.StringPairVector([genn_wrapper.StlContainers.StringPair(vn[0], vn[1])
                                                                                        for vn in postVarNameTypes] )
    
    if isPreSpikeTimeRequired is not None:
        body['isPreSpikeTimeRequired'] = lambda self: isPreSpikeTimeRequired

    if isPostSpikeTimeRequired is not None:
        body['isPostSpikeTimeRequired'] = lambda self: isPostSpikeTimeRequired
    
    if custom_body is not None:
        body.update(custom_body)
    
    return createCustomModelClass(
            className,
            genn_wrapper.WeightUpdateModels.Custom,
            paramNames,
            varNameTypes,
            derivedParams,
            body )


def createCustomCurrentSourceClass(
        className,
        paramNames=None,
        varNameTypes=None,
        derivedParams=None,
        injectionCode=None,
        extraGlobalParams=None,
        custom_body=None ):
    """This helper function creates a custom NeuronModel class.

    sa createCustomNeuronClass
    sa createCustomWeightUpdateClass
    sa createCustomCurrentSourceClass
    sa createCustomInitVarSnippetClass

    Args:
    className     -- name of the new class

    Keyword args:
    paramNames    -- list of strings with param names of the model
    varNameTypes  -- list of pairs of strings with varible names and types of the model
    derivedParams -- list of pairs, where the first member is string with name of
                        the derived parameter and the second MUST be an instance of the class
                        which inherits from libgenn.Snippet.DerivedParamFunc
    injectionCode -- string with the current injection code
    extraGlobalParams -- list of pairs of strings with names and types of additional parameters

    custom_body   -- dictionary with additional attributes and methods of the new class
    """
    if not isinstance(custom_body, dict) and custom_body is not None:
        raise ValueError("custom_body must be an instance of dict or None")

    body = {}

    if injectionCode is not None:
        body['getInjectionCode'] = lambda self: injectionCode

    if extraGlobalParams is not None:
        body['getExtraGlobalParams'] = lambda self: genn_wrapper.StlContainers.StringPairVector(
                [genn_wrapper.StlContainers.StringPair(egp[0], egp[1]) for egp in extraGlobalParams])

    if custom_body is not None:
        body.update(custom_body)

    return createCustomModelClass(
            className,
            genn_wrapper.CurrentSourceModels.Custom,
            paramNames,
            varNameTypes,
            derivedParams,
            body )

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
    sa createCustomCurrentSourceClass
    sa createCustomInitVarSnippetClass

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
    def ctor(self):
        base.__init__(self)

    body = {
            '__init__' : ctor,
    }
    
    if paramNames is not None:
        body['getParamNames'] = lambda self: genn_wrapper.StlContainers.StringVector(paramNames)

    if varNameTypes is not None:
        body['getVars'] = lambda self: genn_wrapper.StlContainers.StringPairVector([genn_wrapper.StlContainers.StringPair(vn[0], vn[1]) for vn in varNameTypes])

    if derivedParams is not None:
        body['getDerivedParams'] = lambda self: genn_wrapper.StlContainers.StringDPFPairVector(
                [genn_wrapper.StlContainers.StringDPFPair(dp[0], genn_wrapper.Snippet.makeDPF(dp[1]))
                    for dp in derivedParams] )

    if custom_body is not None:
        body.update(custom_body)

    return type(className, (base,), body)


def createDPFClass(dpfunc):

    """Helper function to create derived parameter function class

    Args:
    dpfunc -- a function which computes the derived parameter and takes
                two args "pars" (vector of double) and "dt" (double)
    """

    def ctor(self):
        genn_wrapper.Snippet.DerivedParamFunc.__init__(self)

    def call(self, pars, dt):
        return dpfunc(pars, dt)

    return type('', (genn_wrapper.Snippet.DerivedParamFunc,), {'__init__' : ctor, '__call__' : call})


def createCMLFClass(cmlfFunc):

    """Helper function to create function class for calculating sizes of
    matrices initialised with sparse connectivity initialisation snippet

    Args:
    cmlfFunc -- a function which computes the length and takes
                three args "numPre" (unsigned int), "numPost" (unsigned int)
                and "pars" (vector of double)
    """

    def ctor(self):
        genn_wrapper.InitSparseConnectivitySnippet.CalcMaxLengthFunc.__init__(self)

    def call(self, numPre, numPost, pars):
        return cmlfFunc(numPre, numPost, pars, dt)

    return type('', (genn_wrapper.InitSparseConnectivitySnippet.CalcMaxLengthFunc,), {'__init__' : ctor, '__call__' : call})

def createCustomInitVarSnippetClass(
        className,
        paramNames=None,
        derivedParams=None,
        varInitCode=None,
        custom_body=None ):
    """This helper function creates a custom InitVarSnippet class.

    sa createCustomNeuronClass
    sa createCustomWeightUpdateClass
    sa createCustomPostsynapticClass
    sa createCustomCurrentSourceClass

    Args:
    className     -- name of the new class

    Keyword args:
    paramNames    -- list of strings with param names of the model
    derivedParams -- list of pairs, where the first member is string with name of
                        the derived parameter and the second MUST be an instance of the class
                        which inherits from libgenn.Snippet.DerivedParamFunc
    varInitcode       -- string with the variable initialization code
    custom_body       -- dictionary with additional attributes and methods of the new class
    """

    if not isinstance(custom_body, dict) and custom_body is not None:
        raise ValueError("custom_body must be an instance of dict or None")

    body = {}

    if getCode is not None:
        body['getCode'] = lambda self: varInitCode

    if custom_body is not None:
        body.update(custom_body)

    return createCustomModelClass(
            className,
            genn_wrapper.InitVarSnippet.Custom,
            paramNames,
            None,
            derivedParams,
            body )

def createCustomSparseConnectInitSnippetClass(
        className,
        paramNames=None,
        derivedParams=None,
        rowBuildCode=None,
        rowBuildStateVars=None,
        calcMaxRowLengthFunc=None,
        calcMaxColLengthFunc=None,
        extraGlobalParams=None,
        custom_body=None ):
    """This helper function creates a custom InitSparseConnectivitySnippet class.

    sa createCustomNeuronClass
    sa createCustomWeightUpdateClass
    sa createCustomPostsynapticClass
    sa createCustomCurrentSourceClass

    Args:
    className     -- name of the new class

    Keyword args:
    paramNames    -- list of strings with param names of the model
    derivedParams -- list of pairs, where the first member is string with name of
                        the derived parameter and the second MUST be an instance of the class
                        which inherits from libgenn.Snippet.DerivedParamFunc
    rowBuildCode      -- string with the row building initialization code
    rowBuildStateVars -- list of tuples of state variables, their types and
                         their initial values to use across row building loop
    extraGlobalParams -- list of pairs of strings with names and types of additional parameters

    custom_body       -- dictionary with additional attributes and methods of the new class
    """

    if not isinstance(custom_body, dict) and custom_body is not None:
        raise ValueError("custom_body must be an instance of dict or None")

    body = {}

    if rowBuildCode is not None:
        body['getRowBuildCode'] = lambda self: rowBuildCode

    if rowBuildStateVars is not None:
        body['getRowBuildStateVars'] =\
            lambda self: genn_wrapper.StlContainers.StringStringDoublePairPairVector(
                [genn_wrapper.StlContainers.StringStringDoublePairPair(r[0], genn_wrapper.StlContainers.StringDoublePair(r[1], r[2]))
                 for r in rowBuildStateVars] )

    if calcMaxRowLengthFunc is not None:
        body["getCalcMaxRowLengthFunc"] = lambda self: genn_wrapper.InitSparseConnectivitySnippet.makeCMLF(calcMaxRowLengthFunc)

    if calcMaxColLengthFunc is not None:
        body["getCalcMaxColLengthFunc"] = lambda self: genn_wrapper.InitSparseConnectivitySnippet.makeCMLF(calcMaxColLengthFunc)

    if extraGlobalParams is not None:
        body['getExtraGlobalParams'] =\
            lambda self: genn_wrapper.StlContainers.StringPairVector(
                [genn_wrapper.StlContainers.StringStringDoublePairPair(egp[0], egp[1])
                 for egp in extraGlobalParams] )

    if custom_body is not None:
        body.update(custom_body)

    return createCustomModelClass(
            className,
            genn_wrapper.InitVarSnippet.Custom,
            paramNames,
            None,
            derivedParams,
            body )
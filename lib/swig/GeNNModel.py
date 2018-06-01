import libgenn as lg
import SharedLibraryModel as slm
from os import path
from subprocess import check_call
import numpy as np

class GeNNModel( object ):

    def __init__(self, scalar, modelName = None):
        self._scalar = scalar
        if scalar == 'float':
            self._slm = slm.SharedLibraryModel_f()
            self._npType = np.float32
        elif scalar == 'double':
            self._slm = slm.SharedLibraryModel_d()
            self._npType = np.float64
        else:
            raise ValueError( 'unknown scalar type "{0}"'.format( scalar ) )
        
        self._localhost = lg.init_cuda_mpi()
        
        #lg.cvar.defaultVarMode = lg.VarMode_LOC_HOST_DEVICE_INIT_DEVICE
        lg.setDefaultVarMode( lg.VarMode_LOC_HOST_DEVICE_INIT_DEVICE )
        lg.initGeNN()
        self._model = lg.NNmodel()
        self._modelName = modelName
        if modelName is not None:
            self._model.setName( modelName )
        self._neuronPopulations = {}
        self._synapsePopulations = {}
        self._supportedNeurons = list(lg.NeuronModels.getSupportedNeurons())
        self._supportedPostsyn = list(lg.PostsynapticModels.getSupportedPostsyn())
        self._supportedWUpdate = list(lg.WeightUpdateModels.getSupportedWUpdate())
        self._built = False


    def setModelName( self, modelName ):
        if self._built:
            raise Exception("GeNN model already built")
        self._modelName = modelName
        self._model.setName( modelName )


    def setModelDT( self, dt ):
        if self._built:
            raise Exception("GeNN model already built")
        self._model.setDT( dt )


    def addNeuronPopulation(self, popName, neuronType, numNeurons, neuronParamValues, neuronInitVarValues):
        if self._built:
            raise Exception("GeNN model already built")
        if popName in self._neuronPopulations:
            raise ValueError( 'neuron population "{0}" already exists'.format( popName ) )

        if neuronType not in self._supportedNeurons:
            raise ValueError( 'neuron model "{0}" is not supported'.format( neuronType ) )

        params = eval( 'lg.NeuronModels.make_' + neuronType + '_ParamValues( ' + ','.join( [str(p) for p in neuronParamValues] ) + ')' )
        ini = eval( 'lg.NeuronModels.make_' + neuronType + '_VarValues( ' + ','.join( [str(v) for v in neuronInitVarValues] ) + ')' )
        
        eval( 'self._model.addNeuronPopulation_' + neuronType + '( popName, numNeurons, params, ini )' ) 

        self._neuronPopulations[popName] = {'NT' : neuronType, 'nN' : numNeurons, 'nParams' : len(neuronParamValues), 'nVars' : len(neuronInitVarValues), 'VOI' : None }
    
    def setVarsOfInterest( self, popName, varNames ):
        if popName not in self._neuronPopulations:
            raise ValueError( 'population "{0}" does not exists'.format(popName) )
        self._neuronPopulations[popName]['VOI'] = varNames

    def addSynapsePopulation(self, popName, matrixType, delaySteps, source, target,
                wUpdateType, wuParamValues, wuInitVarValues,
                postsynType, postsynParamValues, postsynInitVarValues):
        if self._built:
            raise Exception("GeNN model already built")

        if popName in self._synapsePopulations:
            raise ValueError( 'synapse population "{0}" already exists'.format( popName ) )

        if wUpdateType not in self._supportedWUpdate:
            raise ValueError( 'weightUpdate model "{0}" is not supported'.format( wUpdateType ) )
        
        if postsynType not in self._supportedPostsyn:
            raise ValueError( 'postsynaptic model "{0}" is not supported'.format( postsynType ) )

        ps_params = eval( 'lg.PostsynapticModels.make_' + postsynType + '_ParamValues( ' + ','.join( [str(p) for p in postsynParamValues] ) + ')' )
        ps_ini = eval( 'lg.PostsynapticModels.make_' + postsynType + '_VarValues( ' + ','.join( [str(v) for v in postsynInitVarValues] ) + ')' )

        wu_params = eval( 'lg.WeightUpdateModels.make_' + wUpdateType + '_ParamValues( ' + ','.join( [str(p) for p in wuParamValues] ) + ')' )
        wu_ini = eval( 'lg.WeightUpdateModels.make_' + wUpdateType + '_VarValues( ' + ','.join( [str(v) for v in wuInitVarValues] ) + ')' )

        mType = eval( "lg.SynapseMatrixType_" + matrixType )


        eval( 'self._model.addSynapsePopulation_' + wUpdateType + '_' + postsynType + '( popName, mType, delaySteps, source, target, wu_params, wu_ini, ps_params, ps_ini )' ) 
        

        # TODO: add synapse population properly. What info do we need for shared model? Do we?
        self._synapsePopulations[popName] = {'WUT': wUpdateType, 'PST' : postsynType, 'src' : source, 'trg' : target, 'delay' : delaySteps, 'mType' : matrixType }



    def initializeVarOnDevice( self, popName, varName, initData ):
        if popName not in self._neuronPopulations and \
            popName not in self._synapsePopulations:
            raise ValueError( 'Failed to initialize variable "{0}": population "{1}" does not exist'.format( varName, popName ) )
        #  if popName in self._neuronPopulations:
        #      popData = self._neuronPopulations[popName]
        #  else:
        #      popData = self._synapsePopulations[popName]
        arr_in = np.array( initData, dtype = self._npType ).flatten()
        self._slm.pushVarToDevice( popName, varName, arr_in )

    def initializeSpikesOnDevice( self, popName, numSpikes, initData ):
        arr_in = np.array( initData, dtype=np.uint32 ).flatten()
        self._slm.pushSpikesToDevice( popName, numSpikes, arr_in )

    def build( self, pathToModel = "./" ):

        if self._built:
            raise Exception("GeNN model already built")
        
        self._model.finalize()
        lg.chooseDevice(self._model, pathToModel, self._localhost)
        lg.finalize_model_runner_generation(self._model, pathToModel, self._localhost)

        check_call( ['make', '-C', path.join( pathToModel, self._modelName + '_CODE' ) ] )

        self._slm.open( pathToModel, self._modelName )

        self._slm.allocateMem()
        self._slm.initialize()
        
        self._nVars = 0
        self._maxPopSize = 0
        self._totalPopVars = 0
        for popName, popData in self._neuronPopulations.items():
            if popData['nVars'] > 0:
                varNamesTypes = eval( 'lg.NeuronModels.' + popData['NT'] + '().getVars()')
                # TODO: handle VOI
                self._slm.addVars( varNamesTypes, popName, popData['nN'] )
                self._nVars += popData['nVars']
                self._maxPopSize = max( self._maxPopSize, popData['nN'] )
                # if popData['VOI'] is None:
                self._totalPopVars += popData['nVars'] * popData['nN']
                # else:
                #    self._totalPopVars += len( popData['VOI'] ) * popData['nN']

        # TODO: add synapse populations. or do we need it at all?

        self._built = True


    #  def run( self, numSteps ):
    #   if not self._built:
    #       raise Exception( "GeNN model has to be built before running" )
    #   out = np.empty( ( numSteps, self._nVars, self._maxPopSize ), dtype = self._npType )
    #   if not self._slm.runGPUInPlace( out, numSteps ):
    #       return None
    #   ret = {}
    #   popVarStart = 0
    #   for popName, popData in self._neuronPopulations.items():
    #       ret[popName] = out[:,popVarStart : popData['nVars'],:popData['nN']]
    #       popVarStart += popData['nVars']
    #   return ret;


    def run( self, numSteps ):
        if not self._built:
            raise Exception( "GeNN model has to be built before running" )
        out = np.empty( ( numSteps, self._totalPopVars ), dtype = self._npType )
        if not self._slm.runGPUInPlace( out, numSteps ):
            return None
        ret = {}
        popVarStart = 0
        for popName, popData in self._neuronPopulations.items():
            ret[popName] = np.reshape( out[:,popVarStart : popVarStart + popData['nVars'] * popData['nN']], ( numSteps, popData['nVars'], popData['nN'] ) )
            popVarStart += popData['nVars']
        return ret;


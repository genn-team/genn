// GeNN includes
#include "modelSpec.h"

// GeNN userproject includes
#include "../../include/normalDistribution.h"

// Model includes
#include "PotjansMicrocircuitParams.h"

void modelDefinition(NNmodel &model)
{
#ifdef DEBUG
    GENN_PREFERENCES.debugCode = true;
#else
    GENN_PREFERENCES.optimizeCode = true;
#endif // DEBUG

#ifdef _GPU_DEVICE
    GENN_PREFERENCES.deviceSelectMethod = DeviceSelect::MANUAL;
    GENN_PREFERENCES.manualDeviceID = _GPU_DEVICE;
#endif
    model.setDT(Parameters::dtMs);
    model.setName("PotjansMicrocircuit");
    model.setTiming(Parameters::measureTiming);
    model.setDefaultVarLocation(VarLocation::DEVICE);
    model.setDefaultSparseConnectivityLocation(VarLocation::DEVICE);
    model.setMergePostsynapticModels(true);
    model.setDefaultNarrowSparseIndEnabled(true);

    InitVarSnippet::Normal::ParamValues vDist(
        -58.0, // 0 - mean
        5.0);  // 1 - sd

    // LIF initial conditions
    NeuronModels::LIF::VarValues lifInit(
        initVar<InitVarSnippet::Normal>(vDist), // 0 - V
        0.0);                                   // 1 - RefracTime

    CurrentSourceModels::PoissonExp::VarValues poissonInit(0.0);    // 0 - Current

    // Exponential current parameters
    PostsynapticModels::ExpCurr::ParamValues excitatoryExpCurrParams(
        0.5);  // 0 - TauSyn (ms)

    PostsynapticModels::ExpCurr::ParamValues inhibitoryExpCurrParams(
        0.5);  // 0 - TauSyn (ms)

    const double quantile = 0.9999;
    const double maxDelayMs[Parameters::PopulationMax] = {
        Parameters::meanDelay[Parameters::PopulationE] + (Parameters::delaySD[Parameters::PopulationE] * normalCDFInverse(quantile)),
        Parameters::meanDelay[Parameters::PopulationI] + (Parameters::delaySD[Parameters::PopulationI] * normalCDFInverse(quantile))};
    std::cout << "Max excitatory delay: " << maxDelayMs[Parameters::PopulationE] << "ms, max inhibitory delay: " << maxDelayMs[Parameters::PopulationI] << "ms" << std::endl;

    // Calculate maximum dendritic delay slots
    // **NOTE** it seems inefficient using maximum for all but this allows more aggressive merging of postsynaptic models
    const unsigned int maxDendriticDelaySlots = (unsigned int)std::rint(std::max(maxDelayMs[Parameters::PopulationE], maxDelayMs[Parameters::PopulationI])  / Parameters::dtMs);
    std::cout << "Max dendritic delay slots:" << maxDendriticDelaySlots << std::endl;

    // Loop through populations and layers
    std::cout << "Creating neuron populations:" << std::endl;
    unsigned int totalNeurons = 0;
    for(unsigned int layer = 0; layer < Parameters::LayerMax; layer++) {
        for(unsigned int pop = 0; pop < Parameters::PopulationMax; pop++) {
            // Determine name of population
            const std::string popName = Parameters::getPopulationName(layer, pop);

            // Calculate external input rate, weight and current
            const double extInputRate = (Parameters::numExternalInputs[layer][pop] *
                                         Parameters::connectivityScalingFactor *
                                         Parameters::backgroundRate);
            const double extWeight = Parameters::externalW / sqrt(Parameters::connectivityScalingFactor);

            const double extInputCurrent = 0.001 * 0.5 * (1.0 - sqrt(Parameters::connectivityScalingFactor)) * Parameters::getFullMeanInputCurrent(layer, pop);
            assert(extInputCurrent >= 0.0);

            // LIF model parameters
            NeuronModels::LIF::ParamValues lifParams(
                0.25,               // 0 - C
                10.0,               // 1 - TauM
                -65.0,              // 2 - Vrest
                -65.0,              // 3 - Vreset
                -50.0,              // 4 - Vthresh
                extInputCurrent,    // 5 - Ioffset
                2.0);               // 6 - TauRefrac

            CurrentSourceModels::PoissonExp::ParamValues poissonParams(
                extWeight,      // 0 - Weight
                0.5,            // 1 - TauSyn
                extInputRate);  // 2 - Rate

            // Create population
            const unsigned int popSize = Parameters::getScaledNumNeurons(layer, pop);
            auto *neuronPop = model.addNeuronPopulation<NeuronModels::LIF>(popName, popSize,
                                                                           lifParams, lifInit);

            // Add poisson current source population
            model.addCurrentSource<CurrentSourceModels::PoissonExp>(popName + "_poisson", popName,
                                                                    poissonParams, poissonInit);
            // Make recordable on host
            neuronPop->setSpikeRecordingEnabled(true);
            std::cout << "\tPopulation " << popName << ": num neurons:" << popSize << ", external weight:" << extWeight << ", external input rate:" << extInputRate << std::endl;

            // Add number of neurons to total
            totalNeurons += popSize;
        }
    }

    // Loop through target populations and layers
    std::cout << "Creating synapse populations:" << std::endl;
    unsigned int totalSynapses = 0;
    for(unsigned int trgLayer = 0; trgLayer < Parameters::LayerMax; trgLayer++) {
        for(unsigned int trgPop = 0; trgPop < Parameters::PopulationMax; trgPop++) {
            const std::string trgName = Parameters::getPopulationName(trgLayer, trgPop);

            // Loop through source populations and layers
            for(unsigned int srcLayer = 0; srcLayer < Parameters::LayerMax; srcLayer++) {
                for(unsigned int srcPop = 0; srcPop < Parameters::PopulationMax; srcPop++) {
                    const std::string srcName = Parameters::getPopulationName(srcLayer, srcPop);

                    // Determine mean weight
                    const double meanWeight = Parameters::getMeanWeight(srcLayer, srcPop, trgLayer, trgPop) / sqrt(Parameters::connectivityScalingFactor);

                    // Determine weight standard deviation
                    double weightSD;
                    if(srcPop == Parameters::PopulationE && srcLayer == Parameters::Layer4 && trgLayer == Parameters::Layer23 && trgPop == Parameters::PopulationE) {
                        weightSD = meanWeight * Parameters::layer234RelW;
                    }
                    else {
                        weightSD = fabs(meanWeight * Parameters::relW);
                    }

                    // Calculate number of connections
                    const unsigned int numConnections = Parameters::getScaledNumConnections(srcLayer, srcPop, trgLayer, trgPop);

                    if(numConnections > 0) {
                        const double prob = (double)numConnections / ((double)Parameters::getScaledNumNeurons(srcLayer, srcPop) * (double)Parameters::getScaledNumNeurons(trgLayer, trgPop));
                        std::cout << "\tConnection between '" << srcName << "' and '" << trgName << "': numConnections=" << numConnections << "(" << prob << "), meanWeight=" << meanWeight << ", weightSD=" << weightSD << ", meanDelay=" << Parameters::meanDelay[srcPop] << ", delaySD=" << Parameters::delaySD[srcPop] << std::endl;

                        // Build parameters for fixed number total connector
                        InitSparseConnectivitySnippet::FixedNumberTotalWithReplacement::ParamValues connectParams(
                            numConnections);                            // 0 - number of connections

                        totalSynapses += numConnections;

                        // Build unique synapse name
                        const std::string synapseName = srcName + "_" + trgName;

                        // Excitatory
                        if(srcPop == Parameters::PopulationE) {
                            // Build distribution for weight parameters
                            InitVarSnippet::NormalClipped::ParamValues wDist(
                                meanWeight,                                 // 0 - mean
                                weightSD,                                   // 1 - sd
                                0.0,                                        // 2 - min
                                std::numeric_limits<float>::max());         // 3 - max

                            // Build distribution for delay parameters
                            InitVarSnippet::NormalClippedDelay::ParamValues dDist(
                                Parameters::meanDelay[srcPop],              // 0 - mean
                                Parameters::delaySD[srcPop],                // 1 - sd
                                0.0,                                        // 2 - min
                                maxDelayMs[srcPop]);                        // 3 - max


                            // Create weight parameters
                            WeightUpdateModels::StaticPulseDendriticDelay::VarValues staticSynapseInit(
                                initVar<InitVarSnippet::NormalClipped>(wDist),          // 0 - Wij (nA)
                                initVar<InitVarSnippet::NormalClippedDelay>(dDist));    // 1 - delay (ms)

                            // Add synapse population
                            auto *synPop = model.addSynapsePopulation<WeightUpdateModels::StaticPulseDendriticDelay, PostsynapticModels::ExpCurr>(
                                synapseName, SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY, srcName, trgName,
                                {}, staticSynapseInit,
                                excitatoryExpCurrParams, {},
                                initConnectivity<InitSparseConnectivitySnippet::FixedNumberTotalWithReplacement>(connectParams));

                            // Set max dendritic delay and span type
                            synPop->setMaxDendriticDelayTimesteps(maxDendriticDelaySlots);
                        }
                        // Inhibitory
                        else {
                            // Build distribution for weight parameters
                            InitVarSnippet::NormalClipped::ParamValues wDist(
                                meanWeight,                                 // 0 - mean
                                weightSD,                                   // 1 - sd
                                -std::numeric_limits<float>::max(),         // 2 - min
                                0.0);                                       // 3 - max

                            // Build distribution for delay parameters
                            InitVarSnippet::NormalClippedDelay::ParamValues dDist(
                                Parameters::meanDelay[srcPop],              // 0 - mean
                                Parameters::delaySD[srcPop],                // 1 - sd
                                0.0,                                        // 2 - min
                                maxDelayMs[srcPop]);                        // 3 - max

                            // Create weight parameters
                            WeightUpdateModels::StaticPulseDendriticDelay::VarValues staticSynapseInit(
                                initVar<InitVarSnippet::NormalClipped>(wDist),          // 0 - Wij (nA)
                                initVar<InitVarSnippet::NormalClippedDelay>(dDist));    // 1 - delay (ms)

                            // Add synapse population
                            auto *synPop = model.addSynapsePopulation<WeightUpdateModels::StaticPulseDendriticDelay, PostsynapticModels::ExpCurr>(
                                synapseName, SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY, srcName, trgName,
                                {}, staticSynapseInit,
                                inhibitoryExpCurrParams, {},
                                initConnectivity<InitSparseConnectivitySnippet::FixedNumberTotalWithReplacement>(connectParams));

                            // Set max dendritic delay and span type
                            synPop->setMaxDendriticDelayTimesteps(maxDendriticDelaySlots);
                        }

                    }
                }
            }
        }
    }

    std::cout << "Total neurons=" << totalNeurons << ", total synapses=" << totalSynapses << std::endl;
}

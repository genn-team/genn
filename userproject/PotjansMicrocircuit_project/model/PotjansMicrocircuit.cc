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
    GENN_PREFERENCES.generateRuntimePopulationLookup = true;
    model.setDT(Parameters::dtMs);
    model.setName("PotjansMicrocircuit");
    model.setTiming(Parameters::measureTiming);
    model.setDefaultVarLocation(VarLocation::DEVICE);
    model.setDefaultSparseConnectivityLocation(VarLocation::DEVICE);
    model.setMergePostsynapticModels(true);
    model.setDefaultNarrowSparseIndEnabled(true);

    ParamValues vDist{{"mean", -58.0}, {"sd", 5.0}};

    // LIF initial conditions
    VarValues lifInit{{"V",  initVar<InitVarSnippet::Normal>(vDist)},
                      {"RefracTime", 0.0}};

    VarValues poissonInit{{"current", 0.0}};

    // Exponential current parameters
    ParamValues excitatoryExpCurrParams{{"tau", 0.5}};
    ParamValues inhibitoryExpCurrParams{{"tau", 0.5}};

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
            ParamValues lifParams{{"C", 0.25},  {"TauM", 10.0},
                {"Vrest", -65.0}, {"Vreset", -65.0}, {"Vthresh", -50.0},
                {"Ioffset", extInputCurrent}, {"TauRefrac", 2.0}};

            ParamValues poissonParams{{"weight", extWeight}, {"tauSyn", 0.5}, 
                                      {"rate", extInputRate}};

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
                        ParamValues connectParams{{"total", numConnections}};

                        totalSynapses += numConnections;

                        // Build unique synapse name
                        const std::string synapseName = srcName + "_" + trgName;

                        // Excitatory
                        if(srcPop == Parameters::PopulationE) {
                            // Build distribution for weight parameters
                            ParamValues wDist{{"mean", meanWeight}, {"sd", weightSD},
                                              {"min", 0.0}, {"max", std::numeric_limits<float>::max()}};

                            // Build distribution for delay parameters
                            ParamValues dDist{{"mean", Parameters::meanDelay[srcPop]}, {"sd", Parameters::delaySD[srcPop]},
                                              {"min", 0.0}, {"max", maxDelayMs[srcPop]}};


                            // Create weight parameters
                            VarValues staticSynapseInit{
                                {"g", initVar<InitVarSnippet::NormalClipped>(wDist)},
                                {"d", initVar<InitVarSnippet::NormalClippedDelay>(dDist)}};

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
                            ParamValues wDist{{"mean", meanWeight}, {"sd", weightSD},
                                              {"min", -std::numeric_limits<float>::max()}, {"max", 0.0}};

                            // Build distribution for delay parameters
                            ParamValues dDist{{"mean", Parameters::meanDelay[srcPop]}, {"sd", Parameters::delaySD[srcPop]},
                                              {"min", 0.0}, {"max", maxDelayMs[srcPop]}};

                            // Create weight parameters
                            VarValues staticSynapseInit{
                                {"g", initVar<InitVarSnippet::NormalClipped>(wDist)},
                                {"d", initVar<InitVarSnippet::NormalClippedDelay>(dDist)}};

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

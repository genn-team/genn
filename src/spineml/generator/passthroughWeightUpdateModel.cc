#include "passthroughWeightUpdateModel.h"

// Standard C++ includes
#include <iostream>

// SpineML common includes
#include "spineMLLogging.h"

// Spine ML generator includes
#include "neuronModel.h"

//----------------------------------------------------------------------------
// SpineMLGenerator::PassthroughWeightUpdateModel
//----------------------------------------------------------------------------
SpineMLGenerator::PassthroughWeightUpdateModel::PassthroughWeightUpdateModel(const std::string &srcPortName,
                                                                             const NeuronModel *srcNeuronModel,
                                                                             bool heterogeneousDelay)
{
    // If the source neuron model has a send port variable with the specified
    // name, create synapse dynamics code to pass it through synapse
    if(srcNeuronModel->hasSendPortVariable(srcPortName)) {
        LOGD_SPINEML << "\t\tPassing through continuous variable '" << srcPortName << "' to postsynaptic model";

        if(heterogeneousDelay) {
            assert(false);
            m_SynapseDynamicsCode = "$(addToInSynDelay, $(" + srcPortName + "_pre), $(_delay));\n";
        }
        else {
            m_SynapseDynamicsCode = "$(addToInSyn, $(" + srcPortName + "_pre));\n";
        }
    }
    // Otherwise, if the source port is the source neuron's spike send port,
    // create event handler code to add 1 to state variable
    else if(srcNeuronModel->getSendPortSpike() == srcPortName) {
        LOGD_SPINEML << "\t\tPassing through event '" << srcPortName << "' to postsynaptic model";

        if(heterogeneousDelay) {
            assert(false);
            m_SimCode = "$(addToInSynDelay, 1, $(_delay));\n";
        }
        else {
            m_SimCode = "$(addToInSyn, 1);\n";
        }
    }
    else {
        throw std::runtime_error("Passthrough weight update models can only operate on input from analog or event send ports");
    }
}

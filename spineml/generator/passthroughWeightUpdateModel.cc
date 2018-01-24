#include "passthroughWeightUpdateModel.h"

// Standard C++ includes
#include <iostream>

// Spine ML generator includes
#include "neuronModel.h"

//----------------------------------------------------------------------------
// SpineMLGenerator::PassthroughWeightUpdateModel
//----------------------------------------------------------------------------
SpineMLGenerator::PassthroughWeightUpdateModel::PassthroughWeightUpdateModel(const std::string &srcPortName,
                                                                             const NeuronModel *srcNeuronModel)
{
    // If the source neuron model has a send port variable with the specified
    // name, create synapse dynamics code to pass it through synapse
    if(srcNeuronModel->hasSendPortVariable(srcPortName)) {
        std::cout << "\t\tPassing through continuous variable '" << srcPortName << "' to postsynaptic model" << std::endl;

        m_SynapseDynamicsCode =
            "$(addtoinSyn) = $(" + srcPortName + "_pre);\n"
            "$(updatelinsyn);\n";
    }
    // Otherwise, if the source port is the source neuron's spike send port,
    // create event handler code to add 1 to state variable
    else if(srcNeuronModel->getSendPortSpike() == srcPortName) {
        std::cout << "\t\tPassing through event '" << srcPortName << "' to postsynaptic model" << std::endl;

        m_SimCode =
            "$(addtoinSyn) = 1;\n"
            "$(updatelinsyn);\n";
    }
    else {
        throw std::runtime_error("Passthrough weight update models can only operate on input from analog or event send ports");
    }
}
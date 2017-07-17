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
    std::string aliasCode;
    if(srcNeuronModel->hasSendPortVariable(srcPortName)) {
        std::cout << "\t\tPassing through continuous variable '" << srcPortName << "' to postsynaptic model" << std::endl;

        m_SynapseDynamicsCode =
            "$(addtoinSyn) = $(" + srcPortName + "_pre);\n"
            "$(updatelinsyn);\n";
    }
    // Otherwise, if it has an alias with specified name, create synapse dynamics code to calculate alias and pass it through synapse
    // **TODO** could use new weight update model presynaptic variables for this
    else if(srcNeuronModel->getSendPortAlias(srcPortName, "_pre", aliasCode)) {
        std::cout << "\t\tPassing through alias '" << srcPortName << "' to postsynaptic model" << std::endl;
        std::cout << aliasCode << std::endl;

        m_SynapseDynamicsCode =
            "$(addtoinSyn) = " + aliasCode + ";\n"
            "$(updatelinsyn);\n";
    }
    else {
        throw std::runtime_error("Passthrough weight update models can only operate on input from analog send ports");
    }
}
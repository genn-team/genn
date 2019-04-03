/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

#include "modelSpec.h"

#include "sizes.h"

// Snippet for randomizing variables by adding the square of a uniformly distributed offset
class RandomizeSq : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(RandomizeSq, 2);

    SET_CODE(
        "const scalar random = $(gennrand_uniform);\n"
        "$(value) = $(min) + (random * random * $(scale));");

    SET_PARAM_NAMES({"min", "scale"});
};
IMPLEMENT_SNIPPET(RandomizeSq);

//----------------------------------------------------------------------------
// FixedNumberPostWithReplacement
//----------------------------------------------------------------------------
//! Initialises variable by sampling from the uniform distribution
class FixedNumberPostWithReplacement : public InitSparseConnectivitySnippet::Base
{
public:
    DECLARE_SNIPPET(FixedNumberPostWithReplacement, 1);

    SET_ROW_BUILD_CODE(
        "const scalar u = $(gennrand_uniform);\n"
        "x += (1.0 - x) * (1.0 - pow(u, 1.0 / (scalar)($(rowLength) - c)));\n"
        "const unsigned int postIdx = (unsigned int)(x * $(num_post));\n"
        "if(postIdx < $(num_post)) {\n"
        "   $(addSynapse, postIdx);\n"
        "}\n"
        "else {\n"
        "   $(addSynapse, $(num_post) - 1);\n"
        "}\n"
        "c++;\n"
        "if(c >= $(rowLength)) {\n"
        "   $(endRow);\n"
        "}\n");
    SET_ROW_BUILD_STATE_VARS({{"x", {"scalar", 0.0}},{"c", {"unsigned int", 0}}});

    SET_PARAM_NAMES({"rowLength"});

    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int, unsigned int, const std::vector<double> &pars)
        {
            return (unsigned int)pars[0];
        });

    /*SET_CALC_MAX_COL_LENGTH_FUNC(
        [](unsigned int numPre, unsigned int numPost, const std::vector<double> &pars)
        {
            // Calculate suitable quantile for 0.9999 change when drawing numPre times
            const double quantile = pow(0.9999, 1.0 / (double)numPost);

            // There are numConnections connections amongst the numPre*numPost possible connections.
            // Each of the numConnections connections has an independent p=float(numPost)/(numPre*numPost)
            // probability of being selected, and the number of synapses in the sub-row is binomially distributed
            return binomialInverseCDF(quantile, pars[0], (double)numPre / ((double)numPre * (double)numPost));
        });*/
};
IMPLEMENT_SNIPPET(FixedNumberPostWithReplacement);

void modelDefinition(ModelSpec &model) 
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

    // Parameters for gaussian noise current source - excitatory population
    CurrentSourceModels::GaussianNoise::ParamValues IzhExc_curr_par(
        0.0,                // 0 -Mean
        5.0 * _InputFac);   // 1 - SD

    // Parameters for randomizing C parameter - inhibitory population
    RandomizeSq::ParamValues randomizedExcC(
        -65.0,  // 0 - min
        15.0);  // 1 - scale

    // Parameters for randomizing C parameter - inhibitory population
    RandomizeSq::ParamValues randomizedExcD(
        8.0,    // 0 - min
        -6.0);  // 1 - scale

    // Izhikevich model initial conditions - excitatory population
    NeuronModels::IzhikevichVariable::VarValues IzhExc_ini(
        -65.0,                                  // 0 - V
        -65.0 * 0.2,                            // 1 - U
        0.02,                                   // 2 - a
        0.2,                                    // 3 - b
        initVar<RandomizeSq>(randomizedExcC),   // 4 - c
        initVar<RandomizeSq>(randomizedExcD));  // 5 - d

    // Parameters for gaussian noise current source - inhibitory population
    CurrentSourceModels::GaussianNoise::ParamValues IzhInh_curr_par(
        0.0,                // 0 -Mean
        2.0 * _InputFac);    // 1 - SD

    // Parameters for uniformly distributing A parameter - inhibitory population
    InitVarSnippet::Uniform::ParamValues randomizedInhA(
        0.02,           // 0 - min
        0.02 + 0.08);   // 1 - max

    // Parameters for uniformly distributing B parameter - inhibitory population
    InitVarSnippet::Uniform::ParamValues randomizedInhB(
        0.25 - 0.05,    // 0 - min
        0.25);          // 1 - max

    // Izhikevich model initial conditions - inhibitory population
    NeuronModels::IzhikevichVariable::VarValues IzhInh_ini(
        -65.0,                                              // 0 - V
        uninitialisedVar(),                                 // 1 - U
        initVar<InitVarSnippet::Uniform>(randomizedInhA),   // 2 - a
        initVar<InitVarSnippet::Uniform>(randomizedInhB),   // 3 - b
        -65.0,                                              // 4 - c
        2.0);                                               // 5 - d

    // Calculate number of neurons in each population
    const unsigned int nExc = (unsigned int)ceil(4.0 * _NNeurons / 5.0);
    const unsigned int nInh = _NNeurons - nExc;

    // Calculate number of postsynaptic neurons to connect to each pre in connections targetting each population
    const unsigned int nEConn = (unsigned int)ceil(4.0 * _NConn / 5.0);
    const unsigned int nIConn = _NConn - nEConn;

    // Fixed number post with replacement parameters - connections targetting excitatory population
    FixedNumberPostWithReplacement::ParamValues Exc_conn_par(nEConn);

    // Fixed number post with replacement parameters - connections targetting inhibitory population
    FixedNumberPostWithReplacement::ParamValues Inh_conn_par(nIConn);

    // Uniform weights for excitatory weights
    InitVarSnippet::Uniform::ParamValues Exc_weight_par(
        0.0,            // 0 - Min
        0.5 * _GScale); // 1 - Max

    // Uniform weights for inhibitory weights
    InitVarSnippet::Uniform::ParamValues Inh_weight_par(
        -1.0 * _GScale, // 0 - Min
        0.0);           // 1 - Max

    // Weights are initialised from file so don't initialise
    WeightUpdateModels::StaticPulse::VarValues SynExc_ini(
        initVar<InitVarSnippet::Uniform>(Exc_weight_par));

    WeightUpdateModels::StaticPulse::VarValues SynInh_ini(
        initVar<InitVarSnippet::Uniform>(Inh_weight_par));

    model.setName("IzhSparse");
    model.setDT(1.0);
    model.addNeuronPopulation<NeuronModels::IzhikevichVariable>("PExc", nExc, {}, IzhExc_ini);
    model.addCurrentSource<CurrentSourceModels::GaussianNoise>("PExcCurr", "PExc",
                                                               IzhExc_curr_par, {});

    model.addNeuronPopulation<NeuronModels::IzhikevichVariable>("PInh", nInh, {}, IzhInh_ini);
    model.addCurrentSource<CurrentSourceModels::GaussianNoise>("PInhCurr", "PInh",
                                                               IzhInh_curr_par, {});

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Exc_Exc", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "PExc", "PExc",
        {}, SynExc_ini,
        {}, {},
        initConnectivity<FixedNumberPostWithReplacement>(Exc_conn_par));
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Exc_Inh", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "PExc", "PInh",
        {}, SynExc_ini,
        {}, {},
        initConnectivity<FixedNumberPostWithReplacement>(Inh_conn_par));
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Inh_Exc", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "PInh", "PExc",
        {}, SynInh_ini,
        {}, {},
        initConnectivity<FixedNumberPostWithReplacement>(Exc_conn_par));
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Inh_Inh",  SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "PInh", "PInh",
        {}, SynInh_ini,
        {}, {},
        initConnectivity<FixedNumberPostWithReplacement>(Inh_conn_par));

    model.setPrecision(_FTYPE);
    model.setTiming(_TIMING);
}

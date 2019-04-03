#include "../include/generateRun.h"

class GenerateRun : public GenerateRunBase
{
public:
    GenerateRun()
    :   GenerateRunBase("IzhSparse"), m_NumNeurons(10000), m_NumConnections(1000), m_GScale(1.0), m_InputFac(1.0)
    {
        getApp().add_option("--num-neurons", m_NumNeurons, "Number of neurons to simulate", true);
        getApp().add_option("--num-connections", m_NumConnections, "Number of connections per neuron", true);
        getApp().add_option("--gscale", m_GScale, "Scaling of synaptic conductances", true);
        getApp().add_option("--input-fac", m_InputFac, "Input factor", true);
    }

    //------------------------------------------------------------------------
    // GenerateRunBase virtuals
    //------------------------------------------------------------------------
    virtual void writeSizes(std::ofstream &sizes) const override
    {
        // Superclass
        GenerateRunBase::writeSizes(sizes);

        sizes << "#define _NNeurons " << m_NumNeurons << std::endl;
        sizes << "#define _NConn " << m_NumConnections << std::endl;
        sizes << "#define _GScale " << m_GScale << std::endl;
        sizes << "#define _InputFac " << m_InputFac << std::endl;
    }

private:
    unsigned int m_NumNeurons;
    unsigned int m_NumConnections;
    double m_GScale;
    double m_InputFac;
};

int main(int argc, char *argv[])
{
    // Parse command line
    GenerateRun generateRun;
    try {
        generateRun.parseCommandLine(argc, argv);
    }
    catch(const CLI::ParseError &e) {
        return generateRun.getExitCode(e);
    }

    // Write model sizes
    {
        std::ofstream sizes("model/sizes.h");
        generateRun.writeSizes(sizes);
    }

    // Build and run model
    return generateRun.buildAndRun();
}

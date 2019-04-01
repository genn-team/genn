#include "../include/generateRun.h"

class GenerateRun : public GenerateRunBase
{
public:
    GenerateRun()
    :   GenerateRunBase("IzhSparse")
    {
        getApp().add_option("numNeurons", m_NumNeurons, "Number of neurons to simulate", true)->required();
        getApp().add_option("numConnections", m_NumConnections, "Number of connections per neuron", true)->required();
        getApp().add_option("gscale", m_GScale, "Scaling of synaptic conductances", true)->required();
        getApp().add_option("inputFac", m_InputFac, "Input factor", true)->required();
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

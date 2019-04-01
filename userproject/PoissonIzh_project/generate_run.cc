#include "../include/generateRun.h"

class GenerateRun : public GenerateRunBase
{
public:
    GenerateRun()
    :   GenerateRunBase("PoissonIzh")
    {
        getApp().add_option("numPoisson", m_NumPoisson, "Number of Poisson sources to simulate", true)->required();
        getApp().add_option("numIzh", m_NumIzh, "Number of Izhikievich neurons to simulate", true)->required();
        getApp().add_option("pConn", m_PConn, "Probability of connection between poisson source and neuron", true)->required();
        getApp().add_option("gscale", m_GScale, "Scaling of synaptic conductances", true)->required();
    }

    //------------------------------------------------------------------------
    // GenerateRunBase virtuals
    //------------------------------------------------------------------------
    virtual void writeSizes(std::ofstream &sizes) const override
    {
        // Superclass
        GenerateRunBase::writeSizes(sizes);

        sizes << "#define _NPoisson " << m_NumPoisson << std::endl;
        sizes << "#define _NIzh " << m_NumIzh << std::endl;
        sizes << "#define _PConn " << m_PConn << std::endl;
        sizes << "#define _GScale " << m_GScale << std::endl;
    }

private:
    unsigned int m_NumPoisson;
    unsigned int m_NumIzh;
    double m_PConn;
    double m_GScale;
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

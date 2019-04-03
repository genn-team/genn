#include "../include/generateRun.h"

class GenerateRun : public GenerateRunBase
{
public:
    GenerateRun()
    :   GenerateRunBase("PoissonIzh"), m_NumPoisson(100), m_NumIzh(10), m_PConn(0.5), m_GScale(2.0), m_Sparse(false)
    {
        getApp().add_option("--num-poisson", m_NumPoisson, "Number of Poisson sources to simulate", true);
        getApp().add_option("--num-izh", m_NumIzh, "Number of Izhikievich neurons to simulate", true);
        getApp().add_option("--pconn", m_PConn, "Probability of connection between poisson source and neuron", true);
        getApp().add_option("--gscale", m_GScale, "Scaling of synaptic conductances", true);
        getApp().add_flag("--sparse", m_Sparse, "Use sparse rather than dense connectivity");
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

        if(m_Sparse) {
            sizes << "#define _SPARSE_CONNECTIVITY" << std::endl;
        }
    }

private:
    unsigned int m_NumPoisson;
    unsigned int m_NumIzh;
    double m_PConn;
    double m_GScale;
    bool m_Sparse;
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

#include "../include/generateRun.h"

class GenerateRun : public GenerateRunBase
{
public:
    GenerateRun()
    :   GenerateRunBase("PotjansMicrocircuit"), m_NeuronScale(0.5), m_ConnectivityScale(0.5), m_DurationMs(1000.0)
    {
        getApp().add_option("--neuron-scale", m_NeuronScale, "Scaling factor for number of neurons", true);
        getApp().add_option("--connectivity-scale", m_ConnectivityScale, "Scaling factor for connectivity", true);
        getApp().add_option("--duration", m_DurationMs, "Duration of simulation [ms]", true);
    }

    //------------------------------------------------------------------------
    // GenerateRunBase virtuals
    //------------------------------------------------------------------------
    virtual void writeSizes(std::ofstream &sizes) const override
    {
        // Superclass
        GenerateRunBase::writeSizes(sizes);

        sizes << "#define _NeuronScale " << m_NeuronScale << std::endl;
        sizes << "#define _ConnectivityScale " << m_ConnectivityScale << std::endl;
        sizes << "#define _DurationMs " << m_DurationMs << std::endl;
    }

private:
    double m_NeuronScale;
    double m_ConnectivityScale;
    double m_DurationMs;
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

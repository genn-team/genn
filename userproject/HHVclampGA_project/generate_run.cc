#include "../include/generateRun.h"

class GenerateRun : public GenerateRunBase
{
public:
    GenerateRun()
    :   GenerateRunBase("HHVClamp")
    {
        getApp().add_option("protocol", m_Protocol, "Protocol to run")->required();
        getApp().add_option("numPops", m_NumPops, "Number of populations to simulate")->required();
        getApp().add_option("totalTime", m_TotalTime, "Total time to simulate for")->required();
    }

    //------------------------------------------------------------------------
    // GenerateRunBase virtuals
    //------------------------------------------------------------------------
    virtual void writeSizes(std::ofstream &sizes) const override
    {
        // Superclass
        GenerateRunBase::writeSizes(sizes);

        sizes << "#define NPOP " << m_NumPops << std::endl;
        sizes << "#define TOTALT " << m_TotalTime << std::endl;
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    int getProtocol() const{ return m_Protocol; }

private:
    int m_Protocol;
    unsigned int m_NumPops;
    double m_TotalTime;
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
        std::ofstream sizes("model/HHVClampParameters.h");
        generateRun.writeSizes(sizes);
    }

    // Build and run model
    return generateRun.buildAndRun({std::to_string(generateRun.getProtocol())});
}

#include "../include/generateRun.h"

class GenerateRun : public GenerateRunBase
{
public:
    GenerateRun()
    :   GenerateRunBase("OneComp"), m_NumNeurons(1)
    {
        getApp().add_option("--num-neurons", m_NumNeurons, "Number of neurons to simulate", true);
    }

    //------------------------------------------------------------------------
    // GenerateRunBase virtuals
    //------------------------------------------------------------------------
    virtual void writeSizes(std::ofstream &sizes) const override
    {
        // Superclass
        GenerateRunBase::writeSizes(sizes);

        sizes << "#define _NN " << m_NumNeurons << std::endl;
    }

private:
    unsigned int m_NumNeurons;
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

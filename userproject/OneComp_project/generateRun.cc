#include "../include/generateRun.h"

class GenerateRun : public GenerateRunBase
{
public:
    GenerateRun()
    :   GenerateRunBase("OneComp")
    {
    }

    //------------------------------------------------------------------------
    // GenerateRunBase virtuals
    //------------------------------------------------------------------------
    virtual int parseCommandLine(int argc, char *argv[]) override
    {
        getApp().add_option("numNeurons", m_NumNeurons, "Number of neurons to simulate", true)->required();

        // Superclass
        return GenerateRunBase::parseCommandLine(argc, argv);
    }

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
    const int parseRetVal = generateRun.parseCommandLine(argc, argv);
    if(parseRetVal != EXIT_SUCCESS) {
        return parseRetVal;
    }

    // Write model sizes
    {
        std::ofstream sizes("model/sizes.h");
        generateRun.writeSizes(sizes);
    }

    // Build and run model
    return generateRun.buildAndRun();
}

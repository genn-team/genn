// Filesystem includes
#include "path.h"

// Google test includes
#include "gtest/gtest.h"

// pugixml includes
#include "pugixml/pugixml.hpp"

// SpineML simulator includes
#include "connectors.h"

using namespace SpineMLSimulator;

//------------------------------------------------------------------------
// Anonymous namespace
//------------------------------------------------------------------------
namespace
{
void checkOneToOne(const unsigned int *rowLength, const unsigned int *ind, unsigned int numPre)
{
    for(unsigned int i = 0; i < numPre; i++) {
        EXPECT_EQ(rowLength[i], 1);
        EXPECT_EQ(ind[i], i);
    }
}

void checkTriangle(const unsigned int *rowLength, const unsigned int *ind, unsigned int maxRowLength, unsigned int numPre)
{
    for(unsigned int i = 0; i < numPre; i++) {
        EXPECT_EQ(rowLength[i], i);
        for(unsigned int j = 0; j < rowLength[i]; j++){
            EXPECT_EQ(ind[(i * maxRowLength) + j], j);
        }
    }
}
}   // Anonymous namespace

//------------------------------------------------------------------------
// FixedProbabilityConnection tests
//------------------------------------------------------------------------
/*TEST(FixedProbabilityConnectionTest, LowProbabilitySparse) {
    // XML fragment specifying connector
    const char *connectorXML =
        "<LL:Synapse>\n"
        "   <FixedProbabilityConnection probability=\"0.1\" seed=\"123\"/>\n"
        "</LL:Synapse>\n";

    // Load XML and get root LL:Synapse element
    pugi::xml_document connectorDocument;
    connectorDocument.load_string(connectorXML);
    auto synapse = connectorDocument.child("LL:Synapse");

    unsigned int *rowLength = new unsigned int[42];
    unsigned int *ind
    //const pugi::xml_node &node, unsigned int numPre, unsigned int numPost,
    //unsigned int **rowLength, unsigned int **ind, unsigned int *maxRowLength,
    //const filesystem::path &basePath, std::vector<unsigned int> &remapIndices
    // Parse XML and create sparse connector
    filesystem::path basePath;
    std::vector<unsigned int> remapIndices;
    Connectors::create(synapse, 42, 42,
                       &sparse42, allocate42, basePath,
                       remapIndices);

    delete [] rowLength;
}
//------------------------------------------------------------------------
TEST(FixedProbabilityConnectionTest, LowProbabilityDenseDeath) {
    // XML fragment specifying connector
    const char *connectorXML =
        "<LL:Synapse>\n"
        "   <FixedProbabilityConnection probability=\"0.1\" seed=\"123\"/>\n"
        "</LL:Synapse>\n";

    // Load XML and get root LL:Synapse element
    pugi::xml_document connectorDocument;
    connectorDocument.load_string(connectorXML);
    auto synapse = connectorDocument.child("LL:Synapse");

    // Parse XML and create sparse connector
    filesystem::path basePath;
    try
    {
        std::vector<unsigned int> remapIndices;
        Connectors::create(synapse, 42, 42,
                           nullptr, nullptr, basePath,
                           remapIndices);
        FAIL();
    }
    catch(const std::runtime_error &)
    {
    }
}*/
//------------------------------------------------------------------------
TEST(FixedProbabilityConnectionTest, FullyConnectedDense) {
    // XML fragment specifying connector
    const char *connectorXML =
        "<LL:Synapse>\n"
        "   <FixedProbabilityConnection probability=\"1.0\" seed=\"123\"/>\n"
        "</LL:Synapse>\n";

    // Load XML and get root LL:Synapse element
    pugi::xml_document connectorDocument;
    connectorDocument.load_string(connectorXML);
    auto synapse = connectorDocument.child("LL:Synapse");

    // Parse XML and check connector creation correctly realises
    // this should be implemented as dense matrix and thus
    // doesn't require SparseProjection or allocation function
    filesystem::path basePath;
    std::vector<unsigned int> remapIndices;
    unsigned int n = Connectors::create(synapse, 100, 100,
                                        nullptr, nullptr, nullptr,
                                        basePath, remapIndices);
    EXPECT_EQ(n, 100 * 100);
}

//------------------------------------------------------------------------
// AllToAllConnection tests
//------------------------------------------------------------------------
TEST(AllToAllConnectionTest, Dense) {
    // XML fragment specifying connector
    const char *connectorXML =
        "<LL:Synapse>\n"
        "   <AllToAllConnection/>\n"
        "</LL:Synapse>\n";

    // Load XML and get root LL:Synapse element
    pugi::xml_document connectorDocument;
    connectorDocument.load_string(connectorXML);
    auto synapse = connectorDocument.child("LL:Synapse");

    // Parse XML and check connector creation correctly realises
    // this should be implemented as dense matrix and thus
    // doesn't require SparseProjection or allocation function
    filesystem::path basePath;
    std::vector<unsigned int> remapIndices;
    unsigned int n = Connectors::create(synapse, 100, 100,
                                        nullptr, nullptr, nullptr,
                                        basePath, remapIndices);
    EXPECT_EQ(n, 100 * 100);
}
//------------------------------------------------------------------------
TEST(AllToAllConnectionTest, SparseDeath) {
    // XML fragment specifying connector
    const char *connectorXML =
        "<LL:Synapse>\n"
        "   <AllToAllConnection/>\n"
        "</LL:Synapse>\n";

    // Load XML and get root LL:Synapse element
    pugi::xml_document connectorDocument;
    connectorDocument.load_string(connectorXML);
    auto synapse = connectorDocument.child("LL:Synapse");

    // Parse XML and check connector creation correctly realises
    // this should be implemented as dense matrix and thus
    // doesn't require SparseProjection or allocation function
    filesystem::path basePath;
    std::vector<unsigned int> remapIndices;
    const unsigned int maxRowLength = 42;
    unsigned int *rowLength = new unsigned int[42];
    unsigned int *ind = new unsigned int[42 * maxRowLength];
    try
    {
        Connectors::create(synapse, 42, 42,
                           &rowLength, &ind, &maxRowLength,
                           basePath, remapIndices);
        FAIL();
    }
    catch(const std::runtime_error &)
    {
    }

    delete [] ind;
    delete [] rowLength;
}

//------------------------------------------------------------------------
// OneToOneConnection tests
//------------------------------------------------------------------------
TEST(OneToOneConnectionTest, DenseDeath) {
    // XML fragment specifying connector
    const char *connectorXML =
        "<LL:Synapse>\n"
        "   <OneToOneConnection/>\n"
        "</LL:Synapse>\n";

    // Load XML and get root LL:Synapse element
    pugi::xml_document connectorDocument;
    connectorDocument.load_string(connectorXML);
    auto synapse = connectorDocument.child("LL:Synapse");

    // Parse XML and check connector creation correctly realises
    // this should be implemented as dense matrix and thus
    // doesn't require SparseProjection or allocation function
    filesystem::path basePath;
    std::vector<unsigned int> remapIndices;
    try
    {
        Connectors::create(synapse, 100, 100,
                           nullptr, nullptr, nullptr,
                           basePath, remapIndices);
        FAIL();
    }
    catch(const std::runtime_error &)
    {
    }
}

//------------------------------------------------------------------------
// ConnectionList tests
//------------------------------------------------------------------------
TEST(ConnectionListTest, DenseDeath) {
    // XML fragment specifying connector
    const char *connectorXML =
        "<LL:Synapse>\n"
        "   <BinaryFile file_name=\"connection1.bin\" num_connections=\"294\" explicit_delay_flag=\"0\" packed_data=\"\"/>\n"
        "</LL:Synapse>\n";

    // Load XML and get root LL:Synapse element
    pugi::xml_document connectorDocument;
    connectorDocument.load_string(connectorXML);
    auto synapse = connectorDocument.child("LL:Synapse");

    // Parse XML and check connector creation correctly realises
    // this should be implemented as dense matrix and thus
    // doesn't require SparseProjection or allocation function
    filesystem::path basePath;
    std::vector<unsigned int> remapIndices;
    try
    {
        Connectors::create(synapse, 42, 42,
                           nullptr, nullptr, nullptr,
                           basePath, remapIndices);
        FAIL();
    }
    catch(const std::runtime_error &)
    {
    }
}
//------------------------------------------------------------------------
TEST(ConnectionListTest, InlineOneToOneSparse) {
    // XML fragment specifying connector
    const char *connectorXML =
        "<LL:Synapse>\n"
        "   <ConnectionList>\n"
        "       <Connection src_neuron=\"0\" dst_neuron=\"0\"/>\n"
        "       <Connection src_neuron=\"1\" dst_neuron=\"1\"/>\n"
        "       <Connection src_neuron=\"2\" dst_neuron=\"2\"/>\n"
        "       <Connection src_neuron=\"3\" dst_neuron=\"3\"/>\n"
        "       <Connection src_neuron=\"4\" dst_neuron=\"4\"/>\n"
        "       <Connection src_neuron=\"5\" dst_neuron=\"5\"/>\n"
        "       <Connection src_neuron=\"6\" dst_neuron=\"6\"/>\n"
        "       <Connection src_neuron=\"7\" dst_neuron=\"7\"/>\n"
        "       <Connection src_neuron=\"8\" dst_neuron=\"8\"/>\n"
        "       <Connection src_neuron=\"9\" dst_neuron=\"9\"/>\n"
        "   </ConnectionList>\n"
        "</LL:Synapse>\n";

    // Load XML and get root LL:Synapse element
    pugi::xml_document connectorDocument;
    connectorDocument.load_string(connectorXML);
    auto synapse = connectorDocument.child("LL:Synapse");

    // Parse XML and create sparse connector
    filesystem::path basePath;
    std::vector<unsigned int> remapIndices;
    const unsigned int maxRowLength = 1;
    unsigned int *rowLength = new unsigned int[10];
    unsigned int *ind = new unsigned int[10 * maxRowLength];
    Connectors::create(synapse, 10, 10,
                       &rowLength, &ind, &maxRowLength,
                       basePath, remapIndices);

    // Check number of connections matches XML
    EXPECT_EQ(remapIndices.size(), 10);
    checkOneToOne(rowLength, ind, 10);
    delete [] ind;
    delete [] rowLength;
}
//------------------------------------------------------------------------
TEST(ConnectionListTest, InlineTriangleSparse) {
    // XML fragment specifying connector
    const char *connectorXML =
        "<LL:Synapse>\n"
        "   <ConnectionList>\n"
        "       <Connection src_neuron=\"1\" dst_neuron=\"0\"/>\n"

        "       <Connection src_neuron=\"2\" dst_neuron=\"0\"/>\n"
        "       <Connection src_neuron=\"2\" dst_neuron=\"1\"/>\n"

        "       <Connection src_neuron=\"3\" dst_neuron=\"0\"/>\n"
        "       <Connection src_neuron=\"3\" dst_neuron=\"1\"/>\n"
        "       <Connection src_neuron=\"3\" dst_neuron=\"2\"/>\n"

        "       <Connection src_neuron=\"4\" dst_neuron=\"0\"/>\n"
        "       <Connection src_neuron=\"4\" dst_neuron=\"1\"/>\n"
        "       <Connection src_neuron=\"4\" dst_neuron=\"2\"/>\n"
        "       <Connection src_neuron=\"4\" dst_neuron=\"3\"/>\n"
        "   </ConnectionList>\n"
        "</LL:Synapse>\n";

    // Load XML and get root LL:Synapse element
    pugi::xml_document connectorDocument;
    connectorDocument.load_string(connectorXML);
    auto synapse = connectorDocument.child("LL:Synapse");

    // Parse XML and create sparse connector
    filesystem::path basePath;
    std::vector<unsigned int> remapIndices;
    const unsigned int maxRowLength = 4;
    unsigned int *rowLength = new unsigned int[5];
    unsigned int *ind = new unsigned int[5 * maxRowLength];
    Connectors::create(synapse, 5, 5,
                       &rowLength, &ind, &maxRowLength,
                       basePath, remapIndices);

    // Check number of connections matches XML
    EXPECT_EQ(remapIndices.size(), 10);
    checkTriangle(rowLength, ind, maxRowLength, 5);
    delete [] ind;
    delete [] rowLength;
}
//------------------------------------------------------------------------
TEST(ConnectionListTest, InlineShuffleTriangleSparse) {
    // XML fragment specifying connector
    const char *connectorXML =
        "<LL:Synapse>\n"
        "   <ConnectionList>\n"
        "       <Connection src_neuron=\"4\" dst_neuron=\"3\"/>\n"
        "       <Connection src_neuron=\"1\" dst_neuron=\"0\"/>\n"
        "       <Connection src_neuron=\"3\" dst_neuron=\"0\"/>\n"
        "       <Connection src_neuron=\"4\" dst_neuron=\"2\"/>\n"
        "       <Connection src_neuron=\"2\" dst_neuron=\"0\"/>\n"
        "       <Connection src_neuron=\"4\" dst_neuron=\"0\"/>\n"
        "       <Connection src_neuron=\"3\" dst_neuron=\"1\"/>\n"
        "       <Connection src_neuron=\"4\" dst_neuron=\"1\"/>\n"
        "       <Connection src_neuron=\"2\" dst_neuron=\"1\"/>\n"
        "       <Connection src_neuron=\"3\" dst_neuron=\"2\"/>\n"
        "   </ConnectionList>\n"
        "</LL:Synapse>\n";

    // Load XML and get root LL:Synapse element
    pugi::xml_document connectorDocument;
    connectorDocument.load_string(connectorXML);
    auto synapse = connectorDocument.child("LL:Synapse");

    // Parse XML and create sparse connector
    filesystem::path basePath;
    std::vector<unsigned int> remapIndices;
    const unsigned int maxRowLength = 4;
    unsigned int *rowLength = new unsigned int[5];
    unsigned int *ind = new unsigned int[5 * maxRowLength];
    Connectors::create(synapse, 5, 5,
                       &rowLength, &ind, &maxRowLength,
                       basePath, remapIndices);

    // Check that resultant connectivity is triangle
    checkTriangle(rowLength, ind, maxRowLength, 5);

    // Check number of connections matches XML and that remapping is correct
    EXPECT_EQ(remapIndices.size(), 10);
    EXPECT_EQ(remapIndices[0], (4 * maxRowLength) + 3);
    EXPECT_EQ(remapIndices[1], (1 * maxRowLength) + 0);
    EXPECT_EQ(remapIndices[2], (3 * maxRowLength) + 0);
    EXPECT_EQ(remapIndices[3], (4 * maxRowLength) + 2);
    EXPECT_EQ(remapIndices[4], (2 * maxRowLength) + 0);
    EXPECT_EQ(remapIndices[5], (4 * maxRowLength) + 0);
    EXPECT_EQ(remapIndices[6], (3 * maxRowLength) + 1);
    EXPECT_EQ(remapIndices[7], (4 * maxRowLength) + 1);
    EXPECT_EQ(remapIndices[8], (2 * maxRowLength) + 1);
    EXPECT_EQ(remapIndices[9], (3 * maxRowLength) + 2);

    delete [] ind;
    delete [] rowLength;
}
//------------------------------------------------------------------------
TEST(ConnectionListTest, InlineShuffleOneToOneSparse) {
    // XML fragment specifying connector
    const char *connectorXML =
        "<LL:Synapse>\n"
        "   <ConnectionList>\n"
        "       <Connection src_neuron=\"6\" dst_neuron=\"6\"/>\n"
        "       <Connection src_neuron=\"1\" dst_neuron=\"1\"/>\n"
        "       <Connection src_neuron=\"7\" dst_neuron=\"7\"/>\n"
        "       <Connection src_neuron=\"3\" dst_neuron=\"3\"/>\n"
        "       <Connection src_neuron=\"4\" dst_neuron=\"4\"/>\n"
        "       <Connection src_neuron=\"9\" dst_neuron=\"9\"/>\n"
        "       <Connection src_neuron=\"8\" dst_neuron=\"8\"/>\n"
        "       <Connection src_neuron=\"0\" dst_neuron=\"0\"/>\n"
        "       <Connection src_neuron=\"5\" dst_neuron=\"5\"/>\n"
        "       <Connection src_neuron=\"2\" dst_neuron=\"2\"/>\n"
        "   </ConnectionList>\n"
        "</LL:Synapse>\n";

    // Load XML and get root LL:Synapse element
    pugi::xml_document connectorDocument;
    connectorDocument.load_string(connectorXML);
    auto synapse = connectorDocument.child("LL:Synapse");

    // Parse XML and create sparse connector
    filesystem::path basePath;
    std::vector<unsigned int> remapIndices;
    const unsigned int maxRowLength = 1;
    unsigned int *rowLength = new unsigned int[10];
    unsigned int *ind = new unsigned int[10 * maxRowLength];
    Connectors::create(synapse, 10, 10,
                       &rowLength, &ind, &maxRowLength,
                       basePath, remapIndices);

    // Check that resultant connectivity is one-to-one
    checkOneToOne(rowLength, ind, 10);

    // Check number of connections matches XML and that remapping is correct
    EXPECT_EQ(remapIndices.size(), 10);
    EXPECT_EQ(remapIndices[0], 6);
    EXPECT_EQ(remapIndices[1], 1);
    EXPECT_EQ(remapIndices[2], 7);
    EXPECT_EQ(remapIndices[3], 3);
    EXPECT_EQ(remapIndices[4], 4);
    EXPECT_EQ(remapIndices[5], 9);
    EXPECT_EQ(remapIndices[6], 8);
    EXPECT_EQ(remapIndices[7], 0);
    EXPECT_EQ(remapIndices[8], 5);
    EXPECT_EQ(remapIndices[9], 2);

    delete [] ind;
    delete [] rowLength;
}
//------------------------------------------------------------------------
/*TEST(ConnectionListTest, BinaryFileSparse) {
    // XML fragment specifying connector
    const char *connectorXML =
        "<LL:Synapse>\n"
        "   <ConnectionList>\n"
        "       <BinaryFile file_name=\"connection1.bin\" num_connections=\"294\" explicit_delay_flag=\"0\" packed_data=\"\"/>\n"
        "   </ConnectionList>\n"
        "</LL:Synapse>\n";

    // Load XML and get root LL:Synapse element
    pugi::xml_document connectorDocument;
    connectorDocument.load_string(connectorXML);
    auto synapse = connectorDocument.child("LL:Synapse");

    // Parse XML and create sparse connector
    filesystem::path basePath;
    std::vector<unsigned int> remapIndices;
    Connectors::create(synapse, 42, 42,
                       &sparse42, allocate42, basePath,
                       remapIndices);

    // Check number of connections matches XML
    EXPECT_EQ(remapIndices.size(), 294);
}*/

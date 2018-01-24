// Filesystem includes
#include "filesystem/path.h"

// Google test includes
#include "gtest/gtest.h"

// pugixml includes
#include "pugixml/pugixml.hpp"

// GeNN includes
#include "sparseProjection.h"

// SpineML simulator includes
#include "connectors.h"

using namespace SpineMLSimulator;

//------------------------------------------------------------------------
// Anonymous namespace
//------------------------------------------------------------------------
namespace
{
SparseProjection sparse5;
SparseProjection sparse10;
SparseProjection sparse42;

void allocate5(unsigned int connN){
    sparse5.connN = connN;
    sparse5.indInG = new unsigned int[6];
    sparse5.ind = new unsigned int[connN];
    sparse5.preInd = NULL;
    sparse5.revIndInG = NULL;
    sparse5.revInd = NULL;
    sparse5.remap = NULL;
}

void allocate10(unsigned int connN){
    sparse10.connN = connN;
    sparse10.indInG = new unsigned int[11];
    sparse10.ind = new unsigned int[connN];
    sparse10.preInd = NULL;
    sparse10.revIndInG = NULL;
    sparse10.revInd = NULL;
    sparse10.remap = NULL;
}

void allocate42(unsigned int connN){
    sparse42.connN = connN;
    sparse42.indInG = new unsigned int[43];
    sparse42.ind = new unsigned int[connN];
    sparse42.preInd = NULL;
    sparse42.revIndInG = NULL;
    sparse42.revInd = NULL;
    sparse42.remap = NULL;
}

void checkOneToOne(const SparseProjection &projection, unsigned int num)
{
    EXPECT_EQ(projection.connN, num);
    for(unsigned int i = 0; i < num; i++) {
        EXPECT_EQ(projection.ind[i], i);
        EXPECT_EQ(projection.indInG[i], i);
    }
    EXPECT_EQ(projection.indInG[num], num);
}

void checkTriangle(const SparseProjection &projection, unsigned int num)
{
    for(unsigned int i = 0; i < num; i++) {
        const unsigned int rowLength = projection.indInG[i + 1] - projection.indInG[i];
        EXPECT_EQ(rowLength, i);
        for(unsigned int j = 0; j < rowLength; j++){
            EXPECT_EQ(projection.ind[projection.indInG[i] + j], j);
        }
    }
}
}   // Anonymous namespace

//------------------------------------------------------------------------
// FixedProbabilityConnection tests
//------------------------------------------------------------------------
TEST(FixedProbabilityConnectionTest, LowProbabilitySparse) {
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
    std::vector<unsigned int> remapIndices;
    Connectors::create(synapse, 42, 42,
                       &sparse42, allocate42, basePath,
                       remapIndices);
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
}
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
                                        nullptr, nullptr, basePath,
                                        remapIndices);
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
                                        nullptr, nullptr, basePath,
                                        remapIndices);
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
    try
    {
        Connectors::create(synapse, 42, 42,
                           &sparse42, allocate42, basePath,
                           remapIndices);
        FAIL();
    }
    catch(const std::runtime_error &)
    {
    }
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
                           nullptr, nullptr, basePath,
                           remapIndices);
        FAIL();
    }
    catch(const std::runtime_error &)
    {
    }
}
//------------------------------------------------------------------------
TEST(OneToOneConnectionTest, Sparse) {
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
    Connectors::create(synapse, 42, 42,
                       &sparse42, allocate42, basePath,
                       remapIndices);

    checkOneToOne(sparse42, 42);
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
                           nullptr, nullptr, basePath,
                           remapIndices);
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
    Connectors::create(synapse, 10, 10,
                       &sparse10, allocate10, basePath,
                       remapIndices);

    // Check number of connections matches XML
    EXPECT_EQ(sparse10.connN, 10);
    checkOneToOne(sparse10, 10);
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
    Connectors::create(synapse, 5, 5,
                       &sparse5, allocate5, basePath,
                       remapIndices);

    // Check number of connections matches XML
    EXPECT_EQ(sparse5.connN, 10);
    checkTriangle(sparse5, 5);
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
    Connectors::create(synapse, 5, 5,
                       &sparse5, allocate5, basePath,
                       remapIndices);

    // Check number of connections matches XML
    EXPECT_EQ(sparse5.connN, 10);
    checkTriangle(sparse5, 5);
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
    Connectors::create(synapse, 10, 10,
                       &sparse10, allocate10, basePath,
                       remapIndices);

    // Check number of connections matches XML
    EXPECT_EQ(sparse10.connN, 10);
    checkOneToOne(sparse10, 10);
}
//------------------------------------------------------------------------
TEST(ConnectionListTest, BinaryFileSparse) {
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
    EXPECT_EQ(sparse42.connN, 294);
}

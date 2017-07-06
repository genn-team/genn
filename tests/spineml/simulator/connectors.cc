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
SparseProjection sparse42;

void allocate42(unsigned int connN){
    sparse42.connN = connN;
    sparse42.indInG = new unsigned int[43];
    sparse42.ind = new unsigned int[connN];
    sparse42.preInd = NULL;
    sparse42.revIndInG = NULL;
    sparse42.revInd = NULL;
    sparse42.remap = NULL;
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
    Connectors::create(synapse, 42, 42,
                       &sparse42, allocate42, basePath);
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
        Connectors::create(synapse, 42, 42,
                           nullptr, nullptr, basePath);
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
    unsigned int n = Connectors::create(synapse, 100, 100,
                                        nullptr, nullptr, basePath);
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
    unsigned int n = Connectors::create(synapse, 100, 100,
                                        nullptr, nullptr, basePath);
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
    try
    {
        Connectors::create(synapse, 42, 42,
                           &sparse42, allocate42, basePath);
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
    try
    {
        Connectors::create(synapse, 100, 100,
                           nullptr, nullptr, basePath);
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
    Connectors::create(synapse, 42, 42,
                       &sparse42, allocate42, basePath);

    // Check structure
    EXPECT_EQ(sparse42.connN, 42);
    for(unsigned int i = 0; i < 42; i++) {
        EXPECT_EQ(sparse42.ind[i], i);
        EXPECT_EQ(sparse42.indInG[i], i);
    }
    EXPECT_EQ(sparse42.indInG[42], 42);
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
    try
    {
        Connectors::create(synapse, 42, 42,
                           nullptr, nullptr, basePath);
        FAIL();
    }
    catch(const std::runtime_error &)
    {
    }
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
    Connectors::create(synapse, 42, 42,
                       &sparse42, allocate42, basePath);

    // Check number of connections matches XML
    EXPECT_EQ(sparse42.connN, 294);
}

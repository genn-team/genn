#include "networkClient.h"

// Standard C++ includes
#include <iostreams>

// POSIX includes
#ifdef _WIN32
    #include <winsock2.h>
#else
    #include <arpa/inet.h>
    #include <netinet/in.h>
    #include <sys/socket.h>
    #include <sys/types.h>
    #include <unistd.h>
#endif

//----------------------------------------------------------------------------
// SpineMLSimulator::NetworkClient
//----------------------------------------------------------------------------
bool SpineMLSimulator::NetworkClient::connect(const std::string &hostname, int port, int size, DataType dataType, Mode mode, const std::string &connectionName)
{
    // Create socket
    m_Socket = socket(AF_INET, SOCK_STREAM, 0);
    if(m_Socket < 0) {
        std::cerr << "Unable to create socket" << std::endl;
        return false;
    }
    
    // Disable Nagle algorithm
    // **THINK** should we?
    const int disableNagle = 1;
    if(setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char *>(&flag), sizeof(int)) < 0) {
        std::cerr << "Unable to set socket options" << std::endl;
        return false;
    }
    
    // Create address structure
    sockaddr_in destAddress = {
        .sin_family = AF_INET, 
        .sin_port = htons(port), 
        .sin_addr = { .s_addr = inet_addr(hostname.c_str()) }};
   
   // Connect socket
   if (connect(m_Socket, reinterpret_cast<sockaddr*>(&destAddress), sizeof(destAddress)) < 0) {
        std::cerr << "Unable to connect to " << hostname << ":" << port << std::endl;
        return false;
    }
    
    // Handshake
    Response handshakeResponse;
    if(!sendRequestReadResponse(mode, handshakeResponse)) {
        return false;
    }
    // Check response is a hello
    if(handshakeResponse != Response::Hello) {
        std::cerr << "Invalid handshake response:" << handshakeResponse << std::endl;
        return false;
    }
    
    // Send data type
    Response dataTypeResponse;
    if(!sendRequestReadResponse(dataType, dataTypeResponse)) {
        return false;
    }
    // Check it's not an abort
    if(dataTypeResponse == Response::Abort) {
        std::cerr << "Remote host aborted" << std::endl;
        return false;
    }
    
    // Send size
    Response sizeResponse;
    if(!sendRequestReadResponse(size, sizeResponse)) {
        return false;
    }
    // Check it's not an abort
    if(sizeResponse == Response::Abort) {
        std::cerr << "Remote host aborted" << std::endl;
        return false;
    }
    
    // Send connection name
    Response connectionNameResponse;
    if(!sendRequestReadResponse(connectionName, connectionNameResponse)) {
        return false;
    }
    // Check it's not an abort
    if(sizeResponse == Response::Abort) {
        std::cerr << "Remote host aborted" << std::endl;
        return false;
    }
}
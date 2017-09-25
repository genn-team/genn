#pragma once

// Standard C++ includes
#include <iostream>
#include <string>
#include <vector>

// POSIX includes
#ifdef _WIN32
    #include <winsock2.h>
#else
    #include <arpa/inet.h>
    #include <netinet/in.h>
    #include <netinet/tcp.h>
    #include <sys/socket.h>
    #include <sys/types.h>
    #include <unistd.h>
#endif

//----------------------------------------------------------------------------
// SpineMLSimulator::NetworkClient
//----------------------------------------------------------------------------
namespace SpineMLSimulator
{
class NetworkClient
{
public:
    //------------------------------------------------------------------------
    // Enumerations
    //------------------------------------------------------------------------
    enum class Mode : char
    {
        Source =  45,
        Target = 46,
    };
    
    enum class DataType : char
    {
        Analogue = 31,
        Events = 32,
        Impulses = 33,
    };
    
    NetworkClient();
    NetworkClient(const std::string &hostname, int port, int size, DataType dataType, Mode mode, const std::string &connectionName);
    ~NetworkClient();
    
    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    bool connect(const std::string &hostname, int port, int size, DataType dataType, Mode mode, const std::string &connectionName);

    bool receive(std::vector<double> &buffer);

private:
    //------------------------------------------------------------------------
    // Enumerations
    //------------------------------------------------------------------------
    enum class Response : char
    {
        Hello = 41,
        Received = 42,
        Abort = 43,
        Finished = 44,
    };
    
    //----------------------------------------------------------------------------
    // Private API
    //----------------------------------------------------------------------------
    template<typename Type>
    bool sendRequestReadResponse(Type data, Response &response)
    {
        // Send request
        if(send(m_Socket, &data, sizeof(Type), MSG_DONTWAIT) < 0) {
            std::cerr << "Unable to send request" << std::endl;
            return false;
        }

        // Receive handshake response
        if (recv(m_Socket, &response, sizeof(Response), MSG_WAITALL) < 1) {
            std::cerr << "Unable to receive response" << std::endl;
            return false;
        }

        return true;
    }

    //----------------------------------------------------------------------------
    // Private members
    //----------------------------------------------------------------------------
    int m_Socket;
};

//----------------------------------------------------------------------------
// Template specializations
//----------------------------------------------------------------------------
template<>
inline bool NetworkClient::sendRequestReadResponse<const std::string &>(const std::string &data, Response &response)
{
    // Send string length
    const int stringLength = data.size();
    if(send(m_Socket, &stringLength, sizeof(int), MSG_DONTWAIT) < 0) {
        std::cerr << "Unable to send size" << std::endl;
        return false;
    }

    // Send string
    if(send(m_Socket, data.c_str(), stringLength, MSG_DONTWAIT) < 0) {
        std::cerr << "Unable to send string" << std::endl;
        return false;
    }

    // Receive handshake response
    if (recv(m_Socket, &response, sizeof(Response), MSG_WAITALL) < 1) {
        std::cerr << "Unable to receive response" << std::endl;
        return false;
    }

    return true;
}
}   // namespace SpineMLSimulator
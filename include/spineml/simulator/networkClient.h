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

// SpineML common includes
#include "spineMLLogging.h"
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
    NetworkClient(const std::string &hostname, unsigned int port, unsigned int size, DataType dataType, Mode mode, const std::string &connectionName);
    ~NetworkClient();
    
    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    bool connect(const std::string &hostname, unsigned int port, unsigned int size, DataType dataType, Mode mode, const std::string &connectionName);

    bool receive(std::vector<double> &buffer);
    bool send(const std::vector<double> &buffer);

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
    int startNonBlockingSend()
    {
#ifdef _WIN32
        // Turn on non-blocking mode
        u_long nonBlockingMode = 1;
        ioctlsocket(m_Socket, FIONBIO, &nonBlockingMode);

        // Don't use flags
        return 0;
#else
        // Use flags
        return MSG_DONTWAIT;
#endif
    }

    void endNonBlockingSend()
    {
#ifdef _WIN32
        // Turn off non-blocking mode
        u_long nonBlockingMode = 0;
        ioctlsocket(m_Socket, FIONBIO, &nonBlockingMode);
#endif
    }

    template<typename Type>
    bool sendRequestReadResponse(Type data, Response &response)
    {
        // Start non-blocking send mode and get flags for send (if any)
        const int sendFlags = startNonBlockingSend();

        // Send request
        if(::send(m_Socket, reinterpret_cast<const char*>(&data), sizeof(Type), sendFlags) < 0) {
            LOGE_SPINEML << "Unable to send request";
            return false;
        }

        // End non-blocking send mode
        endNonBlockingSend();

        // Receive handshake response
        if(::recv(m_Socket, reinterpret_cast<char*>(&response), sizeof(Response), MSG_WAITALL) < 1) {
            LOGE_SPINEML << "Unable to receive response";
            return false;
        }

        return true;
    }

    bool sendRequestReadResponse(const std::string &data, Response &response);

    //----------------------------------------------------------------------------
    // Private members
    //----------------------------------------------------------------------------
    int m_Socket;
};

}   // namespace SpineMLSimulator

#include "networkClient.h"

// Standard C++ includes
#include <stdexcept>

// Standard C includes
#include <cstring>

// SpineML common includes
#include "spineMLLogging.h"

//----------------------------------------------------------------------------
// SpineMLSimulator::NetworkClient
//----------------------------------------------------------------------------
SpineMLSimulator::NetworkClient::NetworkClient() : m_Socket(-1)
{
}
//----------------------------------------------------------------------------
SpineMLSimulator::NetworkClient::NetworkClient(const std::string &hostname, unsigned int port, unsigned int size, DataType dataType, Mode mode, const std::string &connectionName)
{
    if(!connect(hostname, port, size, dataType, mode, connectionName)) {
        throw std::runtime_error("Cannot connect network client");
    }
}
//----------------------------------------------------------------------------
SpineMLSimulator::NetworkClient::~NetworkClient()
{
    // Close socket
    if(m_Socket >= 0) {
#ifdef _WIN32
        closesocket(m_Socket);
#else
        close(m_Socket);
#endif
    }
}
//----------------------------------------------------------------------------
bool SpineMLSimulator::NetworkClient::connect(const std::string &hostname, unsigned int port, unsigned int size, DataType dataType, Mode mode, const std::string &connectionName)
{
    // Create socket
    m_Socket = socket(AF_INET, SOCK_STREAM, 0);
    if(m_Socket < 0) {
        LOGE_SPINEML << "Unable to create socket";
        return false;
    }

    // Disable Nagle algorithm
    // **THINK** should we?
    const int disableNagle = 1;
    if(setsockopt(m_Socket, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&disableNagle), sizeof(int)) < 0) {
        LOGE_SPINEML << "Unable to set socket options";
        return false;
}
    
    // Create address structure
    sockaddr_in destAddress;
    memset(&destAddress, 0, sizeof(sockaddr_in));
    destAddress.sin_family = AF_INET;
    destAddress.sin_port = htons((uint16_t)port);
    destAddress.sin_addr.s_addr = inet_addr(hostname.c_str());

   // Connect socket
   if (::connect(m_Socket, reinterpret_cast<sockaddr*>(&destAddress), sizeof(destAddress)) < 0) {
        LOGE_SPINEML << "Unable to connect to " << hostname << ":" << port;
        return false;
    }

    // Handshake
    Response handshakeResponse;
    if(!sendRequestReadResponse(mode, handshakeResponse)) {
        return false;
    }
    // Check response is a hello
    if(handshakeResponse != Response::Hello) {
        LOGE_SPINEML << "Invalid handshake response:" << static_cast<int>(handshakeResponse);
        return false;
    }

    // Send data type
    Response dataTypeResponse;
    if(!sendRequestReadResponse(dataType, dataTypeResponse)) {
        return false;
    }
    // Check it's not an abort
    if(dataTypeResponse == Response::Abort) {
        LOGE_SPINEML << "Remote host aborted";
        return false;
    }

    // Send size
    Response sizeResponse;
    if(!sendRequestReadResponse(size, sizeResponse)) {
        return false;
    }
    // Check it's not an abort
    if(sizeResponse == Response::Abort) {
        LOGE_SPINEML << "Remote host aborted";
        return false;
    }

    // Send connection name
    Response connectionNameResponse;
    if(!sendRequestReadResponse(connectionName, connectionNameResponse)) {
        return false;
    }
    // Check it's not an abort
    if(connectionNameResponse == Response::Abort) {
        LOGE_SPINEML << "Remote host aborted";
        return false;
    }

    // Success!
    return true;
}
//----------------------------------------------------------------------------
bool SpineMLSimulator::NetworkClient::receive(std::vector<double> &buffer)
{
    // Get buffer size and write pointer as bytes
    const int bufferSizeBytes = buffer.size() * sizeof(double);
    char *bufferBytes = reinterpret_cast<char*>(buffer.data());

    // get data
    int totalReceivedBytes = 0;
    while (totalReceivedBytes < bufferSizeBytes) {
        const int receivedBytes = ::recv(m_Socket, bufferBytes + totalReceivedBytes, bufferSizeBytes, MSG_WAITALL);
        if(receivedBytes < 1) {
            LOGE_SPINEML << "Error reading from socket";
            return false;
        }

        totalReceivedBytes += receivedBytes;
    }

    // Start non-blocking send mode and get flags for send (if any)
    const int sendFlags = startNonBlockingSend();

    // Send response
    const Response response = Response::Received;
    if (::send(m_Socket, reinterpret_cast<const char*>(&response), sizeof(Response), sendFlags) < 1) {
        LOGE_SPINEML << "Error writing to socket";
        return false;
    }

    // End non-blocking send mode
    endNonBlockingSend();

    return true;
}
//----------------------------------------------------------------------------
bool SpineMLSimulator::NetworkClient::send(const std::vector<double> &buffer)
{
    // Start non-blocking send mode and get flags for send (if any)
    const int sendFlags = startNonBlockingSend();

     // Get buffer size and write pointer as bytes
    const int bufferSizeBytes = buffer.size() * sizeof(double);
    const char *bufferBytes = reinterpret_cast<const char*>(buffer.data());

    // send data
    int totalSentBytes = 0;
    while (totalSentBytes < bufferSizeBytes) {
        const int sentBytes = ::send(m_Socket, bufferBytes + totalSentBytes, bufferSizeBytes, sendFlags);
        if(sentBytes < 1) {
            LOGE_SPINEML << "Error writing to socket";
            return false;
        }

        totalSentBytes += sentBytes;
    }

    // End non-blocking send mode
    endNonBlockingSend();

    // Read response
    Response response;
    if (::recv(m_Socket, reinterpret_cast<char*>(&response), sizeof(Response), MSG_WAITALL) < 1) {
        LOGE_SPINEML << "Unable to receive response";
        return false;
    }

    // If response is an abort - error
    if (response == Response::Abort) {
        LOGE_SPINEML << "Remote host aborted";
        return false;
    }
    // Otherwise - success!
    else {
        return true;
    }
}
//----------------------------------------------------------------------------
bool SpineMLSimulator::NetworkClient::sendRequestReadResponse(const std::string &data, Response &response)
{
    // Start non-blocking send mode and get flags for send (if any)
    const int sendFlags = startNonBlockingSend();

    // Send string length
    const int stringLength = data.size();
    if(::send(m_Socket, reinterpret_cast<const char*>(&stringLength), sizeof(int), sendFlags) < 0) {
        LOGE_SPINEML << "Unable to send size";
        return false;
    }

    // Send string
    if(::send(m_Socket, data.c_str(), stringLength, sendFlags) < 0) {
        LOGE_SPINEML << "Unable to send string";
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

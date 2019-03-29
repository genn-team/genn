#pragma once

// Standard C++ includes
#include <fstream>
#include <string>

//----------------------------------------------------------------------------
// AnalogueRecorder
//----------------------------------------------------------------------------
template<typename T>
class AnalogueRecorder
{
public:
    AnalogueRecorder(const std::string &filename,  T *variable, unsigned int popSize)
    : m_Stream(filename), m_Variable(variable), m_PopSize(popSize)
    {
        // Set precision
        m_Stream.precision(16);
    }

    void record(double t)
    {
        m_Stream << t << " ";
        for(unsigned int i = 0; i <  m_PopSize; i++) {
            m_Stream << m_Variable[i] << " ";
        }
        m_Stream << std::endl;
    }

private:

    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::ofstream m_Stream;
    T *m_Variable;
    unsigned int m_PopSize;
};

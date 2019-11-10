#pragma once

// Standard C++ includes
#include <fstream>
#include <initializer_list>
#include <string>
#include <vector>

//----------------------------------------------------------------------------
// AnalogueRecorder
//----------------------------------------------------------------------------
template<typename T>
class AnalogueRecorder
{
public:
    AnalogueRecorder(const std::string &filename, std::initializer_list<T*> variables, unsigned int popSize, const std::string &delimiter=" ")
    :   m_Stream(filename), m_Variables(variables), m_PopSize(popSize), m_Delimiter(delimiter)
    {
        // Set precision
        m_Stream.precision(16);
    }
    AnalogueRecorder(const std::string &filename, T *variable, unsigned int popSize, const std::string &delimiter=" ")
    :   AnalogueRecorder(filename, {variable}, popSize, delimiter)
    {
    }

    void record(double t)
    {
        m_Stream << t << m_Delimiter;

        for(auto *v : m_Variables) {
            for(unsigned int i = 0; i <  m_PopSize; i++) {
                m_Stream << v[i] << m_Delimiter;
            }
        }
        m_Stream << std::endl;
    }

private:

    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::ofstream m_Stream;
    std::vector<T*> m_Variables;
    const unsigned int m_PopSize;
    const std::string m_Delimiter;
};

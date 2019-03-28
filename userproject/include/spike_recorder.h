#pragma once

// Standard C++ includes
#include <algorithm>
#include <fstream>
#include <iterator>
#include <list>
#include <vector>

//----------------------------------------------------------------------------
// SpikeRecorder
//----------------------------------------------------------------------------
class SpikeRecorder
{
public:
    SpikeRecorder(const std::string &filename, const unsigned int *spkCnt, const unsigned int *spk)
    :   m_Stream(filename), m_SpkCnt(spkCnt), m_Spk(spk)
    {
        // Set precision 
        m_Stream.precision(16);
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    void record(double t)
    {
        for(unsigned int i = 0; i < m_SpkCnt[0]; i++) {
            m_Stream << t << " " << m_Spk[i] << std::endl;
        }
    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::ofstream m_Stream;
    const unsigned int *m_SpkCnt;
    const unsigned int *m_Spk;
};

//----------------------------------------------------------------------------
// SpikeRecorderDelay
//----------------------------------------------------------------------------
class SpikeRecorderDelay
{
public:
    SpikeRecorderDelay(const std::string &filename, unsigned int popSize,
                       const unsigned int &spkQueuePtr, const unsigned int *spkCnt, const unsigned int *spk)
    :   m_Stream(filename), m_SpkQueuePtr(spkQueuePtr), m_SpkCnt(spkCnt), m_Spk(spk), m_PopSize(popSize)
    {
        // Set precision
        m_Stream.precision(16);
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    void record(double t)
    {
        const unsigned int *currentSpk = getCurrentSpk();
        for(unsigned int i = 0; i < getCurrentSpkCnt(); i++) {
            m_Stream << t << " " << currentSpk[i] << std::endl;
        }
    }

private:
    //----------------------------------------------------------------------------
    // Private methods
    //----------------------------------------------------------------------------
    const unsigned int *getCurrentSpk() const
    {
        return &m_Spk[m_SpkQueuePtr * m_PopSize];
    }

    unsigned int getCurrentSpkCnt() const
    {
        return m_SpkCnt[m_SpkQueuePtr];
    }

    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::ofstream m_Stream;
    const unsigned int &m_SpkQueuePtr;
    const unsigned int *m_SpkCnt;
    const unsigned int *m_Spk;
    const unsigned int m_PopSize;
};


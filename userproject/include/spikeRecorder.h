#pragma once

// Standard C++ includes
#include <algorithm>
#include <fstream>
#include <iterator>
#include <list>
#include <tuple>
#include <vector>

//----------------------------------------------------------------------------
// SpikeWriterText
//----------------------------------------------------------------------------
//! Class to write spikes to text file
class SpikeWriterText
{
public:
    SpikeWriterText(const std::string &filename, const std::string &delimiter = " ", bool header = false)
    :   m_Stream(filename), m_Delimiter(delimiter)
    {
        // Set precision
        m_Stream.precision(16);

        if(header) {
            m_Stream << "Time [ms], Neuron ID" << std::endl;
        }
    }

protected:
    //----------------------------------------------------------------------------
    // Protected API
    //----------------------------------------------------------------------------
    void recordSpikes(double t, unsigned int spikeCount, const unsigned int *currentSpikes)
    {
        for(unsigned int i = 0; i < spikeCount; i++) {
            m_Stream << t << m_Delimiter << currentSpikes[i] << std::endl;
        }
    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::ofstream m_Stream;
    const std::string m_Delimiter;
};

//----------------------------------------------------------------------------
// SpikeWriterTextCached
//----------------------------------------------------------------------------
//! Class to write spikes to text file, caching in memory before writing
class SpikeWriterTextCached
{
public:
    SpikeWriterTextCached(const std::string &filename, const std::string &delimiter = " ", bool header = false)
    :   m_Stream(filename), m_Delimiter(delimiter)
    {
        // Set precision
        m_Stream.precision(16);

        if(header) {
            m_Stream << "Time [ms], Neuron ID" << std::endl;
        }
    }

    ~SpikeWriterTextCached()
    {
        writeCache();
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    void writeCache()
    {
        // Loop through timesteps
        for(const auto &timestep : m_Cache) {
            // Loop through spikes
            for(unsigned int spike : timestep.second) {
                // Write CSV
                m_Stream << timestep.first << m_Delimiter << spike << std::endl;
            }
        }

        // Clear cache
        m_Cache.clear();
    }

protected:
    //----------------------------------------------------------------------------
    // Protected API
    //----------------------------------------------------------------------------
    void recordSpikes(double t, unsigned int spikeCount, const unsigned int *currentSpikes)
    {
        // Add a new entry to the cache
        m_Cache.emplace_back();

        // Fill in time
        m_Cache.back().first = t;

        // Reserve vector to hold spikes
        m_Cache.back().second.reserve(spikeCount);

        // Copy spikes into vector
        std::copy_n(currentSpikes, spikeCount, std::back_inserter(m_Cache.back().second));
    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::ofstream m_Stream;
    const std::string m_Delimiter;

    std::list<std::pair<double, std::vector<unsigned int>>> m_Cache;
};

//----------------------------------------------------------------------------
// SpikeRecorderBase
//----------------------------------------------------------------------------
//! Class to read spikes from neuron groups
template<typename Writer = SpikeWriterText>
class SpikeRecorder : public Writer
{
public:
    typedef unsigned int& (*GetCurrentSpikeCountFunc)();
    typedef unsigned int* (*GetCurrentSpikesFunc)();
    
    template<typename... WriterArgs>
    SpikeRecorder(GetCurrentSpikesFunc getCurrentSpikes, GetCurrentSpikeCountFunc getCurrentSpikeCount,
                  WriterArgs &&... writerArgs)
    :   Writer(std::forward<WriterArgs>(writerArgs)...), m_GetCurrentSpikes(getCurrentSpikes),
        m_GetCurrentSpikeCount(getCurrentSpikeCount), m_Sum(0)
    {
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    void record(double t)
    {
        const unsigned int spikeCount = m_GetCurrentSpikeCount();
        m_Sum += spikeCount;
        this->recordSpikes(t, spikeCount, m_GetCurrentSpikes());
    }
    
    unsigned int getSum() const{ return m_Sum; }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    GetCurrentSpikesFunc m_GetCurrentSpikes;
    GetCurrentSpikeCountFunc m_GetCurrentSpikeCount;
    unsigned int m_Sum;
};

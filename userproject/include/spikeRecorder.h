#pragma once

// Standard C++ includes
#include <algorithm>
#include <fstream>
#include <iterator>
#include <list>
#include <tuple>
#include <vector>

//----------------------------------------------------------------------------
// SpikeReader
//----------------------------------------------------------------------------
//! Class to read spikes from neuron groups without axonal delays
class SpikeReader
{
public:
    SpikeReader(const unsigned int *spkCnt, const unsigned int *spk)
    :   m_SpkCnt(spkCnt), m_Spk(spk)
    {
    }

    SpikeReader(const std::tuple<const unsigned int*, const unsigned int*> &args)
    :   SpikeReader(std::get<0>(args), std::get<1>(args))
    {
    }

protected:
    //----------------------------------------------------------------------------
    // Protected API
    //----------------------------------------------------------------------------
    const unsigned int *getCurrentSpikes() const
    {
        return m_Spk;
    }

    unsigned int getCurrentSpikeCount() const
    {
        return m_SpkCnt[0];
    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const unsigned int *m_SpkCnt;
    const unsigned int *m_Spk;
};

//----------------------------------------------------------------------------
// SpikeReaderDelayed
//----------------------------------------------------------------------------
//! Class to read spikes from neuron groups with axonal delays
class SpikeReaderDelayed
{
public:
    SpikeReaderDelayed(unsigned int popSize, const unsigned int &spkQueuePtr,
                       const unsigned int *spkCnt, const unsigned int *spk)
    :   m_PopSize(popSize), m_SpkQueuePtr(spkQueuePtr), m_SpkCnt(spkCnt), m_Spk(spk)
    {
    }
    SpikeReaderDelayed(const std::tuple<unsigned int, const unsigned int&, const unsigned int*, const unsigned int*> &args)
    :   SpikeReaderDelayed(std::get<0>(args), std::get<1>(args), std::get<2>(args), std::get<3>(args))
    {
    }

protected:
    //----------------------------------------------------------------------------
    // Protected API
    //----------------------------------------------------------------------------
    const unsigned int *getCurrentSpikes() const
    {
        return &m_Spk[m_SpkQueuePtr * m_PopSize];
    }

    unsigned int getCurrentSpikeCount() const
    {
        return m_SpkCnt[m_SpkQueuePtr];
    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const unsigned int m_PopSize;
    const unsigned int &m_SpkQueuePtr;
    const unsigned int *m_SpkCnt;
    const unsigned int *m_Spk;
};

//----------------------------------------------------------------------------
// SpikeWriterText
//----------------------------------------------------------------------------
//! Class to write spikes to text file
class SpikeWriterText
{
public:
    SpikeWriterText(const std::string &filename, const std::string &delimiter, bool header)
    :   m_Stream(filename), m_Delimiter(delimiter)
    {
        // Set precision
        m_Stream.precision(16);

        if(header) {
            m_Stream << "Time [ms], Neuron ID" << std::endl;
        }
    }

    SpikeWriterText(const std::tuple<const std::string&, const std::string&, bool> &args)
    :   SpikeWriterText(std::get<0>(args), std::get<1>(args), std::get<2>(args))
    {
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
    SpikeWriterTextCached(const std::string &filename, const std::string &delimiter, bool header)
    :   m_Stream(filename), m_Delimiter(delimiter)
    {
        // Set precision
        m_Stream.precision(16);

        if(header) {
            m_Stream << "Time [ms], Neuron ID" << std::endl;
        }
    }

    SpikeWriterTextCached(const std::tuple<const std::string&, const std::string&, bool> &args)
    :   SpikeWriterTextCached(std::get<0>(args), std::get<1>(args), std::get<2>(args))
    {
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
//! Base class for a spike reader, using policy-based design to combine a Reader and Writer class in parallel
/*! **NOTE** while it would be possible to write some fairly hardcore boilerplate to unpack the tuples
    (see http://cpptruths.blogspot.com/2012/06/perfect-forwarding-of-parameter-groups.html), it's not really worth it */
template<typename Reader, typename Writer>
class SpikeRecorderBase : public Reader, public Writer
{
public:
    template<typename... ReaderArgs, typename... WriterArgs>
    SpikeRecorderBase(std::tuple<ReaderArgs...> readerArgs, std::tuple<WriterArgs...> writerArgs)
    :   Reader(readerArgs), Writer(writerArgs)
    {
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    void record(double t)
    {
        m_Sum += this->getCurrentSpikeCount();
        this->recordSpikes(t, this->getCurrentSpikeCount(), this->getCurrentSpikes());
    }

    unsigned int getSum() const{ return m_Sum; }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    unsigned int m_Sum;
};

//----------------------------------------------------------------------------
// SpikeRecorder
//----------------------------------------------------------------------------
class SpikeRecorder : public SpikeRecorderBase<SpikeReader, SpikeWriterText>
{
public:
    SpikeRecorder(const std::string &filename, const unsigned int *spkCnt, const unsigned int *spk,
                  const std::string &delimiter = " ", bool header = false)
    :   SpikeRecorderBase<SpikeReader, SpikeWriterText>(std::forward_as_tuple(spkCnt, spk),
                                                         std::forward_as_tuple(filename, delimiter, header))
    {
    }
};

//----------------------------------------------------------------------------
// SpikeRecorderCached
//----------------------------------------------------------------------------
class SpikeRecorderCached : public SpikeRecorderBase<SpikeReader, SpikeWriterTextCached>
{
public:
    SpikeRecorderCached(const std::string &filename, const unsigned int *spkCnt, const unsigned int *spk,
                        const std::string &delimiter = " ", bool header = false)
    :   SpikeRecorderBase<SpikeReader, SpikeWriterTextCached>(std::forward_as_tuple(spkCnt, spk),
                                                              std::forward_as_tuple(filename, delimiter, header))
    {
    }
};

//----------------------------------------------------------------------------
// SpikeRecorderDelay
//----------------------------------------------------------------------------
class SpikeRecorderDelay : public SpikeRecorderBase<SpikeReaderDelayed, SpikeWriterText>
{
public:
    SpikeRecorderDelay(const std::string &filename, unsigned int popSize,
                       const unsigned int &spkQueuePtr, const unsigned int *spkCnt, const unsigned int *spk,
                       const std::string &delimiter = " ", bool header =  false)
    :   SpikeRecorderBase<SpikeReaderDelayed, SpikeWriterText>(std::forward_as_tuple(popSize, spkQueuePtr, spkCnt, spk),
                                                               std::forward_as_tuple(filename, delimiter, header))
    {
    }
};

//----------------------------------------------------------------------------
// SpikeRecorderDelayCached
//----------------------------------------------------------------------------
class SpikeRecorderDelayCached : public SpikeRecorderBase<SpikeReaderDelayed, SpikeWriterTextCached>
{
public:
    SpikeRecorderDelayCached(const std::string &filename, unsigned int popSize,
                             const unsigned int &spkQueuePtr, const unsigned int *spkCnt, const unsigned int *spk,
                             const std::string &delimiter = " ", bool header =  false)
    :   SpikeRecorderBase<SpikeReaderDelayed, SpikeWriterTextCached>(std::forward_as_tuple(popSize, spkQueuePtr, spkCnt, spk),
                                                                     std::forward_as_tuple(filename, delimiter, header))
    {
    }
};

#pragma once

// Standard C++ includes
#include <memory>

// Standard C includes
#include <cstdint>

// Forward declarations
namespace libcaer::devices
{
class device;
}
namespace GeNN::Runtime
{
class ArrayBase;
}

//----------------------------------------------------------------------------
// GeNN::DVS::DVS
//----------------------------------------------------------------------------
//! Simply interface for reading spikes from DVS sensors supported by LibCAER into GeNN
namespace GeNN::DVS
{
class DVS
{
public:
    //! How to handle event polarity
    enum class Polarity : uint32_t
    {
        ON_ONLY,    //!< Only process on events
        OFF_ONLY,   //!< Only process off events
        SEPERATE,   //!< Process on and off events seperately
        MERGE,      //!< Merge together on and off events
    };    

    //! Rectangle struct used to 
    struct CropRect
    {
        uint32_t left;
        uint32_t top;
        uint32_t right;
        uint32_t bottom;
    };
    
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Start streaming events from DVS
    void start();

    //! Stop streaming events from DVS
    void stop();

    //! Read all events received since last call to readEvents into array
    void readEvents(GeNN::Runtime::ArrayBase *array, Polarity polarity = Polarity::SEPERATE,
                    float scale = 1.0f, const CropRect *cropRect = nullptr);

    //! Get horizontal resolution of DVS
    uint32_t getWidth() const{ return m_Width; }

    //! Get vertical resolution of DVS
    uint32_t getHeight() const{ return m_Height; }

    //------------------------------------------------------------------------
    // Static API
    //------------------------------------------------------------------------
    //! Create DVS interface for camera type
    template<typename D>
    static DVS createDVS(uint16_t deviceID)
    {
        return DVS(std::make_unique<D>(deviceID), 
                   static_cast<uint32_t>(info.dvsSizeX), 
                   static_cast<uint32_t>(info.dvsSizeY));
    }

private:
    DVS(std::unique_ptr<libcaer::devices::device> device,
        uint32_t width, uint32_t height);

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::unique_ptr<libcaer::devices::device> m_Device;
    uint32_t m_Width;
    uuint32_t m_Height;
};
}
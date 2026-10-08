#pragma once

// Standard C++ includes
#include <array>
#include <memory>
#include <optional>

// Standard C includes
#include <cstdint>

// LibCAER includes
#include <libcaercpp/devices/davis.hpp>
#include <libcaercpp/devices/dvs128.hpp>
#include <libcaercpp/devices/dvxplorer.hpp>


// Forward declarations
namespace GeNN::Runtime
{
class ArrayBase;
}

//----------------------------------------------------------------------------
// GeNN::Sensors::DVS
//----------------------------------------------------------------------------
//! Simply interface for reading spikes from DVS sensors supported by LibCAER into GeNN
namespace GeNN::Sensors
{
class DVS
{
public:
    ~DVS()
    {
        stop();
    }

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
        CropRect(){}
        CropRect(const std::array<uint32_t, 4> &cropRect)
        :   left(cropRect[0]), top(cropRect[1]), right(cropRect[2]), bottom(cropRect[3])
        {
        }

        uint32_t left;
        uint32_t top;
        uint32_t right;
        uint32_t bottom;
    };
    
    DVS(std::unique_ptr<libcaer::devices::device> device, 
        uint32_t width, uint32_t height, Polarity polarity,
        float scale, std::optional<CropRect> cropRect);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Start streaming events from DVS
    void start();

    //! Stop streaming events from DVS
    void stop();

    //! Read all events received since last call to readEvents into array
    void readEvents(GeNN::Runtime::ArrayBase *array);

    uint32_t getOutputWidth() const{ return m_OutputWidth; }
    uint32_t getOutputHeight() const{ return m_OutputHeight; }
    uint32_t getOutputChannels() const{ return m_OutputChannels; }
    uint32_t getOutputArrayWords() const{ return m_OutputArrayWords; }
    
    //------------------------------------------------------------------------
    // Static API
    //------------------------------------------------------------------------
    //! Create DVS interface for camera type
    template<typename D>
    static std::unique_ptr<DVS> create(Polarity polarity = Polarity::SEPERATE, float scale = 1.0f, 
                                       std::optional<CropRect> cropRect = std::nullopt,
                                       uint16_t deviceID = 1)
    {
        auto device = std::make_unique<D>(deviceID);
        auto info = device->infoGet();

        return std::make_unique<DVS>(std::move(device), static_cast<uint32_t>(info.dvsSizeX), 
                                     static_cast<uint32_t>(info.dvsSizeY), polarity, scale, cropRect);
    }

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    bool isPolarityCorrect(const libcaer::events::PolarityEvent &event) const;
    bool isInCrop(const libcaer::events::PolarityEvent &event) const;
    std::tuple<uint32_t, uint32_t> scaleEvent(uint32_t x, uint32_t y) const;
    void setEvent(uint32_t x, uint32_t y, bool polarity, uint32_t *array) const;
    void setEvent(uint32_t x, uint32_t y, uint32_t *array) const;
    
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::unique_ptr<libcaer::devices::device> m_Device;

    //! Horizontal resolution of DVS output after scaling, cropping etc
    uint32_t m_OutputWidth;

    //! Vertical resolution of DVS after scaling, cropping etc
    uint32_t m_OutputHeight;

    //! Number of output channels of DVS after scaling cropping etc
    uint32_t m_OutputChannels;

    //! Correct size of output array for this DVS in words
    uint32_t m_OutputArrayWords;

    Polarity m_Polarity;
    float m_Scale;
    std::optional<CropRect> m_CropRect;
};
}

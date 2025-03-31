#include "dvs/dvs.h"

// GeNN code generator includes
// **YUCK**
#include "code_generator/codeGenUtils.h"

// GeNN runtime includes
#include "runtime/runtime.h"

// LibCAER includes
#include <libcaercpp/devices/device.hpp>

using namespace GeNN::DVS;

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
inline bool isPolarityCorrect(const libcaer::events::PolarityEvent &event, DVS::Polarity polarity)
{
    // On event - correct if not set to OFF_ONLY
    if(event.getPolarity()) {
        return (polarity != DVS::Polarity::OFF_ONLY);
    }
    // Off event - correct if not set to ON_ONLY
    else {
        return (polarity != DVS::Polarity::ON_ONLY);
    }
}

inline bool isInCrop(const libcaer::events::PolarityEvent &event, const DVS::CropRect *cropRect) 
{
    return ((event.getX() >= cropRect->left) && (event.getX() < cropRect->right)
            && (event.getY() >= cropRect->top) && (event.getY() < cropRect->bottom));
}

std::tuple<uint32_t, uin32_t> scaleEvent(uint32_t x, uint32_t y,  float scale)
{
    return std::make_tuple(static_cast<uint32_t>(std::round(x * scale)),
                           static_cast<uint32_t>(std::round(y * scale)));
}

inline void setEvent(uint32_t x, uint32_t y, bool polarity, uint32_t outputWidth, uint32_t *array)
{
    const size_t address = (polarity ? 1 : 0) + (x * 2) + (y * 2 * outputWidth);
    array[address / 32] |= (1 << (address % 32));
}

inline void setEvent(uint32_t x, uint32_t y, uint32_t outputWidth, uint32_t *array)
{
    const size_t address = x + (y * outputWidth);
    array[address / 32] |= (1 << (address % 32));
}

template<typename P>
inline void forEachPolarityEvent(const libcaer::events::EventPacketContainer &eventPacketContainer,
                          P onPolarityEventFn)
{
    // Loop through packets
    for(auto &packet : *packetContainer) {
        // If packet's empty, skip
        if (packet == nullptr) {
            continue;
        }
        // Otherwise if this is a polarity event
        else if (packet->getEventType() == POLARITY_EVENT) {
            // Cast to polarity packet
            auto polarityPacket = std::static_pointer_cast<libcaer::events::PolarityEventPacket>(packet);

            // Loop through events
            for(const auto &event : *polarityPacket) {
                onPolarityEventFn(event);
            }
        }
    }
}
}

//----------------------------------------------------------------------------
// GeNN::DVS::DVS
//----------------------------------------------------------------------------
namespace GeNN::DVS
{
void DVS::start()
{
    m_Device->dataStart(nullptr, nullptr, nullptr, nullptr, nullptr);
}
//----------------------------------------------------------------------------
void DVS::stop()
{
    m_Device->dataStop();
}
//----------------------------------------------------------------------------
void DVS::readEvents(GeNN::Runtime::ArrayBase *array, Polarity polarity,
                     float scale, const CropRect *cropRect)
{
    // Determine the output width before scaling
    const uint32_t preScaleWidth = cropRect ? (cropRect->right - cropRect->left) : m_Width;
    const uint32_t preScaleHeight = cropRect ? (cropRect->bottom - cropRect->top) : m_Height;

    // Apply scale
    const uint32_t outputWidth = static_cast<uint32_t>(std::round(preScaleWidth * scale));
    const uint32_t outputHeight = static_cast<uint32_t>(std::round(preScaleHeight * scale));
    const uint32_t outputChannels = (polarity == Polarity::SEPERATE) ? 2 : 1;
    const uint32_t outputPixels = outputWidth * outputHeight * outputChannels;

    // Cast array pointer
    uint32_t *arrayPointer = array->getHostPointer<uint32_t>();


    // Check datatype
    if(array->getType() != Type::Uint32) {
        throw std::runtime_error("DVS interface expects to read "
                                 "events into uint32 'bitmask' array");
    }

    // Check count
    if(array->getCount() != GeNN::CodeGenerator::ceilDivide(outputPixels, 32)) {
        throw std::runtime_error("DVS interface trying to write " + std::to_string(outputPixels)
                                 + " to array with space for " + std::to_string(array->getCount() * 32));
    }

    
    // Get data from DVS
    auto packetContainer = m_Device->dataGet();
    if (packetContainer == nullptr) {
        return;
    }

    // If output will have one polarity channel
    if(polarity != Polarity::SEPERATE) {
        // If we're scaling AND cropping
        if(scale != 1.0f && cropRect) {
            forEachPolarityEvent(
                *packetContainer,
                [=](const auto &event)
                {
                    if(isPolarityCorrect(event, polarity) && isInCrop(event, cropRect)) {
                        const auto [x, y] = scaleEvent(event.getX() - cropRect->x, 
                                                       event.getY() - cropRect->top, scale);
                        setEvent(x, y, outputWidth, arrayPointer);
                    }
                });
        }
        // If we're cropping
        else if(cropRect) {
            forEachPolarityEvent(
                *packetContainer,
                [=](const auto &event)
                {
                    if(isPolarityCorrect(event, polarity) && isInCrop(event, cropRect)) {
                        setEvent(event.getX() - cropRect->left, event.getY() - cropRect->top, 
                                 outputWidth, arrayPointer);
                    }
                });

        }
        // If we're scaling
        else if(scale != 1.0f) {
            forEachPolarityEvent(
                *packetContainer,
                [=](const auto &event)
                {
                    if(isPolarityCorrect(event, polarity)) {
                        const auto [x, y] = scaleEvent(event, scale);
                        setEvent(x, y, outputWidth, arrayPointer);
                    }
                });
        }
        // If we're doing nothing
        else {
            forEachPolarityEvent(
                *packetContainer,
                [=](const auto &event)
                {
                    if(isPolarityCorrect(event, polarity)) {
                        setEvent(event.getX(), event.getY(), outputWidth, arrayPointer);
                    }
                });
        }
    }
    // Otherwise, if output will have two polarity channels
    else {
        // If we're scaling AND cropping
        if(scale != 1.0f && cropRect) {
            forEachPolarityEvent(
                *packetContainer,
                [=](const auto &event)
                {
                    if(isInCrop(event, cropRect)) {
                        const auto [x, y] = scaleEvent(event.getX() - cropRect->x, 
                                                       event.getY() - cropRect->top, scale);
                        setEvent(x, y, event.getPolarity(), 
                                 outputWidth, arrayPointer);
                    }
                });
        }
        // If we're cropping
        else if(cropRect) {
            forEachPolarityEvent(
                *packetContainer,
                [=](const auto &event)
                {
                    if(isInCrop(event, cropRect)) {
                        setEvent(event.getX() - cropRect->left, event.getY() - cropRect->top, 
                                 event.getPolarity(), outputWidth, arrayPointer);
                    }
                });

        }
        // If we're scaling
        else if(scale != 1.0f) {
            forEachPolarityEvent(
                *packetContainer,
                [=](const auto &event)
                {
                    const auto [x, y] = scaleEvent(event, scale);
                    setEvent(x, y, event.getPolarity(), 
                                outputWidth, arrayPointer);
                });
        }
        // If we're doing nothing
        else {
            forEachPolarityEvent(
                *packetContainer,
                [=](const auto &event)
                {
                    setEvent(event.getX(), event.getY(), event.getPolarity(), 
                             outputWidth, arrayPointer);
                });
        }
    }  
}
//----------------------------------------------------------------------------
DVS::DeviceInterface(std::unique_ptr<libcaer::devices::device> device,
                     unsigned int width, unsigned int height)
:   m_Device(device), m_Width(width), m_Height(height)
{
    // Send the default configuration before using the device.
    // No configuration is sent automatically!
    m_Device->sendDefaultConfig();
}
}
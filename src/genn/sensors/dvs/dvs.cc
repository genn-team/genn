#include "dvs.h"

// GeNN code generator includes
// **YUCK**
#include "code_generator/codeGenUtils.h"

// GeNN runtime includes
#include "runtime/runtime.h"

// LibCAER includes
#include <libcaercpp/devices/device.hpp>

using namespace GeNN::Sensors;

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
template<typename P>
inline void forEachPolarityEvent(const libcaer::events::EventPacketContainer &eventPacketContainer,
                                 P onPolarityEventFn)
{
    // Loop through packets
    for(auto &packet : eventPacketContainer) {
        // If packet's empty, skip
        if (packet == nullptr) {
            continue;
        }
        // Otherwise if this is a polarity event
        else if (packet->getEventType() == POLARITY_EVENT) {
            // Cast to polarity packet
            auto polarityPacket = std::static_pointer_cast<const libcaer::events::PolarityEventPacket>(packet);

            // Loop through events
            for(const auto &event : *polarityPacket) {
                onPolarityEventFn(event);
            }
        }
    }
}
}

//----------------------------------------------------------------------------
// GeNN::Sensors::DVS
//----------------------------------------------------------------------------
namespace GeNN::Sensors
{
DVS::DVS(std::unique_ptr<libcaer::devices::device> device,
         unsigned int width, unsigned int height, Polarity polarity,
         float scale, std::optional<CropRect> cropRect)
:   m_Device(std::move(device)), m_Polarity(polarity), m_Scale(scale), m_CropRect(cropRect)
{
    // Applying cropping
    const uint32_t preScaleWidth = m_CropRect ? (m_CropRect->right - m_CropRect->left) : width;
    const uint32_t preScaleHeight = m_CropRect ? (m_CropRect->bottom - m_CropRect->top) : height;

    // Apply scale
    m_OutputWidth = static_cast<uint32_t>(std::round(preScaleWidth * m_Scale));
    m_OutputHeight = static_cast<uint32_t>(std::round(preScaleHeight * m_Scale));
    
    // Determine number of channels based on polarity
    m_OutputChannels = (polarity == Polarity::SEPERATE) ? 2 : 1;
    
    // Calculate correct size of output array in words
    m_OutputArrayWords =  CodeGenerator::ceilDivide(m_OutputWidth * m_OutputHeight * m_OutputChannels, 32);
    
    // Send the default configuration before using the device.
    // No configuration is sent automatically!
    m_Device->sendDefaultConfig();
}
//----------------------------------------------------------------------------
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
void DVS::readEvents(GeNN::Runtime::ArrayBase *array)
{
    // Cast array pointer
    uint32_t *arrayPointer = array->getHostPointer<uint32_t>();


    // Check datatype
    if(array->getType() != Type::Uint32) {
        throw std::runtime_error("DVS interface expects to read "
                                 "events into uint32 'bitmask' array");
    }

    // Check count
    if(array->getCount() != m_OutputArrayWords) {
        throw std::runtime_error("DVS interface trying to write " + std::to_string(m_OutputArrayWords)
                                 + " words to array with space for " + std::to_string(array->getCount()));
    }

    
    // Get data from DVS
    auto packetContainer = m_Device->dataGet();
    if (packetContainer == nullptr) {
        return;
    }

    // If output will have one polarity channel
    if(m_Polarity != Polarity::SEPERATE) {
        // If we're scaling AND cropping
        if(m_Scale != 1.0f && m_CropRect) {
            forEachPolarityEvent(
                *packetContainer,
                [=](const auto &event)
                {
                    if(isPolarityCorrect(event) && isInCrop(event)) {
                        const auto [x, y] = scaleEvent(event.getX() - m_CropRect->left, 
                                                       event.getY() - m_CropRect->top);
                        setEvent(x, y, arrayPointer);
                    }
                });
        }
        // If we're cropping
        else if(m_CropRect) {
            forEachPolarityEvent(
                *packetContainer,
                [=](const auto &event)
                {
                    if(isPolarityCorrect(event) && isInCrop(event)) {
                        setEvent(event.getX() - m_CropRect->left, event.getY() - m_CropRect->top, 
                                 arrayPointer);
                    }
                });

        }
        // If we're scaling
        else if(m_Scale != 1.0f) {
            forEachPolarityEvent(
                *packetContainer,
                [=](const auto &event)
                {
                    if(isPolarityCorrect(event)) {
                        const auto [x, y] = scaleEvent(event.getX(), event.getY());
                        setEvent(x, y, arrayPointer);
                    }
                });
        }
        // If we're doing nothing
        else {
            forEachPolarityEvent(
                *packetContainer,
                [=](const auto &event)
                {
                    if(isPolarityCorrect(event)) {
                        setEvent(event.getX(), event.getY(), arrayPointer);
                    }
                });
        }
    }
    // Otherwise, if output will have two polarity channels
    else {
        // If we're scaling AND cropping
        if(m_Scale != 1.0f && m_CropRect) {
            forEachPolarityEvent(
                *packetContainer,
                [=](const auto &event)
                {
                    if(isInCrop(event)) {
                        const auto [x, y] = scaleEvent(event.getX() - m_CropRect->left, 
                                                       event.getY() - m_CropRect->top);
                        setEvent(x, y, event.getPolarity(), 
                                 arrayPointer);
                    }
                });
        }
        // If we're cropping
        else if(m_CropRect) {
            forEachPolarityEvent(
                *packetContainer,
                [=](const auto &event)
                {
                    if(isInCrop(event)) {
                        setEvent(event.getX() - m_CropRect->left, event.getY() - m_CropRect->top, 
                                 event.getPolarity(), arrayPointer);
                    }
                });

        }
        // If we're scaling
        else if(m_Scale != 1.0f) {
            forEachPolarityEvent(
                *packetContainer,
                [=](const auto &event)
                {
                    const auto [x, y] = scaleEvent(event.getX(), event.getY());
                    setEvent(x, y, event.getPolarity(), 
                                arrayPointer);
                });
        }
        // If we're doing nothing
        else {
            forEachPolarityEvent(
                *packetContainer,
                [=](const auto &event)
                {
                    setEvent(event.getX(), event.getY(), event.getPolarity(), 
                             arrayPointer);
                });
        }
    }  
}
//----------------------------------------------------------------------------
bool DVS::isPolarityCorrect(const libcaer::events::PolarityEvent &event) const
{
    // On event - correct if not set to OFF_ONLY
    if(event.getPolarity()) {
        return (m_Polarity != DVS::Polarity::OFF_ONLY);
    }
    // Off event - correct if not set to ON_ONLY
    else {
        return (m_Polarity != DVS::Polarity::ON_ONLY);
    }
}
//----------------------------------------------------------------------------
bool DVS::isInCrop(const libcaer::events::PolarityEvent &event) const
{
    return ((event.getX() >= m_CropRect->left) && (event.getX() < m_CropRect->right)
            && (event.getY() >= m_CropRect->top) && (event.getY() < m_CropRect->bottom));
}
//----------------------------------------------------------------------------
std::tuple<uint32_t, uint32_t> DVS::scaleEvent(uint32_t x, uint32_t y) const
{
    return std::make_tuple(static_cast<uint32_t>(std::round(x * m_Scale)),
                           static_cast<uint32_t>(std::round(y * m_Scale)));
}
//----------------------------------------------------------------------------
void DVS::setEvent(uint32_t x, uint32_t y, bool polarity, uint32_t *array) const
{
    const size_t address = (polarity ? 1 : 0) + (x * 2) + (y * 2 * m_OutputWidth);
    array[address / 32] |= (1 << (address % 32));
}
//----------------------------------------------------------------------------
void DVS::setEvent(uint32_t x, uint32_t y, uint32_t *array) const
{
    const size_t address = x + (y * m_OutputWidth);
    array[address / 32] |= (1 << (address % 32));
}

}

#pragma once

// GeNN includes
#include "logging.h"

// Forward declarations
namespace plog
{
class IAppender;
}

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
// Shorthand macros for logging to 'SpineML' channel
#define LOGV_SPINEML LOGV_(SpineMLCommon::SpineMLLogging::CHANNEL_SPINEML)
#define LOGD_SPINEML LOGD_(SpineMLCommon::SpineMLLogging::CHANNEL_SPINEML)
#define LOGI_SPINEML LOGI_(SpineMLCommon::SpineMLLogging::CHANNEL_SPINEML)
#define LOGW_SPINEML LOGW_(SpineMLCommon::SpineMLLogging::CHANNEL_SPINEML)
#define LOGE_SPINEML LOGE_(SpineMLCommon::SpineMLLogging::CHANNEL_SPINEML)
#define LOGF_SPINEML LOGF_(SpineMLCommon::SpineMLLogging::CHANNEL_SPINEML)

//----------------------------------------------------------------------------
// SpineMLCommon::SpineMLLogging
//----------------------------------------------------------------------------
namespace SpineMLCommon
{
namespace SpineMLLogging
{
enum Channel
{
    CHANNEL_SPINEML     = Logging::CHANNEL_MAX,
};

void init(plog::Severity level, plog::IAppender *appender);
}   // namespace SpineMLLogging
}   // namespace SpineMLCommon

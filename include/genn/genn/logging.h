#pragma once

// PLOG includes
#include <plog/Severity.h>

// GeNN includes
#include "gennExport.h"

// Forward declarations
namespace plog
{
class IAppender;
}

/*PLOGV << "verbose";
PLOGD << "debug";
PLOGI << "info";
PLOGW << "warning";
PLOGE << "error";
PLOGF << "fatal";
PLOGN */
#define LOGV_CODE_GENERATOR PLOGV_(Logging::CHANNEL_CODE_GENERATOR)
#define LOGD_CODE_GENERATOR PLOGD_(Logging::CHANNEL_CODE_GENERATOR)
#define LOGI_CODE_GENERATOR PLOGI_(Logging::CHANNEL_CODE_GENERATOR)
#define LOGW_CODE_GENERATOR PLOGW_(Logging::CHANNEL_CODE_GENERATOR)
#define LOGE_CODE_GENERATOR PLOGE_(Logging::CHANNEL_CODE_GENERATOR)
#define LOGF_CODE_GENERATOR PLOGF_(Logging::CHANNEL_CODE_GENERATOR)

#define LOGV_BACKEND PLOGV_(Logging::CHANNEL_BACKEND)
#define LOGD_BACKEND PLOGD_(Logging::CHANNEL_BACKEND)
#define LOGI_BACKEND PLOGI_(Logging::CHANNEL_BACKEND)
#define LOGW_BACKEND PLOGW_(Logging::CHANNEL_BACKEND)
#define LOGE_BACKEND PLOGE_(Logging::CHANNEL_BACKEND)
#define LOGF_BACKEND PLOGF_(Logging::CHANNEL_BACKEND)


//----------------------------------------------------------------------------
// Logging
//----------------------------------------------------------------------------
namespace Logging
{
enum Channel
{
    CHANNEL_GENN            = 0,
    CHANNEL_CODE_GENERATOR  = 1,
    CHANNEL_BACKEND         = 2,
    CHANNEL_MAX
};

GENN_EXPORT void init(plog::Severity gennLevel, plog::Severity codeGeneratorLevel, 
                      plog::IAppender *gennAppender, plog::IAppender *codeGeneratorAppender);
}
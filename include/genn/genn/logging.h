#pragma once

// PLOG includes
#include <plog/Log.h>
#include <plog/Severity.h>

// GeNN includes
#include "gennExport.h"

// Forward declarations
namespace plog
{
class IAppender;
}

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
// Shorthand macros for logging to 'GeNN' channel
#define LOGV_GENN LOGV_(Logging::CHANNEL_GENN)
#define LOGD_GENN LOGD_(Logging::CHANNEL_GENN)
#define LOGI_GENN LOGI_(Logging::CHANNEL_GENN)
#define LOGW_GENN LOGW_(Logging::CHANNEL_GENN)
#define LOGE_GENN LOGE_(Logging::CHANNEL_GENN)
#define LOGF_GENN LOGF_(Logging::CHANNEL_GENN)

// Shorthand macros for logging to 'code generator' channel
#define LOGV_CODE_GEN LOGV_(Logging::CHANNEL_CODE_GEN)
#define LOGD_CODE_GEN LOGD_(Logging::CHANNEL_CODE_GEN)
#define LOGI_CODE_GEN LOGI_(Logging::CHANNEL_CODE_GEN)
#define LOGW_CODE_GEN LOGW_(Logging::CHANNEL_CODE_GEN)
#define LOGE_CODE_GEN LOGE_(Logging::CHANNEL_CODE_GEN)
#define LOGF_CODE_GEN LOGF_(Logging::CHANNEL_CODE_GEN)

// Shorthand macros for logging to 'backend' channel
#define LOGV_BACKEND LOGV_(Logging::CHANNEL_BACKEND)
#define LOGD_BACKEND LOGD_(Logging::CHANNEL_BACKEND)
#define LOGI_BACKEND LOGI_(Logging::CHANNEL_BACKEND)
#define LOGW_BACKEND LOGW_(Logging::CHANNEL_BACKEND)
#define LOGE_BACKEND LOGE_(Logging::CHANNEL_BACKEND)
#define LOGF_BACKEND LOGF_(Logging::CHANNEL_BACKEND)


//----------------------------------------------------------------------------
// GeNN::Logging
//----------------------------------------------------------------------------
namespace GeNN::Logging
{
enum Channel
{
    CHANNEL_GENN        = 0,
    CHANNEL_CODE_GEN    = 1,
    CHANNEL_BACKEND     = 2,
    CHANNEL_MAX
};

GENN_EXPORT void init(plog::Severity gennLevel, plog::Severity codeGeneratorLevel, 
                      plog::IAppender *gennAppender, plog::IAppender *codeGeneratorAppender);
}

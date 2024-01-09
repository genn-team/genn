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
#define LOGV_GENN LOGV_(GeNN::Logging::CHANNEL_GENN)
#define LOGD_GENN LOGD_(GeNN::Logging::CHANNEL_GENN)
#define LOGI_GENN LOGI_(GeNN::Logging::CHANNEL_GENN)
#define LOGW_GENN LOGW_(GeNN::Logging::CHANNEL_GENN)
#define LOGE_GENN LOGE_(GeNN::Logging::CHANNEL_GENN)
#define LOGF_GENN LOGF_(GeNN::Logging::CHANNEL_GENN)

// Shorthand macros for logging to 'code generator' channel
#define LOGV_CODE_GEN LOGV_(GeNN::Logging::CHANNEL_CODE_GEN)
#define LOGD_CODE_GEN LOGD_(GeNN::Logging::CHANNEL_CODE_GEN)
#define LOGI_CODE_GEN LOGI_(GeNN::Logging::CHANNEL_CODE_GEN)
#define LOGW_CODE_GEN LOGW_(GeNN::Logging::CHANNEL_CODE_GEN)
#define LOGE_CODE_GEN LOGE_(GeNN::Logging::CHANNEL_CODE_GEN)
#define LOGF_CODE_GEN LOGF_(GeNN::Logging::CHANNEL_CODE_GEN)

// Shorthand macros for logging to 'transpiler' channel
#define LOGV_TRANSPILER LOGV_(GeNN::Logging::CHANNEL_TRANSPILER)
#define LOGD_TRANSPILER LOGD_(GeNN::Logging::CHANNEL_TRANSPILER)
#define LOGI_TRANSPILER LOGI_(GeNN::Logging::CHANNEL_TRANSPILER)
#define LOGW_TRANSPILER LOGW_(GeNN::Logging::CHANNEL_TRANSPILER)
#define LOGE_TRANSPILER LOGE_(GeNN::Logging::CHANNEL_TRANSPILER)
#define LOGF_TRANSPILER LOGF_(GeNN::Logging::CHANNEL_TRANSPILER)

// Shorthand macros for logging to 'backend' channel
#define LOGV_BACKEND LOGV_(GeNN::Logging::CHANNEL_BACKEND)
#define LOGD_BACKEND LOGD_(GeNN::Logging::CHANNEL_BACKEND)
#define LOGI_BACKEND LOGI_(GeNN::Logging::CHANNEL_BACKEND)
#define LOGW_BACKEND LOGW_(GeNN::Logging::CHANNEL_BACKEND)
#define LOGE_BACKEND LOGE_(GeNN::Logging::CHANNEL_BACKEND)
#define LOGF_BACKEND LOGF_(GeNN::Logging::CHANNEL_BACKEND)


//----------------------------------------------------------------------------
// GeNN::Logging
//----------------------------------------------------------------------------
namespace GeNN::Logging
{
enum Channel
{
    CHANNEL_GENN        = 0,
    CHANNEL_CODE_GEN    = 1,
    CHANNEL_TRANSPILER  = 2,
    CHANNEL_BACKEND     = 3,
    CHANNEL_MAX
};

GENN_EXPORT void init(plog::Severity gennLevel, plog::Severity codeGeneratorLevel, plog::Severity transpilerLevel, 
                      plog::IAppender *gennAppender, plog::IAppender *codeGeneratorAppender, plog::IAppender *transpilerAppender);
}

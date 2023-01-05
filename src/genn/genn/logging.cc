#include "logging.h"

//----------------------------------------------------------------------------
// GeNN::Logging
//----------------------------------------------------------------------------
void GeNN::Logging::init(plog::Severity gennLevel, plog::Severity codeGeneratorLevel,
                         plog::IAppender *gennAppender, plog::IAppender *codeGeneratorAppender)
{
    // If there isn't already a plog instance, initialise one
    if(plog::get<CHANNEL_GENN>() == nullptr) {
        plog::init<CHANNEL_GENN>(gennLevel, gennAppender);
    }
    // Otherwise, set it's max severity from GeNN preferences
    else {
        plog::get<CHANNEL_GENN>()->setMaxSeverity(gennLevel);
    }

    // If there isn't already a plog instance, initialise one
    if(plog::get<CHANNEL_CODE_GEN>() == nullptr) {
        plog::init<CHANNEL_CODE_GEN>(codeGeneratorLevel, codeGeneratorAppender);
    }
    // Otherwise, set it's max severity from GeNN preferences
    else {
        plog::get<CHANNEL_CODE_GEN>()->setMaxSeverity(codeGeneratorLevel);
    }
}

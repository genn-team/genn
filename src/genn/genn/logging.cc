#include "logging.h"

//----------------------------------------------------------------------------
// GeNN::Logging
//----------------------------------------------------------------------------
void GeNN::Logging::init(plog::Severity gennLevel, plog::Severity codeGeneratorLevel, plog::Severity transpilerLevel,
                         plog::IAppender *gennAppender, plog::IAppender *codeGeneratorAppender, plog::IAppender *transpilerAppender)
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

    // If there isn't already a plog instance, initialise one
    if(plog::get<CHANNEL_TRANSPILER>() == nullptr) {
        plog::init<CHANNEL_TRANSPILER>(transpilerLevel, transpilerAppender);
    }
    // Otherwise, set it's max severity from GeNN preferences
    else {
        plog::get<CHANNEL_TRANSPILER>()->setMaxSeverity(transpilerLevel);
    }
}

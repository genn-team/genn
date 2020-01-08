#include "spineMLLogging.h"

//----------------------------------------------------------------------------
// Logging
//----------------------------------------------------------------------------
void SpineMLCommon::SpineMLLogging::init(plog::Severity level, plog::IAppender *appender)
{
    // If there isn't already a plog instance, initialise one
    if(plog::get<CHANNEL_SPINEML>() == nullptr) {
        plog::init<CHANNEL_SPINEML>(level, appender);
    }
    // Otherwise, set it's max severity from GeNN preferences
    else {
        plog::get<CHANNEL_SPINEML>()->setMaxSeverity(level);
    }
}

<?xml version="1.0" encoding="ISO-8859-1"?><xsl:stylesheet version="1.0"
xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:SMLLOWNL="http://www.shef.ac.uk/SpineMLLowLevelNetworkLayer" xmlns:SMLNL="http://www.shef.ac.uk/SpineMLNetworkLayer" xmlns:SMLCL="http://www.shef.ac.uk/SpineMLComponentLayer" xmlns:fn="http://www.w3.org/2005/xpath-functions">
<xsl:output method="text" version="1.0" encoding="UTF-8" indent="yes"/>

<!-- TEMPLATE FOR INSERTING THE START OF THE GENN MODEL FILE -->
<xsl:template name="insert_model_file_start_code">
#include "modelSpec.h"
#include "modelSpec.cc"
</xsl:template>

<!-- TEMPLATE FOR INSERTING THE END OF THE GENN MODEL FILE -->
<xsl:template name="insert_model_file_end_code">

</xsl:template>

</xsl:stylesheet>

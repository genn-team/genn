<?xml version="1.0" encoding="ISO-8859-1"?><xsl:stylesheet version="1.0"
xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:SMLLOWNL="http://www.shef.ac.uk/SpineMLLowLevelNetworkLayer" xmlns:SMLNL="http://www.shef.ac.uk/SpineMLNetworkLayer" xmlns:SMLCL="http://www.shef.ac.uk/SpineMLComponentLayer" xmlns:SMLEX="http://www.shef.ac.uk/SpineMLExperimentLayer" xmlns:fn="http://www.w3.org/2005/xpath-functions">
<xsl:output method="text" version="1.0" encoding="UTF-8" indent="yes"/>
<xsl:template match="/">
<!-- OK, THIS IS THE XSLT SCRIPT THAT GENERATES THE GENN utils.h FILE TO ADD COMPONENTS -->
<!-- VERSION = 'NIX -->


<!-- since we start in the experiment file we need to use for-each to get to the model file -->
<xsl:variable name="model_xml" select="//SMLEX:Model/@network_layer_url"/>
<xsl:for-each select="document($model_xml)"> <!-- GET INTO NETWORK FILE -->

<xsl:choose> <!-- CHOOSE SCHEMA -->

<!-- LOW LEVEL SCHEMA -->
<xsl:when test="SMLLOWNL:SpineML">



<!-- EXTRACT NEW NEURON TYPES FROM POPULATIONS -->
<xsl:for-each select="document(/SMLLOWNL:SpineML/SMLLOWNL:Population/SMLLOWNL:Neuron/@url)">
	<xsl:choose>
		<!-- FIRST HANDLE THE EXISTING NEURON TYPES BY RECOGNISING THEM -->
		<xsl:when test="/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativeTraubMiles']">

		</xsl:when>
		<xsl:when test="/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativeIzhikevich']">

		</xsl:when>
		<xsl:when test="/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativeRulkov']">

		</xsl:when>
		<xsl:when test="/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativePoisson']">

		</xsl:when>
		<xsl:otherwise>
  <!-- UNRECOGNISED NEURON TYPE - ADD TO VECTOR -->

		</xsl:otherwise>
		
	</xsl:choose>
</xsl:for-each>

</xsl:when>

<!-- HIGH LEVEL SCHEMA -->
<xsl:when test="SMLNL:SpineML">

<!-- EXTRACT NEW NEURON TYPES FROM POPULATIONS -->
<xsl:for-each select="document(/SMLNL:SpineML/SMLNL:Population/SMLNL:Neuron/@url)">
	<xsl:choose>
		<!-- FIRST HANDLE THE EXISTING NEURON TYPES BY RECOGNISING THEM -->
		<xsl:when test="/SMLCL:ComponentClass[@name = 'GeNNNativeTraubMiles']">

		</xsl:when>
		<xsl:when test="/SMLCL:ComponentClass[@name = 'GeNNNativeIzhikevich']">

		</xsl:when>
		<xsl:when test="/SMLCL:ComponentClass[@name = 'GeNNNativeRulkov']">

		</xsl:when>
		<xsl:when test="/SMLCL:ComponentClass[@name = 'GeNNNativePoisson']">

		</xsl:when>
		<xsl:otherwise>
			<!-- UNRECOGNISED NEURON TYPE - GENERATE GeNN CLASS -->
// Add new neuron type: 
			<xsl:for-each select="//SMLCL:TimeDerivative/SMLCL:MathInLine"> <!-- DIFFERENTIAL EQN -->
				<xsl:call-template name="add_indices">
					<xsl:with-param name="string" select="."/>
					<xsl:with-param name="params" select="//SMLCL:Parameter | //SMLCL:StateVariable"/>
				</xsl:call-template>
			</xsl:for-each> <!-- DIFFERENTIAL EQN -->

		</xsl:otherwise>
		
	</xsl:choose>
</xsl:for-each>
		
</xsl:when>

</xsl:choose> <!-- END CHOOSE SCHEMA -->

</xsl:for-each> <!-- END GET INTO NETWORK FILE -->


</xsl:template>


</xsl:stylesheet>



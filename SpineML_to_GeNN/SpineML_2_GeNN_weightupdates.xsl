<?xml version="1.0" encoding="ISO-8859-1"?><xsl:stylesheet version="1.0"
xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:SMLLOWNL="http://www.shef.ac.uk/SpineMLLowLevelNetworkLayer" xmlns:SMLNL="http://www.shef.ac.uk/SpineMLNetworkLayer" xmlns:SMLCL="http://www.shef.ac.uk/SpineMLComponentLayer" xmlns:SMLEX="http://www.shef.ac.uk/SpineMLExperimentLayer" xmlns:fn="http://www.w3.org/2005/xpath-functions">
<xsl:output method="text" version="1.0" encoding="UTF-8" indent="yes"/>
<xsl:template match="/">
<!-- OK, THIS IS THE XSLT SCRIPT THAT GENERATES THE GENN extra_weightupdates.h FILE TO ADD COMPONENTS -->
<!-- VERSION = 'NIX -->

<!--xsl:call-template name="insert_utils_file_start_code"/-->

<!-- since we start in the experiment file we need to use for-each to get to the model file -->
<xsl:variable name="model_xml" select="//SMLEX:Model/@network_layer_url"/>
<xsl:for-each select="document($model_xml)"> <!-- GET INTO NETWORK FILE -->

<xsl:choose> <!-- CHOOSE SCHEMA -->

<!-- LOW LEVEL SCHEMA -->
<xsl:when test="SMLLOWNL:SpineML">
	weightUpdateModel wu;
<!-- EXTRACT NEW WEIGHTUPDATE TYPES FROM SYNAPSES -->
<xsl:for-each select="/SMLLOWNL:SpineML/SMLLOWNL:Population/SMLLOWNL:Neuron">
	<xsl:variable name="curr_nrn" select="."/>
	<xsl:for-each select="/SMLLOWNL:SpineML//SMLLOWNL:WeightUpdate[not(document(@url)/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativeSynapse' or @name = 'GeNNNativeLearningSynapse']) and ../../@dst_population=$curr_nrn/@name]">
	<xsl:variable name="curr_wu" select="."/>
	<xsl:variable name="wu_pos" select="position()"/>
	<xsl:for-each select="document(@url)">
  <!-- UNRECOGNISED WEIGHTUPDATE TYPE - GENERATE GeNN CLASS -->
  <!-- Sanity - is the weight update type compatible with GeNN?? -->
  <xsl:if test="not(count(//SMLCL:AnalogSendPort)=0)">
  	  <xsl:message terminate="no">
Error: WeightUpdates with AnalogSendPorts are not supported by GeNN
	</xsl:message>
  </xsl:if>
  <!--xsl:if test="not(count(//SMLCL:ImpulseSendPort[@name=$curr_wu/@input_dst_port])=1) and count(//SMLCL:ImpulseSendPort)=1">
  	  <xsl:message terminate="yes">
Error: WeightUpdates must have one ImpulseSendPort
	</xsl:message>
  </xsl:if-->
  <xsl:if test="not(count(//SMLCL:EventReceivePort[@name=$curr_wu/@input_dst_port])=1) and count(//SMLCL:EventReceivePort)=1">
  	  <xsl:message terminate="yes">
Error: WeightUpdates must have one EventReceivePort
	</xsl:message>
  </xsl:if>
  <xsl:if test="count(//SMLCL:TimeDerivative)>0">
  	  <xsl:message terminate="yes">
Error: WeightUpdates cannot contain TimeDerivatives
	</xsl:message>
  </xsl:if>
			
  // Add new weightupdate type - <xsl:value-of select="//SMLCL:ComponentClass/@name"/>: 
  wu.varNames.clear();
  wu.varTypes.clear();
  <xsl:for-each select="//SMLCL:StateVariable | //SMLCL:Parameter">
  	<xsl:variable name="curr_par_name" select="@name"/>
  	<xsl:variable name="curr_par_type" select="local-name(.) = 'Parameter'"/>
  	<xsl:if test="not(count($curr_wu//SMLNL:Property[@name=$curr_par_name]/SMLNL:FixedValue)=1 and ($curr_par_type))">
  wu.varNames.push_back(tS("<xsl:value-of select="@name"/>_WU"));
  wu.varTypes.push_back(tS("float"));<!---->
  	</xsl:if>
  </xsl:for-each>
  <xsl:if test="not(count(//SMLCL:Regime)=1)">
  wu.varNames.push_back(tS("__regime_val"));
  wu.varTypes.push_back(tS("int"));
  </xsl:if>
  wu.pNames.clear();
  <xsl:for-each select="//SMLCL:Parameter">
  	<xsl:variable name="curr_par_name" select="@name"/>
  	<xsl:if test="count($curr_wu//SMLNL:Property[@name=$curr_par_name]/SMLNL:FixedValue)=1">
  wu.pNames.push_back(tS("<xsl:value-of select="@name"/>_WU"));<!---->
  	</xsl:if>
  </xsl:for-each>
  //wu.dpNames.clear();

  wu.simCode = tS(" \
     <!-- LIMIT SCOPE --> { \n \
    <!-- USE TEMP VARIABLE TO GET DATA FROM OTHER COMPONENTS -->
     <xsl:for-each select="//SMLCL:AnalogReceivePort">
     	<xsl:variable name="curr_port_name" select="@name"/>
     	<xsl:if test="$curr_wu/@input_dst_port=$curr_port_name">
     		<!---->	float <xsl:value-of select="@name"/>_WU = $(<xsl:value-of select="$curr_wu/@input_src_port"/>_NB_pre); \n \
     	</xsl:if>
     	<!--xsl:if test="not($curr_wu/SMLLOWNL:Input[@dst_port=$curr_port_name]/@src=$curr_wu/../../@dst_population)">
     		<!- MUCH MORE COMPLICATED - LEAVING FOR NOW! ->
     		<xsl:message terminate="yes">
Error: Connections to WeightUpdates from sources other than the destination Neuron are not supported
			</xsl:message>	
     	</xsl:if-->
     </xsl:for-each>
     <xsl:for-each select="//SMLCL:AnalogReducePort">
     	<xsl:variable name="curr_port_name" select="@name"/>
     	<xsl:if test="$curr_wu/@input_dst_port=$curr_port_name">
     		<!---->	float <xsl:value-of select="@name"/>_WU = $(<xsl:value-of select="$curr_wu/@input_src_port"/>_NB_pre); \n \
     	</xsl:if>
     </xsl:for-each>
     <xsl:for-each select="//SMLCL:OnImpulse"> <!-- ONIMPULSE FOR INPUT FROM SYNAPSE -->
     <!----> float <xsl:value-of select="@src_port"/>_WU = $(inSyn); \
<!---->		<!----><xsl:for-each select="SMLCL:StateAssignment"> <!-- STATEASSIGNMENT -->
				<!---->$(<xsl:value-of select="@variable"/>_WU) = <xsl:call-template name="add_indices">
																	<xsl:with-param name="string" select="SMLCL:MathInline"/>
																	<xsl:with-param name="params" select="//SMLCL:Parameter | //SMLCL:StateVariable | //SMLCL:AnalogReducePort | //SMLCL:ImpulseReceivePort"/>
																</xsl:call-template>; \n \
<!---->		<!----></xsl:for-each> <!-- END STATEASSIGNMENT -->
     </xsl:for-each>
  	 <!-- DO ALIASES  -->
  	 <xsl:for-each select="//SMLCL:Alias[not(@name=//SMLCL:AnalogSendPort/@name)]"> <!-- ALIAS EQN -->
  	 	<!---->float <xsl:value-of select="@name"/>_WU = (<!---->
		<xsl:call-template name="add_indices">
			<xsl:with-param name="string" select="SMLCL:MathInline"/>
			<xsl:with-param name="params" select="//SMLCL:Parameter | //SMLCL:StateVariable | //SMLCL:AnalogReducePort  | //SMLCL:AnalogReceivePort"/>
		</xsl:call-template>); \n \
	 <!---->   	 
  	 </xsl:for-each> <!-- END ALIAS EQN -->
  	 <xsl:for-each select="//SMLCL:Alias[@name=$curr_wu/../SMLLOWNL:PostSynapse/@input_src_port]"> <!-- ALIAS PORT EQN -->
  	 	<!---->$(addtoinSyn) = (<!---->
		<xsl:call-template name="add_indices">
			<xsl:with-param name="string" select="SMLCL:MathInline"/>
			<xsl:with-param name="params" select="//SMLCL:Parameter | //SMLCL:StateVariable | //SMLCL:AnalogReducePort | //SMLCL:AnalogReceivePort"/>
		</xsl:call-template>); \n \
		$(updatelinsyn); \n \
	 <!---->   	 
  	 </xsl:for-each> <!-- END ALIAS PORT EQN -->
  	  <!-- LIMIT SCOPE --> } \n \
  	 <!-- DO REGIME TIME-DERIVATIVE -->
  	 <xsl:for-each select="//SMLCL:Regime"> <!-- REGIMES -->
  	 	<xsl:if test="not(count(//SMLCL:Regime)=1)">
  	 	<!---->if ($(__regime_val)==<xsl:value-of select="position()"/>) { \n \
  	 	</xsl:if>
	 	<!-- TRANSITIONS -->
	 	<xsl:for-each select="SMLCL:OnCondition"> <!-- ONCONDITION -->
	 		<!---->if (<xsl:call-template name="add_indices">
							<xsl:with-param name="string" select="SMLCL:Trigger/SMLCL:MathInline"/>
							<xsl:with-param name="params" select="//SMLCL:Parameter | //SMLCL:StateVariable | //SMLCL:AnalogReducePort | //SMLCL:AnalogReceivePort"/>
						</xsl:call-template>) { \n \
<!---->		<!----><xsl:for-each select="SMLCL:StateAssignment"> <!-- STATEASSIGNMENT -->
				<!---->$(<xsl:value-of select="@variable"/>_WU) = <xsl:call-template name="add_indices">
																	<xsl:with-param name="string" select="SMLCL:MathInline"/>
																	<xsl:with-param name="params" select="//SMLCL:Parameter | //SMLCL:StateVariable | //SMLCL:AnalogReducePort | //SMLCL:AnalogReceivePort"/>
																</xsl:call-template>; \n \
<!---->		<!----></xsl:for-each> <!-- END STATEASSIGNMENT -->
			<!-- REGIME CHANGE -->
			<xsl:if test="not(count(//SMLCL:Regime)=1)">
			<xsl:variable name="targ_regime" select="@target_regime"/>
<!--     -->$(__regime_val) = <xsl:for-each select="//SMLCL:Regime"> <!-- FIND TARGET REGIME -->
							<xsl:if test="@name=$targ_regime">
								<xsl:value-of select="position()"/>
							</xsl:if>
						 </xsl:for-each> <!-- END FIND TARGET REGIME -->
						 <!---->; \n \
			</xsl:if>
<!---->} \n \
<!----> </xsl:for-each> <!-- END ONCONDITION -->
	 	<xsl:for-each select="SMLCL:OnEvent"> <!-- ONEVENT -->
<!---->		<!----><xsl:for-each select="SMLCL:StateAssignment"> <!-- STATEASSIGNMENT -->
				<!---->$(<xsl:value-of select="@variable"/>_WU) = <xsl:call-template name="add_indices">
																	<xsl:with-param name="string" select="SMLCL:MathInline"/>
																	<xsl:with-param name="params" select="//SMLCL:Parameter | //SMLCL:StateVariable | //SMLCL:AnalogReducePort | //SMLCL:AnalogReceivePort"/>
																</xsl:call-template>; \n \
<!---->		<!----></xsl:for-each> <!-- END STATEASSIGNMENT -->
			<!-- EMIT IMPULSE -->
			<xsl:for-each select="SMLCL:ImpulseOut">
				<!-- this is the output to the linSyn variable - we need to assign the specified SV or Parameter onto it! -->
				<!---->	$(addtoinSyn) = $(<xsl:value-of select="@port"/>_WU); \n \
				$(updatelinsyn); \n \
<!---->		</xsl:for-each>
			<!-- END EMIT IMPULSE -->
			<!-- REGIME CHANGE -->
			<xsl:if test="not(count(//SMLCL:Regime)=1)">
			<xsl:variable name="targ_regime" select="@target_regime"/>
<!--     -->$(__regime_val) = <xsl:for-each select="//SMLCL:Regime"> <!-- FIND TARGET REGIME -->
							<xsl:if test="@name=$targ_regime">
								<xsl:value-of select="position()"/>
							</xsl:if>
						 </xsl:for-each> <!-- END FIND TARGET REGIME -->
						 <!---->; \n \
			</xsl:if>
<!----> </xsl:for-each> <!-- END ONEVENT -->
		<xsl:if test="not(count(//SMLCL:Regime)=1)">
<!---->} \n \
		</xsl:if>
<!----></xsl:for-each> <!-- END REGIMES -->
  	<!---->");

  weightUpdateModels.push_back(wu);

	</xsl:for-each>
	</xsl:for-each>
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

<!--xsl:call-template name="insert_utils_file_end_code"/-->

</xsl:template>


<!-- ADD THE $() TO VARIABLES -->
<xsl:template name="add_indices_helper">

	<xsl:param name="start"/>
	<xsl:param name="end"/>	
	<xsl:param name="params"/>
	<xsl:param name="aliases"/>
	<xsl:param name="param"/>
	<xsl:choose>
		<xsl:when test="contains($end,$param/@name)">
		<xsl:variable name="startTemp" select="concat($start,substring-before($end,$param/@name))"/>
		<xsl:variable name="endTemp" select="substring-after($end,$param/@name)"/>
			<xsl:choose>
			<xsl:when test="contains('+-*/() =&lt;&gt;',substring($startTemp,string-length($startTemp),1))">
			<xsl:choose>
			<xsl:when test="contains('+-*/() =&lt;&gt;',substring($endTemp,1,1))">
				<xsl:choose>				
				<xsl:when test="local-name($param)='AnalogReducePort' or local-name($param)='ImpulseReceivePort' or local-name($param)='AnalogReceivePort'">
				<xsl:call-template name="add_indices_helper">
					<xsl:with-param name="params" select="$params"/>
					<xsl:with-param name="param" select="$param"/>
					<xsl:with-param name="start" select="concat($startTemp,$param/@name,'_WU')"/>
					<xsl:with-param name="end" select="$endTemp"/>
				</xsl:call-template>
				</xsl:when>
				<xsl:otherwise>
				<xsl:call-template name="add_indices_helper">
					<xsl:with-param name="params" select="$params"/>
					<xsl:with-param name="param" select="$param"/>
					<xsl:with-param name="start" select="concat($startTemp,'($(',$param/@name,'_WU))')"/>
					<xsl:with-param name="end" select="$endTemp"/>
				</xsl:call-template>
				</xsl:otherwise>
				</xsl:choose>
			</xsl:when>
			<xsl:otherwise>
			<xsl:call-template name="add_indices_helper">
				<xsl:with-param name="params" select="$params"/>
				<xsl:with-param name="param" select="$param"/>
				<xsl:with-param name="start" select="concat($startTemp,$param/@name)"/>
				<xsl:with-param name="end" select="$endTemp"/>
			</xsl:call-template>
			</xsl:otherwise>
			</xsl:choose>
			</xsl:when>
			<xsl:otherwise>
			<xsl:call-template name="add_indices_helper">
				<xsl:with-param name="params" select="$params"/>
				<xsl:with-param name="param" select="$param"/>
				<xsl:with-param name="start" select="concat($startTemp,$param/@name)"/>
				<xsl:with-param name="end" select="$endTemp"/>
			</xsl:call-template>
			</xsl:otherwise>
			</xsl:choose>			
		</xsl:when>
		<xsl:otherwise>
			<xsl:call-template name="add_indices">
				<xsl:with-param name="params" select="$params[position() > 1]"/>
				<xsl:with-param name="string" select="concat($start,$end)"/>
			</xsl:call-template>
		</xsl:otherwise>
	</xsl:choose>

</xsl:template>

<xsl:template name="add_indices">

	<xsl:param name="params"/>
	<xsl:param name="string"/>
	<xsl:choose>
		<xsl:when test="not($params)">
		<xsl:value-of select="$string"/>
		</xsl:when>
		<xsl:otherwise>
		<xsl:variable name="param" select = "$params[1]"/>
			<xsl:choose>
			<xsl:when test="contains($string,$param/@name)">
			<xsl:call-template name="add_indices_helper">
				<xsl:with-param name="params" select="$params"/>
				<xsl:with-param name="param" select="$param"/>
				<xsl:with-param name="start" select="@thisshouldnotexist"/>
				<xsl:with-param name="end" select="$string"/>
			</xsl:call-template>
			</xsl:when>
			<xsl:otherwise>
			<xsl:call-template name="add_indices">
	
			<xsl:with-param name="params" select="$params[position() > 1]"/>
				<xsl:with-param name="string" select="$string"/>
			</xsl:call-template>
			</xsl:otherwise>
			</xsl:choose>
		</xsl:otherwise>
	</xsl:choose>

</xsl:template>	



<xsl:include href="utils_file_code.xsl"/>

</xsl:stylesheet>



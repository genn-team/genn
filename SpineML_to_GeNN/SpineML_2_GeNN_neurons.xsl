<?xml version="1.0" encoding="ISO-8859-1"?><xsl:stylesheet version="1.0"
xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:SMLLOWNL="http://www.shef.ac.uk/SpineMLLowLevelNetworkLayer" xmlns:SMLNL="http://www.shef.ac.uk/SpineMLNetworkLayer" xmlns:SMLCL="http://www.shef.ac.uk/SpineMLComponentLayer" xmlns:SMLEX="http://www.shef.ac.uk/SpineMLExperimentLayer" xmlns:fn="http://www.w3.org/2005/xpath-functions">
<xsl:output method="text" version="1.0" encoding="UTF-8" indent="yes"/>
<xsl:template match="/">
<!-- OK, THIS IS THE XSLT SCRIPT THAT GENERATES THE GENN extra_neurons.h FILE TO ADD COMPONENTS -->
<!-- VERSION = 'NIX -->

<!--xsl:call-template name="insert_utils_file_start_code"/-->

<xsl:variable name="expt_root" select="."/>

<!-- since we start in the experiment file we need to use for-each to get to the model file -->
<xsl:variable name="model_xml" select="//SMLEX:Model/@network_layer_url"/>
<xsl:for-each select="document($model_xml)"> <!-- GET INTO NETWORK FILE -->

<xsl:variable name="model_root" select="."/>

	neuronModel n;

<xsl:choose> <!-- CHOOSE SCHEMA -->

<!-- LOW LEVEL SCHEMA -->
<xsl:when test="SMLLOWNL:SpineML">

<!-- EXTRACT POPULATIONS - WE'LL WRITE A NEW NEURON TYPE FOR EACH POPULATION TO ALLOW GENERIC INPUTS ETC -->
<xsl:for-each select="/SMLLOWNL:SpineML/SMLLOWNL:Population/SMLLOWNL:Neuron">
	<xsl:variable name="curr_nb" select="."/>
	<!-- ENTER THE COMPONENT FILE -->
	<xsl:for-each select="document(@url)">
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
  <!-- UNRECOGNISED NEURON TYPE - GENERATE GeNN CLASS -->
  <!-- Sanity - is the neuron type compatible with GeNN?? -->
  <xsl:if test="not(count(//SMLCL:EventSendPort)=1)">
  	  <xsl:message terminate="no">
Error: Trying to add a neuron with more than one event send port - which really just won't work in GeNN
(basically I'm flagging this because you don't have just one EventSendPort)
	</xsl:message>
  </xsl:if>
			
  // Add new neuron type - <xsl:value-of select="//SMLCL:ComponentClass/@name"/>: 
  n.varNames.clear();
  n.varTypes.clear();
  <!-- ADD A VALUE FOR V - WHICH WILL BE USED TO TRIGGER THE EVENT - NEEDED? -->
  n.varNames.push_back(tS("<xsl:value-of select="'V'"/>"));
  n.varTypes.push_back(tS("float"));<!---->
  <!--xsl:if test="not(//SMLCL:StateVariable[@name='V'])">
  	<xsl:message terminate="no">
Warning: Trying to add a neuron type without a variable 'V'. At the moment this will fail
	</xsl:message>
  </xsl:if>
  <xsl:if test="not(//SMLCL:AnalogReducePort[@name='Isyn'])">
	<xsl:message terminate="yes">
Error: Trying to add a neuron type without an input 'Isyn'. At the moment this will fail
	</xsl:message>
  </xsl:if-->
  <xsl:for-each select="//SMLCL:StateVariable">
  n.varNames.push_back(tS("<xsl:value-of select="concat(@name,'_NB')"/>"));
  n.varTypes.push_back(tS("float"));<!---->
  </xsl:for-each>
  n.varNames.push_back(tS("__regime_val"));
  n.varTypes.push_back(tS("int"));
  n.pNames.clear();
  <xsl:for-each select="//SMLCL:Parameter">
  n.pNames.push_back(tS("<xsl:value-of select="concat(@name,'_NB')"/>"));<!---->
  </xsl:for-each>
  n.dpNames.clear();
  n.resetCode = "";
  n.thresholdConditionCode = "";

  n.simCode = tS(" \
  	 <!-- CREATE UNUSED ANALOG INPUT PORT VARIABLES AND SET TO 0  -->
  	 <xsl:for-each select="//SMLCL:AnalogReducePort">
  	 	<xsl:variable name="curr_ap_name" select="@name"/>
  	 	<xsl:if test="not(count($curr_nb/SMLLOWNL:Input[@dst_port=$curr_ap_name] | $model_root//SMLLOWNL:Projection[@dst_population=$curr_nb/@name]//SMLLOWNL:PostSynapse[@output_dst_port=$curr_ap_name]))">
  	 		<!---->float <xsl:value-of select="@name"/>_NB = 0; \n\
<!---->
			</xsl:if>
  	 </xsl:for-each>
  	 <!-- GENERATE INPUTS -->
  	 <xsl:for-each select="$expt_root//SMLEX:TimeVaryingInput[@target=$curr_nb/@name]">
  	 	<!-- We know the input port is declared, so we can quickly update the values -->
  	 	<xsl:for-each select="SMLEX:TimePointValue">
  	 		<!---->if (t > <xsl:value-of select="@time"/>) { \n \
<!----> <!----><xsl:value-of select="../@port"/>_NB = <xsl:value-of select="@value"/>; \n \
<!----> <!---->} \n \
<!----></xsl:for-each>
  	 </xsl:for-each>
<!---->  	 //hack \n \
  	 float randomNormal = 0; \n \
  	 <!-- DO ALIASES  -->
  	 <xsl:for-each select="//SMLCL:Alias"> <!-- ALIAS EQN -->
  	 	<!---->float <xsl:value-of select="@name"/> = (<!---->
		<xsl:call-template name="add_indices">
			<xsl:with-param name="string" select="SMLCL:MathInline"/>
			<xsl:with-param name="params" select="//SMLCL:Parameter | //SMLCL:StateVariable | //SMLCL:AnalogReducePort"/>
		</xsl:call-template>); \n \
	 <!---->   	 
  	 </xsl:for-each> <!-- END ALIAS EQN -->
  	 <!-- DO REGIME TIME-DERIVATIVE SECOND -->
  	 <xsl:for-each select="//SMLCL:Regime"> <!-- REGIMES -->
  	 	<!---->if ($(__regime_val)==<xsl:value-of select="position()"/>) { \n \
<!----> <xsl:for-each select=".//SMLCL:TimeDerivative"> <!-- DIFFERENTIAL EQN -->
  		 	<!---->$(<xsl:value-of select="@variable"/>_NB) += (<!---->
			<xsl:call-template name="add_indices">
				<xsl:with-param name="string" select="SMLCL:MathInline"/>
				<xsl:with-param name="params" select="//SMLCL:Parameter | //SMLCL:StateVariable | //SMLCL:AnalogReducePort"/>
			</xsl:call-template>)*DT; \n \
	 	<!----> 
	 	</xsl:for-each> <!-- END DIFFERENTIAL EQN -->
	 	<!-- TRANSITIONS -->
	 	<xsl:for-each select="SMLCL:OnCondition"> <!-- ONCONDITION -->
	 		<xsl:if test="count(SMLCL:EventOut)=0">
	 		<!---->if (<xsl:call-template name="add_indices">
							<xsl:with-param name="string" select="SMLCL:Trigger/SMLCL:MathInline"/>
							<xsl:with-param name="params" select="//SMLCL:Parameter | //SMLCL:StateVariable | //SMLCL:AnalogReducePort"/>
						</xsl:call-template>) { \n \
<!---->		<!----><xsl:for-each select="SMLCL:StateAssignment"> <!-- STATEASSIGNMENT -->
				<!---->$(<xsl:value-of select="@variable"/>_NB) = <xsl:call-template name="add_indices">
																	<xsl:with-param name="string" select="SMLCL:MathInline"/>
																	<xsl:with-param name="params" select="//SMLCL:Parameter | //SMLCL:StateVariable | //SMLCL:AnalogReducePort"/>
																</xsl:call-template>; \n \
<!---->		<!----></xsl:for-each> <!-- END STATEASSIGNMENT -->
<!---->		<!----><xsl:for-each select="SMLCL:EventOut"> <!-- EVENTOUT -->
				<!-- SINCE THERE CAN ONLY BE ONE EVENTSENDPORT TRIGGER THE SPIKE -->
<!---->		<!---->		<!--$(V) = 100000;--> \
<!---->		<!----></xsl:for-each> <!-- END EVENTOUT -->
			<!-- REGIME CHANGE -->
			<xsl:variable name="targ_regime" select="@target_regime"/>
<!--     -->$(__regime_val) = <xsl:for-each select="//SMLCL:Regime"> <!-- FIND TARGET REGIME -->
							<xsl:if test="@name=$targ_regime">
								<xsl:value-of select="position()"/>
							</xsl:if>
						 </xsl:for-each> <!-- END FIND TARGET REGIME -->
						 <!---->; \n \
<!---->} \n \
					</xsl:if>
<!----> </xsl:for-each> <!-- END ONCONDITION -->
<!---->} \n \
<!----></xsl:for-each> <!-- END REGIMES -->
  	<xsl:variable name="curr_c" select="/"/>
  	<xsl:if test="count($curr_nb/../SMLLOWNL:Projection//SMLLOWNL:WeightUpdate[@input_src_port=$curr_c//SMLCL:AnalogSendPort/@name]) > 0">
<!---->oldSpike = false; \n \
  	</xsl:if>
  	<!---->");
  	<!-- IF WE ARE A SPIKING NEURON -->
  	<xsl:if test="count(//SMLCL:EventSendPort)>0">
  	n.thresholdConditionCode = tS(" \
  	   <xsl:for-each select="//SMLCL:Regime"> <!-- REGIMES -->
  	   <!-- TRANSITIONS -->
	 		 <xsl:for-each select="SMLCL:OnCondition"> <!-- ONCONDITION -->
	 		 	<xsl:if test="count(SMLCL:EventOut)=1">
					<xsl:call-template name="add_indices">
							<xsl:with-param name="string" select="SMLCL:Trigger/SMLCL:MathInline"/>
							<xsl:with-param name="params" select="//SMLCL:Parameter | //SMLCL:StateVariable | //SMLCL:AnalogReducePort"/>
					</xsl:call-template>
				 </xsl:if>
<!----> </xsl:for-each> <!-- END ONCONDITION -->
<!----></xsl:for-each> <!-- END REGIMES -->
  	<!---->");  
  	n.resetCode = tS(" \
  	  <xsl:for-each select="//SMLCL:Regime"> <!-- REGIMES -->
  	   <!-- TRANSITIONS -->
	 		 <xsl:for-each select="SMLCL:OnCondition"> <!-- ONCONDITION -->
	 		 	<xsl:if test="count(SMLCL:EventOut)=1">
<!---->		<!----><xsl:for-each select="SMLCL:StateAssignment"> <!-- STATEASSIGNMENT -->
				<!---->$(<xsl:value-of select="@variable"/>_NB) = <xsl:call-template name="add_indices">
																	<xsl:with-param name="string" select="SMLCL:MathInline"/>
																	<xsl:with-param name="params" select="//SMLCL:Parameter | //SMLCL:StateVariable | //SMLCL:AnalogReducePort"/>
																</xsl:call-template>; \n \
<!---->		<!----></xsl:for-each> <!-- END STATEASSIGNMENT -->
				 </xsl:if>
<!----> </xsl:for-each> <!-- END ONCONDITION -->
<!----></xsl:for-each> <!-- END REGIMES -->
  	<!---->");  
  	</xsl:if>
  	<!-- IF NO EVENT SEND PORTS WE ARE NOT A SPIKING NEURON, SO TRIGGER 
  			UPDATE EVERY TIME STEP -->	
  	<xsl:if test="count($curr_nb/../SMLLOWNL:Projection//SMLLOWNL:WeightUpdate[@input_src_port=$curr_c//SMLCL:AnalogSendPort/@name]) > 0">
  	n.thresholdConditionCode = "true";
  	n.resetCode = "";
  	</xsl:if>

  nModels.push_back(n);

		</xsl:otherwise>
		
	</xsl:choose>
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
				<xsl:when test="local-name($param)='AnalogReducePort'">
				<xsl:call-template name="add_indices_helper">
					<xsl:with-param name="params" select="$params"/>
					<xsl:with-param name="param" select="$param"/>
					<xsl:with-param name="start" select="concat($startTemp,$param/@name,'_NB')"/>
					<xsl:with-param name="end" select="$endTemp"/>
				</xsl:call-template>
				</xsl:when>
				<xsl:otherwise>
				<xsl:call-template name="add_indices_helper">
					<xsl:with-param name="params" select="$params"/>
					<xsl:with-param name="param" select="$param"/>
					<xsl:with-param name="start" select="concat($startTemp,'($(',$param/@name,'_NB))')"/>
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



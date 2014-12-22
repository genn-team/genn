<?xml version="1.0" encoding="ISO-8859-1"?><xsl:stylesheet version="1.0"
xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:SMLLOWNL="http://www.shef.ac.uk/SpineMLLowLevelNetworkLayer" xmlns:SMLNL="http://www.shef.ac.uk/SpineMLNetworkLayer" xmlns:SMLCL="http://www.shef.ac.uk/SpineMLComponentLayer" xmlns:SMLEX="http://www.shef.ac.uk/SpineMLExperimentLayer" xmlns:fn="http://www.w3.org/2005/xpath-functions">
<xsl:output method="text" version="1.0" encoding="UTF-8" indent="yes"/>
<xsl:template match="/">
<!-- OK, THIS IS THE XSLT SCRIPT THAT GENERATES THE GENN model.cc FILE -->
<!-- VERSION = 'NIX -->

<!-- EXPERIMENT SET-UP -->
<!-- DEFINE DT -->
#define DT <xsl:value-of select="//SMLEX:Simulation//@dt"/>
<xsl:text>
</xsl:text>

<xsl:call-template name="insert_model_file_start_code"/>

<!-- since we start in the experiment file we need to use for-each to get to the model file -->
<xsl:variable name="model_xml" select="//SMLEX:Model/@network_layer_url"/>
<xsl:for-each select="document($model_xml)"> <!-- GET INTO NETWORK FILE -->

<xsl:choose> <!-- CHOOSE SCHEMA -->

<!-- LOW LEVEL SCHEMA -->
<xsl:when test="SMLLOWNL:SpineML">

	<!-- INITIAL SANITY CHECKING -->
	<!-- IF WE HAVE A NATIVE WEIGHT UPDATE WE MUST HAVE A NATIVE POSTSYNAPSE TYPE -->
	<!--xsl:for-each select="//SMLLOWNL:Synapse">
		<xsl:if test="document(SMLLOWNL:WeightUpdate/@url)/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativeSynapse']">
			<xsl:if test="not(document(SMLLOWNL:PostSynapse/@url)/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativePostSynapse'])">
				<xsl:message terminate="yes">
Error: Mismatch of Native WeightUpdate and PostSynapse types
				</xsl:message>
			</xsl:if>
		</xsl:if>
		<xsl:if test="document(SMLLOWNL:WeightUpdate/@url)/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativeGradedSynapse']">
			<xsl:if test="not(document(SMLLOWNL:PostSynapse/@url)/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativeGradedPostSynapse'])">
				<xsl:message terminate="yes">
Error: Mismatch of Native WeightUpdate and PostSynapse types
				</xsl:message>
			</xsl:if>
		</xsl:if>		
		<xsl:if test="document(SMLLOWNL:WeightUpdate/@url)/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativeLearningSynapse']">
			<xsl:if test="not(document(SMLLOWNL:PostSynapse/@url)/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativeLearningPostSynapse'])">
				<xsl:message terminate="yes">
Error: Mismatch of Native WeightUpdate and PostSynapse types
				</xsl:message>
			</xsl:if>
		</xsl:if>	
	</xsl:for-each-->
	<!-- NO GENERIC INPUTS -->
	<!--xsl:if test="count(//SMLLOWNL:Input)>0">
		<xsl:message terminate="no">
Error: Low level API not supported - Generic Inputs are not supported by GeNN
		</xsl:message>
	</xsl:if -->
	<!-- NO GROUPS -->
	<xsl:if test="count(//SMLLOWNL:Group)>0">
		<xsl:message terminate="yes">
Error: Low level API not supported - Groups are not supported by GeNN
		</xsl:message>
	</xsl:if>
	<!-- NOTHING BUT FIXED PROPERTIES (EXCEPT G) -->
	<xsl:if test="//SMLNL:Property[not(@name='g')]/SMLNL:ValueList | //SMLNL:Property[not(@name='g')]/SMLNL:UniformDistribution | //SMLNL:Property[not(@name='g')]/SMLNL:NormalDistribution | //SMLNL:Property[not(@name='g')]/SMLNL:PoissonDistribution">
		<xsl:message terminate="yes">
Error: Non-FixedValue Parameter or StateVariable value found - Only FixedValues are supported by GeNN
		</xsl:message>
	</xsl:if>

	<!-- CREATE POPULATION PARAMETERS AND STATE VARIABLES -->
	<xsl:for-each select="/SMLLOWNL:SpineML/SMLLOWNL:Population"> <!-- FOR-EACH POPULATION -->
		<!-- STORE REFERENCE TO PROPERTIES AND POPULATION NAME -->
		<xsl:variable name="curr_props" select=".//SMLLOWNL:Neuron/SMLNL:Property"/>
		
		<!-- ### PARAMETERS ### -->
		<!-- ADD ARRAY WITH NAME AND SIZE DERIVED FROM COMPONENT -->
		<!---->double p__<xsl:value-of select="translate(SMLLOWNL:Neuron/@name,' -','SH')"/>[<xsl:value-of select="count(document(SMLLOWNL:Neuron/@url)//SMLCL:Parameter)"/>]={
		<xsl:for-each select="document(SMLLOWNL:Neuron/@url)//SMLCL:Parameter"> <!-- ENTER CURRENT POPULATION COMPONENT PARAMETERS -->
			<xsl:variable name="curr_par_name" select="@name"/>
			<xsl:choose>
				<!-- IF THERE EXISTS A FIXED PROPERTY FOR THIS PARAMETER -->
				<xsl:when test="$curr_props[@name=$curr_par_name]/SMLNL:FixedValue">
					<!----><xsl:value-of select="$curr_props[@name=$curr_par_name]/SMLNL:FixedValue/@value"/>, // <xsl:value-of select="position()-1"/> - <xsl:value-of select="$curr_par_name"/>
					<xsl:text>
		</xsl:text>				
				</xsl:when>
				<!-- IF THERE EXISTS A RANDOM PROPERTY FOR THIS PARAMETER -->
				<xsl:when test="$curr_props[@name=$curr_par_name]/SMLNL:UniformDistribution | $curr_props[@name=$curr_par_name]/SMLNL:NormalDistribution | $curr_props[@name=$curr_par_name]/SMLNL:PoissonDistribution">
					<!-- NOT CURRENTLY SUPPORTED--> // <xsl:value-of select="position()-1"/> - <xsl:value-of select="$curr_par_name"/>
					<xsl:text>
		</xsl:text>
					<xsl:message terminate="yes">
Error: Random parameter value for '<xsl:value-of select="$curr_par_name"/>' used in model - this is not currently supported by GeNN
					</xsl:message>
				</xsl:when>
				<!-- IF THERE EXISTS A PROPERTY LIST FOR THIS PARAMETER -->
				<xsl:when test="$curr_props[@name=$curr_par_name]/SMLNL:ValueList">
					<!-- NOT CURRENTLY SUPPORTED --> // <xsl:value-of select="position()-1"/> - <xsl:value-of select="$curr_par_name"/>
					<xsl:text>
		</xsl:text>
					<xsl:message terminate="yes">
Error: Explicit list of parameter values for '<xsl:value-of select="$curr_par_name"/>' used in model - this is not currently supported by GeNN
					</xsl:message>
				</xsl:when>
				<!-- IF NO PROPERTY FOR THIS PARAMETER -->
				<xsl:otherwise>
					<!---->0.0, // <xsl:value-of select="position()-1"/> - <xsl:value-of select="$curr_par_name"/>
					<xsl:text>
		</xsl:text>
				</xsl:otherwise>
			</xsl:choose>
		</xsl:for-each> <!-- END ENTER CURRENT POPULATION COMPONENT PARAMETERS -->
		<!---->};
<!---->		

		<!-- ### STATE VARIABLES ### -->
		<!-- ADD ARRAY WITH NAME AND SIZE DERIVED FROM COMPONENT -->
		<!---->double ini__<xsl:value-of select="translate(SMLLOWNL:Neuron/@name,' -','SH')"/>[<!---->
			<xsl:if test="document(SMLLOWNL:Neuron/@url)/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativeTraubMiles'] | document(SMLLOWNL:Neuron/@url)/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativeIzhikevich'] | document(SMLLOWNL:Neuron/@url)/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativeRulkov'] | document(SMLLOWNL:Neuron/@url)/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativePoisson']">
				<xsl:value-of select="count(document(SMLLOWNL:Neuron/@url)//SMLCL:StateVariable)"/>]={
			</xsl:if>
			<xsl:if test="not(document(SMLLOWNL:Neuron/@url)/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativeTraubMiles'] | document(SMLLOWNL:Neuron/@url)/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativeIzhikevich'] | document(SMLLOWNL:Neuron/@url)/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativeRulkov'] | document(SMLLOWNL:Neuron/@url)/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativePoisson'])">
				<xsl:value-of select="count(document(SMLLOWNL:Neuron/@url)//SMLCL:StateVariable)+2"/>]={
			</xsl:if>
		<!-- ADD V AS A VARIABLE, IF NOT A NATIVE COMPONENT -->
		<xsl:if test="not(document(SMLLOWNL:Neuron/@url)/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativeTraubMiles'] | document(SMLLOWNL:Neuron/@url)/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativeIzhikevich'] | document(SMLLOWNL:Neuron/@url)/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativeRulkov'] | document(SMLLOWNL:Neuron/@url)/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativePoisson'])">
			<xsl:value-of select="0.0"/>, // voltage for triggering spikes
<!---->	</xsl:if>
		<xsl:for-each select="document(SMLLOWNL:Neuron/@url)//SMLCL:StateVariable"> <!-- ENTER CURRENT POPULATION COMPONENT STATE VARIABLES -->
			<xsl:variable name="curr_par_name" select="@name"/>
			<xsl:choose>
				<!-- IF THERE EXISTS A FIXED PROPERTY FOR THIS STATE VARIABLE -->
				<xsl:when test="$curr_props[@name=$curr_par_name]/SMLNL:FixedValue">
					<!----><xsl:value-of select="$curr_props[@name=$curr_par_name]/SMLNL:FixedValue/@value"/>, // <xsl:value-of select="position()-1"/> - <xsl:value-of select="$curr_par_name"/>
					<xsl:text>
		</xsl:text>				
				</xsl:when>
				<!-- IF THERE EXISTS A RANDOM PROPERTY FOR THIS STATE VARIABLE -->
				<xsl:when test="$curr_props[@name=$curr_par_name]/SMLNL:UniformDistribution | $curr_props[@name=$curr_par_name]/SMLNL:NormalDistribution | $curr_props[@name=$curr_par_name]/SMLNL:PoissonDistribution">
					<!-- NOT CURRENTLY SUPPORTED--> // <xsl:value-of select="position()-1"/> - <xsl:value-of select="$curr_par_name"/>
					<xsl:text>
		</xsl:text>
					<xsl:message terminate="yes">
Error: Random state variable value for '<xsl:value-of select="$curr_par_name"/>' used in model - this is not currently supported by GeNN
					</xsl:message>
				</xsl:when>
				<!-- IF THERE EXISTS A PROPERTY LIST FOR THIS STATE VARIABLE -->
				<xsl:when test="$curr_props[@name=$curr_par_name]/SMLNL:ValueList">
					<!-- NOT CURRENTLY SUPPORTED --> // <xsl:value-of select="position()-1"/> - <xsl:value-of select="$curr_par_name"/>
					<xsl:text>
		</xsl:text>
					<xsl:message terminate="yes">
Error: Explicit list of state variable values used for '<xsl:value-of select="$curr_par_name"/>' in model - this is not currently supported by GeNN
					</xsl:message>
				</xsl:when>
				<!-- IF NO PROPERTY FOR THIS STATE VARIABLE -->
				<xsl:otherwise>
					<!---->0.0, // <xsl:value-of select="position()-1"/> - <xsl:value-of select="$curr_par_name"/>
					<xsl:text>
		</xsl:text>
				</xsl:otherwise>
			</xsl:choose>
		</xsl:for-each> <!-- END ENTER CURRENT POPULATION COMPONENT STATE VARIABLES -->
		<!-- ADD THE REGIME AS A VARIABLE, IF NOT A NATIVE COMPONENT -->
		<xsl:if test="not(document(SMLLOWNL:Neuron/@url)/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativeTraubMiles'] | document(SMLLOWNL:Neuron/@url)/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativeIzhikevich'] | document(SMLLOWNL:Neuron/@url)/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativeRulkov'] | document(SMLLOWNL:Neuron/@url)/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativePoisson'])">
			<xsl:for-each select="document(SMLLOWNL:Neuron/@url)//SMLCL:Regime">
				<xsl:if test="@name=//@initial_regime">
					<xsl:value-of select="position()"/>, // initial regime
<!---->			</xsl:if>
			</xsl:for-each>
		</xsl:if>
		<!---->};
<!---->		

	</xsl:for-each> <!-- END FOR-EACH POPULATION -->
	
	<!-- CREATE SYNAPSE PARAMETERS -->
		<xsl:for-each select="/SMLLOWNL:SpineML//SMLLOWNL:Synapse"> <!-- FOR-EACH SYNAPSE -->
		<!-- STORE REFERENCE TO PROPERTIES AND POPULATION NAME -->
		<xsl:variable name="curr_props" select="./SMLLOWNL:WeightUpdate/SMLNL:Property"/>
		
		<!-- ### PARAMETERS FOR WU ### -->
		<!-- ADD ARRAY WITH NAME AND SIZE DERIVED FROM COMPONENT FOR WU -->
		<!---->double p__<xsl:value-of select="concat('WeightUpdate',position())"/>[<!-- EMPTY BRACKETS AS C++ DOES NOT REQUIRE A NUMBER -->]={
		<!-- IMPORTANT - - - - - - IN THE NEXT LINE WE EXCLUDE G FROM THE PARS - THIS WILL NEED UNDOING  -->
		<xsl:for-each select="document(SMLLOWNL:WeightUpdate/@url)//SMLCL:Parameter"> <!-- ENTER CURRENT POPULATION COMPONENT PARAMETERS -->
			<xsl:variable name="curr_par_name" select="@name"/>
			<xsl:choose>
				<!-- IF THERE EXISTS A FIXED PROPERTY FOR THIS PARAMETER -->
				<xsl:when test="$curr_props[@name=$curr_par_name]/SMLNL:FixedValue">
					<!----><xsl:value-of select="$curr_props[@name=$curr_par_name]/SMLNL:FixedValue/@value"/>, // <xsl:value-of select="position()-1"/> - <xsl:value-of select="$curr_par_name"/>
					<xsl:text>
		</xsl:text>				
				</xsl:when>
				<!-- IF THERE EXISTS A RANDOM PROPERTY FOR THIS PARAMETER -->
				<xsl:when test="$curr_props[@name=$curr_par_name]/SMLNL:UniformDistribution | $curr_props[@name=$curr_par_name]/SMLNL:NormalDistribution | $curr_props[@name=$curr_par_name]/SMLNL:PoissonDistribution">
					<!-- NOT CURRENTLY SUPPORTED--> // <xsl:value-of select="position()-1"/> - <xsl:value-of select="$curr_par_name"/>
					<xsl:text>
		</xsl:text>
					<xsl:message terminate="no">
Error: Random parameter value for '<xsl:value-of select="$curr_par_name"/>' used in model - this is not supported by GeNN
					</xsl:message>
				</xsl:when>
				<!-- IF THERE EXISTS A PROPERTY LIST FOR THIS PARAMETER -->
				<xsl:when test="$curr_props[@name=$curr_par_name]/SMLNL:ValueList">
					<!-- NOT CURRENTLY SUPPORTED --> // <xsl:value-of select="position()-1"/> - <xsl:value-of select="$curr_par_name"/>
					<xsl:text>
		</xsl:text>
					<xsl:message terminate="no">
Error: Explicit list of parameter values for '<xsl:value-of select="$curr_par_name"/>' used in model - this is not supported by GeNN
					</xsl:message>
				</xsl:when>
				<!-- IF NO PROPERTY FOR THIS PARAMETER -->
				<xsl:otherwise>
					<!---->0.0, // <xsl:value-of select="position()-1"/> - <xsl:value-of select="$curr_par_name"/>
					<xsl:text>
		</xsl:text>
				</xsl:otherwise>
			</xsl:choose>
		</xsl:for-each> <!-- END ENTER CURRENT POPULATION COMPONENT PARAMETERS -->
		<!---->};
<!---->		

		<!-- ### STATE VARIABLES FOR WU ### -->
		<!-- ADD ARRAY WITH NAME AND SIZE DERIVED FROM COMPONENT -->
		<!---->double ini__<xsl:value-of select="concat('WeightUpdate',position())"/>[<!-- EMPTY BRACKETS AS C++ DOES NOT REQUIRE A NUMBER -->]={
		<xsl:for-each select="document(SMLLOWNL:WeightUpdate/@url)//SMLCL:StateVariable | document(SMLLOWNL:WeightUpdate/@url)//SMLCL:Parameter"> <!-- ENTER CURRENT POPULATION COMPONENT STATE VARIABLE -->
				<xsl:message terminate="no">
Warning: State variable in Synapses are not currently supported by GeNN
				</xsl:message>		
			<xsl:variable name="curr_par_name" select="@name"/>
			<xsl:variable name="curr_par_type" select="local-name(.) = 'Parameter'"/>
			<xsl:choose>
				<!-- IF THERE EXISTS A FIXED PROPERTY FOR THIS STATE VARIABLES -->
				<xsl:when test="count($curr_props[@name=$curr_par_name]/SMLNL:FixedValue)=1 and not($curr_par_type)">
				<!----><xsl:value-of select="$curr_props[@name=$curr_par_name]/SMLNL:FixedValue/@value"/>, // <xsl:value-of select="position()-1"/> - <xsl:value-of select="$curr_par_name"/>
					<xsl:text>
		</xsl:text>				
				</xsl:when>
				<!-- IF THERE EXISTS A RANDOM PROPERTY FOR THIS STATE VARIABLES -->
				<xsl:when test="$curr_props[@name=$curr_par_name]/SMLNL:UniformDistribution | $curr_props[@name=$curr_par_name]/SMLNL:NormalDistribution | $curr_props[@name=$curr_par_name]/SMLNL:PoissonDistribution">
					<!-- NOT CURRENTLY SUPPORTED--> // <xsl:value-of select="position()-1"/> - <xsl:value-of select="$curr_par_name"/>
					<xsl:text>
		</xsl:text>
					<xsl:message terminate="no">
Error: Random state variable value for '<xsl:value-of select="$curr_par_name"/>' used in model - this is not currently supported by GeNN
					</xsl:message>
				</xsl:when>
				<!-- IF THERE EXISTS A PROPERTY LIST FOR THIS STATE VARIABLES -->
				<xsl:when test="$curr_props[@name=$curr_par_name]/SMLNL:ValueList">
					<!-- NOT CURRENTLY SUPPORTED --> // <xsl:value-of select="position()-1"/> - <xsl:value-of select="$curr_par_name"/>
					<xsl:text>
		</xsl:text>
					<xsl:message terminate="no">
Error: Explicit list of state variable values used for '<xsl:value-of select="$curr_par_name"/>' in model - this is not currently supported by GeNN
					</xsl:message>
				</xsl:when>
				<!-- IF NO PROPERTY FOR THIS STATE VARIABLES -->
				<xsl:when test="not($curr_par_type)">
					<!---->0.0, // <xsl:value-of select="position()-1"/> - <xsl:value-of select="$curr_par_name"/>
					<xsl:text>
		</xsl:text>
				</xsl:when>
				<xsl:otherwise>
				</xsl:otherwise>
			</xsl:choose>
		</xsl:for-each> <!-- END ENTER CURRENT POPULATION COMPONENT STATE VARIABLES -->
		<!---->};
<!---->		

		<!-- STORE REFERENCE TO PROPERTIES AND POPULATION NAME -->
		<xsl:variable name="curr_propsPS" select="./SMLLOWNL:PostSynapse/SMLNL:Property"/>

		<!-- ### PARAMETERS FOR POSTSYNAPSE ### -->
		<!-- ADD ARRAY WITH NAME AND SIZE DERIVED FROM COMPONENT FOR POSTSYNAPSE -->
		<!---->double p__<xsl:value-of select="concat('PostSynapse',position())"/>[<xsl:value-of select="count(document(./SMLLOWNL:PostSynapse/@url)//SMLCL:Parameter)"/>]={
		<!-- IMPORTANT - - - - - - IN THE NEXT LINE WE EXCLUDE G FROM THE PARS - THIS WILL NEED UNDOING  -->
		<xsl:for-each select="document(SMLLOWNL:PostSynapse/@url)//SMLCL:Parameter"> <!-- ENTER CURRENT POPULATION COMPONENT PARAMETERS -->
			<xsl:variable name="curr_par_name" select="@name"/>
			<xsl:choose>
				<!-- IF THERE EXISTS A FIXED PROPERTY FOR THIS PARAMETER -->
				<xsl:when test="$curr_propsPS[@name=$curr_par_name]/SMLNL:FixedValue">
					<!----><xsl:value-of select="$curr_propsPS[@name=$curr_par_name]/SMLNL:FixedValue/@value"/>, // <xsl:value-of select="position()-1"/> - <xsl:value-of select="$curr_par_name"/>
					<xsl:text>
		</xsl:text>				
				</xsl:when>
				<!-- IF THERE EXISTS A RANDOM PROPERTY FOR THIS PARAMETER -->
				<xsl:when test="$curr_propsPS[@name=$curr_par_name]/SMLNL:UniformDistribution | $curr_propsPS[@name=$curr_par_name]/SMLNL:NormalDistribution | $curr_propsPS[@name=$curr_par_name]/SMLNL:PoissonDistribution">
					<!-- NOT CURRENTLY SUPPORTED--> // <xsl:value-of select="position()-1"/> - <xsl:value-of select="$curr_par_name"/>
					<xsl:text>
		</xsl:text>
					<xsl:message terminate="yes">
Error: Random parameter value for '<xsl:value-of select="$curr_par_name"/>' used in model - this is not currently supported by GeNN
					</xsl:message>
				</xsl:when>
				<!-- IF THERE EXISTS A PROPERTY LIST FOR THIS PARAMETER -->
				<xsl:when test="$curr_propsPS[@name=$curr_par_name]/SMLNL:ValueList">
					<!-- NOT CURRENTLY SUPPORTED --> // <xsl:value-of select="position()-1"/> - <xsl:value-of select="$curr_par_name"/>
					<xsl:text>
		</xsl:text>
					<xsl:message terminate="yes">
Error: Explicit list of parameter values for '<xsl:value-of select="$curr_par_name"/>' used in model - this is not currently supported by GeNN
					</xsl:message>
				</xsl:when>
				<!-- IF NO PROPERTY FOR THIS PARAMETER -->
				<xsl:otherwise>
					<!---->0.0, // <xsl:value-of select="position()-1"/> - <xsl:value-of select="$curr_par_name"/>
					<xsl:text>
		</xsl:text>
				</xsl:otherwise>
			</xsl:choose>
		</xsl:for-each> <!-- END ENTER CURRENT POPULATION COMPONENT PARAMETERS -->
		<!---->};
<!---->		

		<!-- ### STATE VARIABLES FOR POSTSYNAPSE ### -->
		<!-- ADD ARRAY WITH NAME AND SIZE DERIVED FROM COMPONENT -->
		<!---->double ini__<xsl:value-of select="concat('PostSynapse',position())"/>[<xsl:value-of select="count(document(./SMLLOWNL:PostSynapse/@url)//SMLCL:StateVariable)"/>]={
		<xsl:for-each select="document(SMLLOWNL:PostSynapse/@url)//SMLCL:StateVariable"> <!-- ENTER CURRENT POPULATION COMPONENT STATE VARIABLE -->
				<xsl:message terminate="no">
Warning: State variable in Synapses are not currently supported by GeNN
				</xsl:message>		
			<xsl:variable name="curr_par_name" select="@name"/>
			<xsl:choose>
				<!-- IF THERE EXISTS A FIXED PROPERTY FOR THIS STATE VARIABLES -->
				<xsl:when test="$curr_propsPS[@name=$curr_par_name]/SMLNL:FixedValue">
					<!----><xsl:value-of select="$curr_propsPS[@name=$curr_par_name]/SMLNL:FixedValue/@value"/>, // <xsl:value-of select="position()-1"/> - <xsl:value-of select="$curr_par_name"/>
					<xsl:text>
		</xsl:text>				
				</xsl:when>
				<!-- IF THERE EXISTS A RANDOM PROPERTY FOR THIS STATE VARIABLES -->
				<xsl:when test="$curr_propsPS[@name=$curr_par_name]/SMLNL:UniformDistribution | $curr_propsPS[@name=$curr_par_name]/SMLNL:NormalDistribution | $curr_propsPS[@name=$curr_par_name]/SMLNL:PoissonDistribution">
					<!-- NOT CURRENTLY SUPPORTED--> // <xsl:value-of select="position()-1"/> - <xsl:value-of select="$curr_par_name"/>
					<xsl:text>
		</xsl:text>
					<xsl:message terminate="yes">
Error: Random state variable value for '<xsl:value-of select="$curr_par_name"/>' used in model - this is not currently supported by GeNN
					</xsl:message>
				</xsl:when>
				<!-- IF THERE EXISTS A PROPERTY LIST FOR THIS STATE VARIABLES -->
				<xsl:when test="$curr_propsPS[@name=$curr_par_name]/SMLNL:ValueList">
					<!-- NOT CURRENTLY SUPPORTED --> // <xsl:value-of select="position()-1"/> - <xsl:value-of select="$curr_par_name"/>
					<xsl:text>
		</xsl:text>
					<xsl:message terminate="yes">
Error: Explicit list of state variable values used for '<xsl:value-of select="$curr_par_name"/>' in model - this is not currently supported by GeNN
					</xsl:message>
				</xsl:when>
				<!-- IF NO PROPERTY FOR THIS STATE VARIABLES -->
				<xsl:otherwise>
					<!---->0.0, // <xsl:value-of select="position()-1"/> - <xsl:value-of select="$curr_par_name"/>
					<xsl:text>
		</xsl:text>
				</xsl:otherwise>
			</xsl:choose>
		</xsl:for-each> <!-- END ENTER CURRENT POPULATION COMPONENT STATE VARIABLES -->
		<!---->};
<!---->		

	</xsl:for-each> <!-- END FOR-EACH SYNAPSE -->
	
	
	<!-- WRITE OUT THE MODEL SETUP FUNCTION -->
	<!---->void modelDefinition(NNmodel &amp;model) 
<!---->{
	GeNNReady = 1;	
	#include "extra_neurons.h"
	#include "extra_postsynapses.h"
	#include "extra_weightupdates.h"
	POISSONNEURON = INT_MAX;
		
<!---->	model.setGPUDevice(0); 
<!---->	model.setName("<xsl:value-of select="translate(/SMLLOWNL:SpineML/@name,' ','_')"/>");<!---->
	<!-- ADD POPULATIONS -->
	<xsl:for-each select="/SMLLOWNL:SpineML/SMLLOWNL:Population"> <!-- FOR-EACH POPULATION -->
	model.addNeuronPopulation("<xsl:value-of select="translate(SMLLOWNL:Neuron/@name,' -','SH')"/>", <!---->
		<xsl:value-of select="SMLLOWNL:Neuron/@size"/>, <!---->
		<!----><xsl:call-template name="get_neuron_type"/>, <!---->
		<!---->p__<xsl:value-of select="translate(SMLLOWNL:Neuron/@name,' -','SH')"/>, <!---->
		<!---->ini__<xsl:value-of select="translate(SMLLOWNL:Neuron/@name,' -','SH')"/>);<!---->
		
	</xsl:for-each> <!-- END FOR-EACH POPULATION -->
	
	<!-- ADD SYNAPSES -->
	<xsl:for-each select="/SMLLOWNL:SpineML//SMLLOWNL:Synapse"> <!-- FOR-EACH SYNAPSE -->
	model.addSynapsePopulation("<xsl:value-of select="concat('Synapse',position())"/>_<xsl:value-of select="translate(../../SMLLOWNL:Neuron/@name,' -','SH')"/>_to_<xsl:value-of select="translate(../@dst_population,' -','SH')"/>",<!---->
		<xsl:call-template name="get_synapse_type"/>, <!---->
		<xsl:choose>
			<xsl:when test="SMLNL:OneToOneConnection">
				<!---->SPARSE, <!-- HANDLED MANUALLY -->
			</xsl:when>
			<xsl:when test="SMLNL:AllToAllConnection">
				<!---->ALLTOALL, <!-- HANDLED NATIVELY -->
			</xsl:when>
			<xsl:otherwise>
				<!-- CALCULATE THE MAX POSSIBLE CONNECTION SIZE -->
				<xsl:variable name="dstPop" select="../@dst_population"/>
				<xsl:variable name="maxConnSize" select="number(//SMLLOWNL:Neuron[@name=$dstPop]/@size)*number(../../SMLLOWNL:Neuron/@size)"/>
				<xsl:choose>
					<xsl:when test="number(SMLNL:FixedProbabilityConnection/@probability)>number(0.1) or (count(.//SMLNL:Connection) div number($maxConnSize))>number(0.1) or (number(.//@num_connections) div number($maxConnSize))>number(0.1) ">
						<!---->DENSE, <!-- USE FULL CONNECTION MATRIX - SHOULD CHECK AND USE SPARSE IF LESS DENSE CONNS -->
					</xsl:when>
					<xsl:otherwise>
						<!---->SPARSE, <!-- USE FULL CONNECTION MATRIX - SHOULD CHECK AND USE SPARSE IF LESS DENSE CONNS -->
					</xsl:otherwise>
				</xsl:choose>
			</xsl:otherwise>
		</xsl:choose>
		<!-- FOR NOW WE'LL DETECT G AS A PARAMETER SEPERATE FROM THE OTHERS... THIS IS NOT NEEDED ANYMORE -->
		<!--xsl:if test="not(SMLLOWNL:WeightUpdate/SMLNL:Property[@name='g'])">
			<xsl:message terminate="yes">
Error: A WeightUpdate component is lacking a value 'g', which is required for GeNN currently... 
			</xsl:message>
		</xsl:if-->
		<!-- Since custom weight updates have been introduced we now always use GLOBAL G as we handle our own G values -->
		<!---->INDIVIDUALG, <!---->
		<!-- NOW HANDLE THE GLOBAL DELAY - FOR NOW WE'LL HARD CODE IT BUT SHOULD DETECT AND FILL THIS IN -->
		<!---->NO_DELAY, <!---->
		<!-- POSTSYNAPSE TYPE -->
		<xsl:call-template name="get_postsynapse_type"/>, <!---->
		<!-- SOURCE AND DESTINATION POPULATIONS-->
		<!---->"<xsl:value-of select="translate(../../SMLLOWNL:Neuron/@name,' -','SH')"/>", <!---->
		<!---->"<xsl:value-of select="translate(../@dst_population,' -','SH')"/>", <!---->
		<!-- VARIABLES FOR WEIGHT UPDATE -->
		<!---->ini__<xsl:value-of select="concat('WeightUpdate',position())"/>, <!---->
		<!-- PARAMETERS FOR WEIGHT UPDATE -->
		<!---->p__<xsl:value-of select="concat('WeightUpdate',position())"/>, <!---->
		<!-- VARIABLES FOR POSTSYNAPSE -->
		<!---->ini__<xsl:value-of select="concat('PostSynapse',position())"/>, <!---->
		<!-- PARAMETERS FOR POSTSYNAPSE -->
		<!---->p__<xsl:value-of select="concat('PostSynapse',position())"/>
		<!---->);<!---->
		<!-- ADD GLOBAL G VALUES -->
		<xsl:if test="SMLLOWNL:WeightUpdate/SMLNL:Property[@name='g']/SMLNL:FixedValue">
	//model.setSynapseG("<xsl:value-of select="concat('Synapse',position())"/>_<xsl:value-of select="translate(../../SMLLOWNL:Neuron/@name,' -','SH')"/>_to_<xsl:value-of select="translate(../@dst_population,' -','SH')"/>",<!---->
<!---->	<xsl:value-of select="SMLLOWNL:WeightUpdate/SMLNL:Property[@name='g']/SMLNL:FixedValue/@value"/>);<!---->
		</xsl:if>
		<!-- We always have a global 'g', just sometimes we do not use it -->
		<xsl:if test="not(SMLLOWNL:WeightUpdate/SMLNL:Property[@name='g']/SMLNL:FixedValue)">
	//model.setSynapseG("<xsl:value-of select="concat('Synapse',position())"/>_<xsl:value-of select="translate(../../SMLLOWNL:Neuron/@name,' -','SH')"/>_to_<xsl:value-of select="translate(../@dst_population,' -','SH')"/>",<!---->
<!---->	0);<!---->
		</xsl:if>
		<!-- For sparse connectivity we need to set the maximum number of connections... -->
		<xsl:variable name="dstPop" select="../@dst_population"/>
		<xsl:variable name="maxConnSize" select="number(//SMLLOWNL:Neuron[@name=$dstPop]/@size)*number(../../SMLLOWNL:Neuron/@size)"/>
		<xsl:if test="count(SMLNL:AllToAllConnection | SMLNL:OneToOneConnectivity) = 0">
			<xsl:if test="not(number(SMLNL:FixedProbabilityConnection/@probability)>number(0.1) or (count(.//SMLNL:Connection) div number($maxConnSize))>number(0.1) or (number(.//@num_connections) div number($maxConnSize))>number(0.1))">
		model.setMaxConn("<xsl:value-of select="concat('Synapse',position())"/>_<xsl:value-of select="translate(../../SMLLOWNL:Neuron/@name,' -','SH')"/>_to_<xsl:value-of select="translate(../@dst_population,' -','SH')"/>", <!---->
		<xsl:if test="SMLNL:OneToOneConnection">
			<xsl:value-of select="../../SMLLOWNL:Neuron/@size"/><!-- size of src nrn -->
		</xsl:if>
		<xsl:if test="SMLNL:FixedProbabilityConnection">
			<xsl:value-of select="number(.//@probability) * number($maxConnSize) * number(2.5)"/><!-- 2.5 is pretty safe I've found -->
		</xsl:if>
		<xsl:if test="count(.//SMLNL:Connection) > 0">
			<xsl:value-of select="count(.//SMLNL:Connection)"/>
		</xsl:if>
		<xsl:if test="count(.//@num_connections) > 0">
			<xsl:value-of select=".//@num_connections"/>
		</xsl:if>
		<!----> );
			</xsl:if>
		</xsl:if>
	</xsl:for-each> <!-- END FOR-EACH SYNAPSE -->
<!---->
<!---->
	model.setPrecision(0);
}
	
</xsl:when>

<!-- HIGH LEVEL SCHEMA -->
<xsl:when test="SMLNL:SpineML">

	<!-- IGNORING FOR NOW SINCE GUI ALWAYS WRITES LOW LEVEL SCHEMA -->
	<xsl:message terminate="yes">
Error: High level schema support not currently implemented for GeNN
	</xsl:message>
		
</xsl:when>

</xsl:choose> <!-- END CHOOSE SCHEMA -->

</xsl:for-each> <!-- END GET INTO NETWORK FILE -->

<xsl:call-template name="insert_model_file_end_code"/>

</xsl:template>

<!-- TEMPLATE TO GET THE NUMBER OR NAME OF THE NEURON TYPE WE HAVE -->
<xsl:template name="get_neuron_type">
	<!--- NOTE - WE SHOULD BE IN A POPULATION TAG -->
	<xsl:variable name="network_file" select="/"/>
	<xsl:variable name="curr_nrn" select="SMLLOWNL:Neuron"/>
	<!-- EXTRACT NEURON TYPES FROM CURRENT POPULATION -->
	<xsl:for-each select="document(SMLLOWNL:Neuron/@url)">
		<xsl:choose>
			<!-- FIRST HANDLE THE EXISTING NEURON TYPES BY RECOGNISING THEM -->
			<xsl:when test="/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativeTraubMiles']">
				<!---->TRAUBMILES<!---->
			</xsl:when>
			<xsl:when test="/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativeIzhikevich']">
				<!---->IZHIKEVICH<!---->
			</xsl:when>
			<xsl:when test="/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativeRulkov']">
				<!---->MAPNEURON<!---->
			</xsl:when>
			<xsl:when test="/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativePoisson']">
				<!---->POISSONNEURON<!---->
			</xsl:when>
			<xsl:otherwise>
				<!-- UNRECOGNISED NEURON TYPE - FIND INDEX -->
				<!-- GET BACK INTO NETWORK FILE FOR THIS -->
				<xsl:for-each select="$network_file">
					<xsl:call-template name="get_neuron_type_number">
						<xsl:with-param name="curr_component" select="$curr_nrn"/>
					</xsl:call-template>
				</xsl:for-each>
			</xsl:otherwise>
		
		</xsl:choose>
	</xsl:for-each>
</xsl:template>

<xsl:template name="get_neuron_type_number">
	<xsl:param name="number" select="number(-1)"/> <!-- START ON -1 -->
	<xsl:param name="curr_component"/>
	<xsl:for-each select="//SMLLOWNL:Neuron[not(document(@url)//SMLCL:ComponentClass/@name='GeNNNativeTraubMiles' or document(@url)//SMLCL:ComponentClass/@name='GeNNNativeIzhikevich' or document(@url)//SMLCL:ComponentClass/@name='GeNNNativeRulkov' or document(@url)//SMLCL:ComponentClass/@name='GeNNNativePoisson')]">
		<xsl:if test="generate-id(.) = generate-id($curr_component)">
			<!---->0+<xsl:value-of select="position()+$number"/>
		</xsl:if>
	</xsl:for-each>
</xsl:template>

<!-- TEMPLATE TO GET THE NUMBER OR NAME OF THE POSTSYNAPSE TYPE WE HAVE -->
<xsl:template name="get_postsynapse_type">
	<!--- NOTE - WE SHOULD BE IN A SYNAPSE TAG -->
	<xsl:variable name="network_file" select="/"/>
	<xsl:variable name="curr_ps" select="SMLLOWNL:PostSynapse"/>
	<!-- EXTRACT NEURON TYPES FROM CURRENT POSTSYNAPSE -->
	<xsl:for-each select="document(SMLLOWNL:PostSynapse/@url)">
		<xsl:choose>
			<!-- FIRST HANDLE THE EXISTING POSTSYNAPSE TYPES BY RECOGNISING THEM -->
			<xsl:when test="/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativePostSynapse']">
				<!---->EXPDECAY<!---->
			</xsl:when>
			<xsl:otherwise>
				<!-- UNRECOGNISED POSTSYNAPSE TYPE - FIND INDEX -->
				<xsl:variable name="curr_component" select="/SMLCL:SpineML/SMLCL:ComponentClass/@name"/>
				<!-- GET BACK INTO NETWORK FILE FOR THIS -->
				<xsl:for-each select="$network_file">
					<xsl:call-template name="get_postsynapse_type_number">
						<xsl:with-param name="curr_component" select="$curr_ps"/>
						<xsl:with-param name="neurons" select="//SMLLOWNL:Neuron"/>
					</xsl:call-template>
				</xsl:for-each>
			</xsl:otherwise>
		
		</xsl:choose>
	</xsl:for-each>
</xsl:template>

<xsl:template name="get_postsynapse_type_number">
	<xsl:param name="number" select="number(-1)"/> <!-- START ON -1 -->
	<xsl:param name="curr_component"/>
	<xsl:param name="neurons"/>
	<xsl:variable name="curr_nrn" select="$neurons[1]"/>
	<xsl:for-each select="//SMLLOWNL:PostSynapse[../../@dst_population=$curr_nrn/@name and not(document(@url)//SMLCL:ComponentClass/@name='GeNNNativePostSynapse')]">
		<xsl:if test="generate-id(.) = generate-id($curr_component)">
			<!---->0+<xsl:value-of select="position()+$number"/>
		</xsl:if>			
	</xsl:for-each>
	<xsl:if test="not(count($neurons)=0)">
		<xsl:call-template name="get_postsynapse_type_number">
			<xsl:with-param name="number" select="$number+count(//SMLLOWNL:PostSynapse[../../@dst_population=$curr_nrn/@name and not(document(@url)//SMLCL:ComponentClass/@name='GeNNNativePostSynapse')])"/>
			<xsl:with-param name="neurons" select="$neurons[position()>1]"/>
			<xsl:with-param name="curr_component" select="$curr_component"/>
		</xsl:call-template>
	</xsl:if>
</xsl:template>

<!-- TEMPLATE TO GET THE NUMBER OR NAME OF THE SYNAPSE TYPE WE HAVE -->
<xsl:template name="get_synapse_type">
	<!--- NOTE - WE SHOULD BE IN A SYNAPSE TAG -->
	<xsl:variable name="network_file" select="/"/>
	<xsl:variable name="curr_wu" select="SMLLOWNL:WeightUpdate"/>
	<!-- EXTRACT SYNAPSE TYPES FROM CURRENT POPULATION -->
	<xsl:for-each select="document(SMLLOWNL:WeightUpdate/@url)">
		<xsl:choose>
			<!-- FIRST HANDLE THE EXISTING NEURON TYPES BY RECOGNISING THEM -->
			<xsl:when test="/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativeSynapse']">
				<!---->NSYNAPSE<!---->
			</xsl:when>
			<xsl:when test="/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativeGradedSynapse']">
				<!---->NGRADSYNAPSE<!---->
			</xsl:when>
			<xsl:when test="/SMLCL:SpineML/SMLCL:ComponentClass[@name = 'GeNNNativeLearningSynapse']">
				<!---->LEARN1SYNAPSE<!---->
			</xsl:when>
			<xsl:otherwise>
				<!-- UNRECOGNISED SYNAPSE TYPE - FIND INDEX -->
				<xsl:variable name="curr_component" select="/SMLCL:SpineML/SMLCL:ComponentClass/@name"/>
				<!-- GET BACK INTO NETWORK FILE FOR THIS -->
				<xsl:for-each select="$network_file">
					<xsl:call-template name="get_synapse_type_number">
						<xsl:with-param name="curr_component" select="$curr_wu"/>
						<xsl:with-param name="neurons" select="//SMLLOWNL:Neuron"/>
					</xsl:call-template>
				</xsl:for-each>
			</xsl:otherwise>
		
		</xsl:choose>
	</xsl:for-each>
</xsl:template>

<xsl:template name="get_synapse_type_number">
	<xsl:param name="number" select="number(-1)"/> <!-- START ON -1 -->
	<xsl:param name="curr_component"/>
	<xsl:param name="neurons"/>
	<xsl:variable name="curr_nrn" select="$neurons[1]"/>
	<xsl:for-each select="//SMLLOWNL:WeightUpdate[../../@dst_population=$curr_nrn/@name and not(document(@url)//SMLCL:ComponentClass/@name='GeNNNativeGradedSynapse' or document(@url)//SMLCL:ComponentClass/@name='GeNNNativeLearningSynapse' or document(@url)//SMLCL:ComponentClass/@name='GeNNNativeSynapse')]">
		<xsl:if test="generate-id(.) = generate-id($curr_component)">
			<!---->0+<xsl:value-of select="position()+$number"/>
		</xsl:if>			
	</xsl:for-each>
	<xsl:if test="not(count($neurons)=0)">
		<xsl:call-template name="get_synapse_type_number">
			<xsl:with-param name="number" select="$number+count(//SMLLOWNL:WeightUpdate[../../@dst_population=$curr_nrn/@name and not(document(@url)//SMLCL:ComponentClass/@name='GeNNNativeGradedSynapse' or document(@url)//SMLCL:ComponentClass/@name='GeNNNativeLearningSynapse' or document(@url)//SMLCL:ComponentClass/@name='GeNNNativeSynapse')])"/>
			<xsl:with-param name="neurons" select="$neurons[position()>1]"/>
			<xsl:with-param name="curr_component" select="$curr_component"/>
		</xsl:call-template>
	</xsl:if>
</xsl:template>

<xsl:include href="model_file_code.xsl"/>

</xsl:stylesheet>



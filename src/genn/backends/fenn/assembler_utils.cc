#include "assembler/assembler_utils.h"

//----------------------------------------------------------------------------
// AssemblerUtils
//----------------------------------------------------------------------------
namespace AssemblerUtils
{
void generateScalarVectorMemcpy(CodeGenerator &c, VectorRegisterAllocator &vectorRegisterAllocator,
                                ScalarRegisterAllocator &scalarRegisterAllocator,
                                std::variant<uint32_t, ScalarRegisterAllocator::RegisterPtr> scalarPtr, 
                                std::variant<uint32_t, ScalarRegisterAllocator::RegisterPtr> vectorPtr, 
                                std::variant<uint32_t, ScalarRegisterAllocator::RegisterPtr> numVectors)
{
    // Register allocation
    ALLOCATE_SCALAR(SVectorBufferEnd);
    ALLOCATE_SCALAR(SOne);

    // Labels
    Label vectorLoop;

    // If literal is provided for scalar pointer, allocate register and load immediate into it
    ScalarRegisterAllocator::RegisterPtr SDataBuffer;
    if(std::holds_alternative<uint32_t>(scalarPtr)) {
        SDataBuffer = scalarRegisterAllocator.getRegister("SDataBuffer = X");
        c.li(*SDataBuffer, std::get<uint32_t>(scalarPtr));
    }
    // Otherwise, use pointer register directly
    else {
        SDataBuffer = std::get<ScalarRegisterAllocator::RegisterPtr>(scalarPtr);
    }

    // If literal is provided for scalar pointer, allocate register and load immediate into it
    ScalarRegisterAllocator::RegisterPtr SVectorBuffer;
    if(std::holds_alternative<uint32_t>(vectorPtr)) {
        SVectorBuffer = scalarRegisterAllocator.getRegister("SVectorBuffer = X");
        c.li(*SVectorBuffer, std::get<uint32_t>(vectorPtr));
    }
    // Otherwise, use pointer register directly
    else {
        SVectorBuffer = std::get<ScalarRegisterAllocator::RegisterPtr>(vectorPtr);
    }

    // If literal is provided for number of vectors, load immediate and immediate to address
    if(std::holds_alternative<uint32_t>(numVectors)) {
        ALLOCATE_SCALAR(STmp);
        c.li(*STmp, std::get<uint32_t>(numVectors) * 64);
        c.add(*SVectorBufferEnd, *SVectorBuffer, *STmp);
    }
    // Otherwise, add register to address
    else {
        ALLOCATE_SCALAR(STmp);
        c.slli(*STmp, *std::get<ScalarRegisterAllocator::RegisterPtr>(numVectors), 6);
        c.add(*SVectorBufferEnd, *SVectorBuffer, *STmp);
    }

    // Load immedate
    c.li(*SOne, 1);

    // Loop over vectors
    c.L(vectorLoop);
    {
        // Register allocation
        ALLOCATE_VECTOR(VData);
        ALLOCATE_VECTOR(VLane);
        ALLOCATE_SCALAR(SMask);
        ALLOCATE_SCALAR(SVal);
        
        // Unroll lane loop
        for(int l = 0; l < 32; l++) {
            // Load halfword
            c.lh(*SVal, *SDataBuffer, l * 2);

            // SMask = 1 << SLane
            c.slli(*SMask, *SOne, l);

            // Fill vector register
            c.vfill(*VLane, *SVal);

            // VData = SMask ? VLane : VData
            c.vsel(*VData, *SMask, *VLane); 
        }

        // SDataBuffer += 64
        c.addi(*SDataBuffer, *SDataBuffer, 64);
      
        // *SVectorBuffer = VData
        c.vstore(*VData, *SVectorBuffer);

        // SVector += 64
        c.addi(*SVectorBuffer, *SVectorBuffer, 64);

        // If SVectorBuffer != SVectorBufferEnd, goto vector loop
        c.bne(*SVectorBuffer, *SVectorBufferEnd, vectorLoop);
    }
}
//----------------------------------------------------------------------------
void generateVectorScalarMemcpy(CodeGenerator &c, VectorRegisterAllocator &vectorRegisterAllocator,
                                ScalarRegisterAllocator &scalarRegisterAllocator,
                                uint32_t vectorPtr, uint32_t scalarPtr, uint32_t numVectors)
{
    // Register allocation
    ALLOCATE_SCALAR(SDataBuffer);
    ALLOCATE_SCALAR(SVectorBuffer)
    ALLOCATE_SCALAR(SVectorBufferEnd);

    // Labels
    Label vectorLoop;

    c.li(*SVectorBuffer, vectorPtr);
    c.li(*SVectorBufferEnd, vectorPtr + (numVectors * 64));

    // SDataBuffer = scalarPtr
    c.li(*SDataBuffer, scalarPtr);

    // Loop over vectors
    c.L(vectorLoop);
    {
        // Register allocation
        ALLOCATE_VECTOR(VData);
        ALLOCATE_SCALAR(SVal);

        // Load vector
        c.vloadv(*VData, *SVectorBuffer, 0);

        // **STALL**
        c.nop();
        
        // Unroll lane loop
        for(int l = 0; l < 32; l++) {
            // Extract lane into scalar registers
            c.vextract(*SVal, *VData, l);

            // Store halfword
            c.sh(*SVal, *SDataBuffer, l * 2);
        }

        // SVectorBuffer += 64
        c.addi(*SVectorBuffer, *SVectorBuffer, 64);

        // SDataBuffer += 64
        c.addi(*SDataBuffer, *SDataBuffer, 64);
      
        // If SVectorBuffer != SVectorBufferEnd, goto vector loop
        c.bne(*SVectorBuffer, *SVectorBufferEnd, vectorLoop);
    }
}
//----------------------------------------------------------------------------
void generatePerformanceCountWrite(CodeGenerator &c, ScalarRegisterAllocator &scalarRegisterAllocator,
                                   CSR lowCSR, CSR highCSR, uint32_t scalarPtr)
{
    ALLOCATE_SCALAR(SAddress);
    ALLOCATE_SCALAR(SValue);
    
    c.li(*SAddress, scalarPtr);
    c.csrr(*SValue, lowCSR);
    c.sw(*SValue, *SAddress);
    c.csrr(*SValue, highCSR);
    c.sw(*SValue, *SAddress, 4);
}
//----------------------------------------------------------------------------
void unrollLoopBody(CodeGenerator &c, uint32_t numIterations, uint32_t maxUnroll, 
                    Reg testBufferReg, Reg testBufferEndReg, 
                    std::function<void(CodeGenerator&, uint32_t)> genBodyFn, 
                    std::function<void(CodeGenerator&, uint32_t)> genTailFn)
{
    // **TODO** tail loop after unrolling
    assert((numIterations % maxUnroll) == 0);
    const uint32_t numUnrolls = std::min(numIterations, maxUnroll);
    const uint32_t numUnrolledIterations = numIterations / numUnrolls;

    // Input postsynaptic neuron loop
    Label loop;
    c.L(loop);
    {
        // Unroll loop
        for(uint32_t r = 0; r < numUnrolls; r++) {
            genBodyFn(c, r);
        }

        // If we haven't already unrolled everything, generate increment and loop
        // **TODO** this is incorrect if there's a tail
        if(numUnrolledIterations > 1) {
            genTailFn(c, numUnrolls);
        
            c.bne(testBufferReg, testBufferEndReg, loop);
        }
    }
}
//----------------------------------------------------------------------------
void unrollVectorLoopBody(CodeGenerator &c, uint32_t numIterations, uint32_t maxUnroll, 
                          Reg testBufferReg, Reg testBufferEndReg, 
                          std::function<void(CodeGenerator&, uint32_t)> genBodyFn, 
                          std::function<void(CodeGenerator&, uint32_t)> genTailFn)
{
    // Only loop bodies for now
    assert((numIterations % 32) == 0);
    unrollLoopBody(c, numIterations / 32, maxUnroll, 
                   testBufferReg, testBufferEndReg,
                   genBodyFn, genTailFn);

}
//----------------------------------------------------------------------------
std::vector<uint32_t> generateStandardKernel(bool simulate, uint32_t readyFlagPtr, 
                                             std::function<void(CodeGenerator&, VectorRegisterAllocator&, ScalarRegisterAllocator&)> genBodyFn)
{
    CodeGenerator c;
    VectorRegisterAllocator vectorRegisterAllocator;
    ScalarRegisterAllocator scalarRegisterAllocator;
    
    // If we're simulating, generate body and add ecall instruction
    if(simulate) {
        genBodyFn(c, vectorRegisterAllocator, scalarRegisterAllocator);
        c.ecall();
    }
    // Otherwise
    else {
        // Register allocation
        ALLOCATE_SCALAR(SReadyFlagBuffer);

        // Labels
        Label spinLoop;

        // Load ready flag
        c.li(*SReadyFlagBuffer, readyFlagPtr);
        
        // Clear ready flag
        c.sw(Reg::X0, *SReadyFlagBuffer);

        // Generate body
        genBodyFn(c, vectorRegisterAllocator, scalarRegisterAllocator);

        // Set ready flag
        {
            ALLOCATE_SCALAR(STmp);
            c.li(*STmp, 1);
            c.sw(*STmp, *SReadyFlagBuffer);
        }

        // Infinite loop
        c.L(spinLoop);
        {
            // **HACK** 
            //c.j_(spinLoop);
            c.beq(Reg::X0, Reg::X0, spinLoop);
        }
    }

    LOGI << "Max vector registers used: " << vectorRegisterAllocator.getMaxUsedRegisters();
    LOGI << "Max scalar registers used: " << scalarRegisterAllocator.getMaxUsedRegisters();

    return c.getCode();
}
//----------------------------------------------------------------------------
std::vector<uint32_t> generateInitCode(bool simulate, uint32_t startVectorPtr, uint32_t numVectorsPtr, uint32_t readyFlagPtr, uint32_t scalarScratchPtr)
{
    return generateStandardKernel(
        simulate, readyFlagPtr,
        [=](CodeGenerator &c, VectorRegisterAllocator &vectorRegisterAllocator, ScalarRegisterAllocator &scalarRegisterAllocator)
        {
            // Register allocation
            ALLOCATE_SCALAR(SNumVectorsPtr);
            ALLOCATE_SCALAR(SStartVectorPtr);

            // Load pointer to vector memory start address
            {
                ALLOCATE_SCALAR(STmp);
                c.li(*STmp, startVectorPtr);
                c.lw(*SStartVectorPtr, *STmp);
            }

            // Load count of number of vectors
            {
                ALLOCATE_SCALAR(STmp);
                c.li(*STmp, numVectorsPtr);
                c.lw(*SNumVectorsPtr, *STmp);
            }

            // Generate copying code
            generateScalarVectorMemcpy(c, vectorRegisterAllocator, scalarRegisterAllocator,
                                       scalarScratchPtr, SStartVectorPtr, SNumVectorsPtr);
        });
}
}

#pragma once

// Standard C++ includes
#include <functional>
#include <variant>

// Standard C includes
#include <cstdint>

// FeNN backend includes
#include "assembler.h"
#include "isa.h"
#include "registerAllocator.h"

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::FeNN::AssemblerUtils
//----------------------------------------------------------------------------
namespace GeNN::CodeGenerator::FeNN::AssemblerUtils
{
// Generate
void generateScalarVectorMemcpy(Assembler::CodeGenerator &c, FeNN::VectorRegisterAllocator &vectorRegisterAllocator,
                                ScalarRegisterAllocator &scalarRegisterAllocator,
                                std::variant<uint32_t, ScalarRegisterAllocator::RegisterPtr> scalarPtr, 
                                std::variant<uint32_t, ScalarRegisterAllocator::RegisterPtr> vectorPtr, 
                                std::variant<uint32_t, ScalarRegisterAllocator::RegisterPtr> numVectors);

void generateVectorScalarMemcpy(Assembler::CodeGenerator &c, VectorRegisterAllocator &vectorRegisterAllocator,
                                ScalarRegisterAllocator &scalarRegisterAllocator,
                                uint32_t vectorPtr, uint32_t scalarPtr, uint32_t numVectors);

// Generate code to copy 64-bit performance counter value from pair of CSR registers to scalar memory
void generatePerformanceCountWrite(Assembler::CodeGenerator &c, ScalarRegisterAllocator &scalarRegisterAllocator,
                                   CSR lowCSR, CSR highCSR, uint32_t scalarPtr);

// Generate an unrolled loop body
void unrollLoopBody(Assembler::CodeGenerator &c, uint32_t numIterations, uint32_t maxUnroll, 
                    Reg testBufferReg, Reg testBufferEndReg, 
                    std::function<void(Assembler::CodeGenerator&, uint32_t)> genBodyFn, 
                    std::function<void(Assembler::CodeGenerator&, uint32_t)> genTailFn);

// Generate an unrolled loop body for a vectorised loop
void unrollVectorLoopBody(Assembler::CodeGenerator &c, uint32_t numIterations, uint32_t maxUnroll, 
                          Reg testBufferReg, Reg testBufferEndReg, 
                          std::function<void(Assembler::CodeGenerator&, uint32_t)> genBodyFn, 
                          std::function<void(Assembler::CodeGenerator&, uint32_t)> genTailFn);

// Generate preamble and postamble for code using standard ecall instruction to terminate simulations and polling on device
std::vector<uint32_t> generateStandardKernel(bool simulate, uint32_t readyFlagPtr, 
                                             std::function<void(Assembler::CodeGenerator&, VectorRegisterAllocator&, ScalarRegisterAllocator&)> genBodyFn);

// Generate an initialisation kernel which copies dynamically-sized block of data from scalar memory into vector memory
std::vector<uint32_t> generateInitCode(bool simulate, uint32_t startVectorPtr, uint32_t numVectorsPtr, uint32_t readyFlagPtr, uint32_t scalarStartPtr);
}
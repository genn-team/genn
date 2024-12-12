#include "isa.h"

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::FeNN
//----------------------------------------------------------------------------
namespace GeNN::CodeGenerator::FeNN
{
LoadType getLoadType(uint32_t funct3)
{
    switch(funct3) {
    case 0:
        return LoadType::LB;

    case 1: 
        return LoadType::LH;

    case 2: 
        return LoadType::LW;
    
    case 4: 
        return LoadType::LBU;
    
    case 5:
        return LoadType::LHU;
    
    default:
        return LoadType::INVALID;
    }
}
//----------------------------------------------------------------------------
StoreType getStoreType(uint32_t funct3)
{
    switch(funct3) {
    case 0:
        return StoreType::SB;

    case 1:
        return StoreType::SH;

    case 2:
        return StoreType::SW;

    default:
        return StoreType::INVALID;
    }
}
//----------------------------------------------------------------------------
BranchType getBranchType(uint32_t funct3)
{
    switch(funct3) {
    case 0:
        return BranchType::BEQ;

    case 1:
        return BranchType::BNE;
    
    case 4:
        return BranchType::BLT;

    case 5:
        return BranchType::BGE;

    case 6:
        return BranchType::BLTU;

    case 7:
        return BranchType::BGEU;
    
    default:
        return BranchType::INVALID;
    }
}
//----------------------------------------------------------------------------
OpImmType getOpImmType(int32_t imm, uint32_t funct3)
{
    switch(funct3) {
    case 0:
        return OpImmType::ADDI;

    case 1:
    {
        // Split immediate into shamt (shift) and upper field
        const uint32_t shamt = imm & 0b11111;
        const uint32_t pre = imm >> 5;

        if (pre == 0) {
            return OpImmType::SLLI;
        }
        // CLZ/CPOP/CTZ/SEXT
        else if(pre == 0b110000) {
            switch(shamt) {
            case 0:
                return OpImmType::CLZ;

            case 1: // CTZ
                return OpImmType::CTZ;

            case 2:
                return OpImmType::CPOP;

            case 4:
                return OpImmType::SEXT_B;

            case 5:
                return OpImmType::SEXT_H;

            default:
                return OpImmType::INVALID;
            }
        }
        else {
            return OpImmType::INVALID;
        }
    }

    case 2:
        return OpImmType::SLTI;
    
    case 3:
        return OpImmType::SLTIU;
    
    case 4:
        return OpImmType::XORI;

    case 5: // SRLI / SRAI
    {
        // Error if any bits other than the 11th bit are set in immediate field
        if ((imm & ~(0b11111 | 0x400)) != 0) {
            return OpImmType::INVALID;
        }

        // If 11th bit set, SRAI
        if (imm & 0x400) {
            return OpImmType::SRAI;
        }
        // Otherwise SRLI
        else
        {
            return OpImmType::SRLI;
        }
    }
    case 6:
        return OpImmType::ORI;

    case 7:
        return OpImmType::ANDI;

    default:
        return OpImmType::INVALID;
    }
}
//----------------------------------------------------------------------------
OpType getOpType(int32_t funct7, uint32_t funct3)
{
    // If any bits are set in funct7 aside from the one used for distinguishing ops
    if (funct7 & ~0x20) {
        return OpType::INVALID;
    }
    switch(funct3) {
    case 0:
        return (funct7 == 0) ? OpType::ADD : OpType::SUB;

    case 1:
        return OpType::SLL;
    
    case 2:
        return OpType::SLT;

    case 3:
        return OpType::SLTU;

    case 4:
        return OpType::XOR;

    case 5:
        return (funct7 == 0) ? OpType::SRL : OpType::SRA;

    case 6:
        return OpType::OR;

    case 7:
        return OpType::AND;

    default:
        return OpType::INVALID;
    }
}
//----------------------------------------------------------------------------
VOpType getVOpType(uint32_t funct7, uint32_t funct3)
{
    const uint32_t roundMode = ((funct7 >> 4) & 0b011);
    switch(funct3)
    {
    case 0b000:
        return (roundMode == 0) ? VOpType::VADD : VOpType::INVALID;

    case 0b010:
        return (roundMode == 0) ? VOpType::VSUB : VOpType::INVALID;

    // VMUL
    case 0b100:
    {
        // Round-to-zero
        if(roundMode == 0b00) {
            return VOpType::VMUL;
        }
        // Round-to-nearest (half up)
        else if(roundMode == 0b01) {
            return VOpType::VMUL_RN;
        }
        // Stochastic round
        else if (roundMode == 0b10) {
            return VOpType::VMUL_RS;
        }
        else {
            return VOpType::INVALID;
        }
        
    }
    default:
    {
        return VOpType::INVALID;
    }
    }
}
//----------------------------------------------------------------------------
VTstType getVTstType(uint32_t funct3)
{
    switch(funct3)
    {
    case 0b000:
        return VTstType::VTEQ;

    case 0b010:
        return VTstType::VTNE;

    case 0b100:
        return VTstType::VTLT;

    case 0b110:
        return VTstType::VTGE;

    default:
        return VTstType::INVALID;
    }
}
//----------------------------------------------------------------------------
VMovType getVMovType(uint32_t funct3)
{
    switch(funct3) {
    case 0b000:
        return VMovType::VFILL;

    case 0b001:
        return VMovType::VEXTRACT;
    
    default:
        return VMovType::INVALID;
    }
}
//----------------------------------------------------------------------------
VLoadType getVLoadType(uint32_t funct3)
{
    switch(funct3){
    case 0b000:
        return VLoadType::VLOAD;
        
    case 0b001:
        return VLoadType::VLOAD_R0;
    
    case 0b101:
        return VLoadType::VLOAD_R1;
    
    default:
        return VLoadType::INVALID;
    }
}
//----------------------------------------------------------------------------
VSpcType getVSpcType(uint32_t funct3)
{
    switch(funct3){
    case 0b000:
        return VSpcType::VRNG;
        
    default:
        return VSpcType::INVALID;
    }
}
}
#include "assembler/assembler.h"

namespace
{
const char *ConvertErrorToString(int err)
{
    static const char *errTbl[] = {
        "none",
        "offset is too big",
        "code is too big",
        "imm is too big",
        "invalid imm of jal",
        "invalid imm of Btype",
        "label is not found",
        "label is redefined",
        "label is too far",
        "label is not set by L",
        "label is already set by L",
        "can't protect",
        "can't alloc",
        "bad parameter",
        "munmap",
        "internal error"
    };
    assert(ERR_MAX == (sizeof(errTbl) / sizeof(*errTbl)));
    return (err < ERR_MAX) ? errTbl[err] : "unknown err";
}

// split x to hi20bits and low12bits
// return false if x in 12-bit signed integer
inline bool split32bit(int *pH, int* pL, int x) {
    if (inSBit(x, 12)) {
        return false;
    }
    int H = (x >> 12) & mask(20);
    int L = x & mask(12);
    if (x & (1 << 11)) {
        H++;
        L = L | (mask(20) << 12);
    }
    *pH = H;
    *pL = L;
    return true;
}

inline uint32_t get20_10to1_11_19to12_z12(uint32_t v) { return ((v & (1<<20)) << 11)| ((v & (1023<<1)) << 20)| ((v & (1<<11)) << 9)| (v & (255<<12)); }
inline uint32_t get12_10to5_z13_4to1_11_z7(uint32_t v) { return ((v & (1<<12)) << 19)| ((v & (63<<5)) << 20)| ((v & (15<<1)) << 7)| ((v & (1<<11)) >> 4); }



}

//----------------------------------------------------------------------------
// Error
//----------------------------------------------------------------------------
Error::Error(int err) : err_(err)
{
    std::cout << "Error:" << err << std::endl;
    if (err_ < 0 || err_ > ERR_INTERNAL) {
        err_ = ERR_INTERNAL;
    }
}
//----------------------------------------------------------------------------
const char *Error::what() const noexcept
{
    return ConvertErrorToString(err_);
}


//----------------------------------------------------------------------------
// Label
//----------------------------------------------------------------------------
Label::Label(const Label& rhs)
{
    id = rhs.id;
    cg = rhs.cg;
    if (cg) {
        cg->incRefCount(id, this);
    }
}
//----------------------------------------------------------------------------
Label& Label::operator=(const Label& rhs)
{
    if (id) {
        throw Error(ERR_LABEL_IS_ALREADY_SET_BY_L);
    }
    id = rhs.id;
    cg = rhs.cg;
    if (cg) {
        cg->incRefCount(id, this);
    }
    return *this;
}
//----------------------------------------------------------------------------
Label::~Label()
{
    if (id && cg) {
        cg->decRefCount(id, this);
    }
}
//----------------------------------------------------------------------------
uint32_t Label::getAddress() const
{
    if (cg == nullptr) {
        return 0;
    }
    return cg->getAddr(*this);
}


//----------------------------------------------------------------------------
// CodeGenerator
//----------------------------------------------------------------------------
void CodeGenerator::li(Reg rd, int imm)
{
    int H, L;
    if (!split32bit(&H, &L, imm)) {
        addi(rd, Reg::X0, imm);
        return;
    }
    lui(rd, H);
    addi(rd, rd, L);
}

//----------------------------------------------------------------------------
// CodeGenerator::Jmp
//----------------------------------------------------------------------------
uint32_t CodeGenerator::Jmp::encode(uint32_t addr) const
{
    if (addr == 0) {
        return 0;
    }
    if (type == tRawAddress) {
        return addr;
    }
    const int imm = addr - from;
    if (type == tJal) {
        if (!isValidImm(imm, 20)) {
            throw Error(ERR_INVALID_IMM_OF_JAL);
        }
        return get20_10to1_11_19to12_z12(imm) | encoded;
    } else {
        if (!isValidImm(imm, 12)) {
            throw Error(ERR_INVALID_IMM_OF_JAL);
        }
        return get12_10to5_z13_4to1_11_z7(imm) | encoded;
    }
}
//----------------------------------------------------------------------------
void CodeGenerator::Jmp::update(CodeGenerator *base) const
{
    base->write4B(from, encode(base->getCurr()));
}
//----------------------------------------------------------------------------
void CodeGenerator::Jmp::appendCode(CodeGenerator *base, uint32_t addr) const
{
    base->append4B(encode(addr));
}

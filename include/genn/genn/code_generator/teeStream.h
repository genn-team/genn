#pragma once

// Standard C++ includes
#include <ostream>
#include <streambuf>
#include <vector>

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::TeeBuf
//--------------------------------------------------------------------------
// A stream buffer to support 'Teeing' streams - curtesy of http://wordaligned.org/articles/cpp-streambufs
namespace GeNN::CodeGenerator
{
class TeeBuf: public std::streambuf
{
public:
    // Construct a streambuf which tees output to multiple streambufs
    template<typename... T>
    TeeBuf(T&&... streamBufs) : m_StreamBufs({{streamBufs.rdbuf()...}})
    {
    }

private:
    //--------------------------------------------------------------------------
    // std::streambuf virtuals
    //--------------------------------------------------------------------------
    virtual int overflow(int c) override
    {
        if (c == EOF) {
            return !EOF;
        }
        else {
            bool anyEOF = false;
            for(auto &s: m_StreamBufs) {
                if(s->sputc(c) == EOF) {
                    anyEOF = true;
                }
            }
            return anyEOF ? EOF : c;
        }
    }
    
    // Sync all teed buffers.
    virtual int sync() override
    {
        bool anyNonZero = false;
        for(auto &s: m_StreamBufs) {
            if(s->pubsync() != 0) {
                anyNonZero = true;
            }
        }

        return anyNonZero ? -1 : 0;
    }   
private:
    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    const std::vector<std::streambuf*> m_StreamBufs;
};

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::TeeStream
//--------------------------------------------------------------------------
class TeeStream : public std::ostream
{
public:
    template<typename... T>
    TeeStream(T&&... streamBufs)
        : std::ostream(&m_TeeBuf), m_TeeBuf(std::forward<T>(streamBufs)...)
    {
    }
    
private:
    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    TeeBuf m_TeeBuf;
};
}   // namespace GeNN::CodeGenerator

#pragma once

// Standard C++ includes
#include <ostream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <vector>

// PLOG includes
#include <plog/Log.h>

// GeNN includes
#include "gennExport.h"

//----------------------------------------------------------------------------
// CodeGenerator::CodeStream
//----------------------------------------------------------------------------
//! Helper class for generating code - automatically inserts brackets, indents etc
/*! Based heavily on: https://stackoverflow.com/questions/15053753/writing-a-manipulator-for-a-custom-stream-class */
namespace CodeGenerator
{
class GENN_EXPORT CodeStream : public std::ostream
{
private:
    //------------------------------------------------------------------------
    // IndentBuffer
    //------------------------------------------------------------------------
    class IndentBuffer : public std::streambuf
    {
    public:
        IndentBuffer() : m_Sink(NULL), m_NewLine(false), m_IndentLevel(0){}

        //--------------------------------------------------------------------
        // Public API
        //--------------------------------------------------------------------
        void indent()
        {
            m_IndentLevel++;
        }

        void deindent()
        {
            m_IndentLevel--;
        }

        void setSink(std::streambuf *sink)
        {
            m_Sink = sink;
        }

    private:
        //--------------------------------------------------------------------
        // Streambuf overrides
        //--------------------------------------------------------------------
        virtual int overflow(int c) override;

        //----------------------------------------------------------------------------
        // Members
        //----------------------------------------------------------------------------
        std::streambuf *m_Sink;
        bool m_NewLine;
        unsigned int m_IndentLevel;
    };

public:
    //------------------------------------------------------------------------
    // OB
    //------------------------------------------------------------------------
    //! An open bracket marker
    /*! Write to code stream ``os`` using:
     * \code os << OB(16); \endcode */
    struct OB
    {
        OB(unsigned int level) : Level(level){}

        const unsigned int Level;
    };

    //------------------------------------------------------------------------
    // CB
    //------------------------------------------------------------------------
    //! A close bracket marker
    /*! Write to code stream ``os`` using:
     * \code os << CB(16); \endcode */
    struct CB
    {
        CB(unsigned int level) : Level(level){}

        const unsigned int Level;
    };

    //------------------------------------------------------------------------
    // Scope
    //------------------------------------------------------------------------
    class Scope
    {
    public:
        Scope(CodeStream &codeStream)
        :   m_CodeStream(codeStream), m_Level(s_NextLevel++)
        {
            m_CodeStream << CodeStream::OB(m_Level);
        }

        ~Scope()
        {
            // If we're not in the middle of handling an uncaught exception
            if(!std::uncaught_exception()) {
                try
                {
                    m_CodeStream << CodeStream::CB(m_Level);
                }
                catch(const std::runtime_error &exception)
                {
                    LOGE << exception.what();
                }
            }
        }

    private:
        //------------------------------------------------------------------------
        // Static members
        //------------------------------------------------------------------------
        GENN_EXPORT static unsigned int s_NextLevel;

        //------------------------------------------------------------------------
        // Members
        //------------------------------------------------------------------------
        CodeStream &m_CodeStream;
        const unsigned int m_Level;
    };

    CodeStream(): std::ostream(&m_Buffer) {
        m_Braces.push_back(0);
    }

    CodeStream(std::ostream &stream): CodeStream() {
        setSink(stream);
    }

    void setSink(std::ostream &stream)
    {
        m_Buffer.setSink(stream.rdbuf());
    }

private:
    //------------------------------------------------------------------------
    // Friends
    //------------------------------------------------------------------------
    GENN_EXPORT friend std::ostream&  operator << (std::ostream& s, const OB &ob);
    GENN_EXPORT friend std::ostream&  operator << (std::ostream& s, const CB &cb);

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    IndentBuffer m_Buffer;
    std::vector<unsigned int> m_Braces;
};

//------------------------------------------------------------------------
// Operators
//------------------------------------------------------------------------
GENN_EXPORT std::ostream& operator << (std::ostream& s, const CodeStream::OB &ob);
GENN_EXPORT std::ostream& operator << (std::ostream& s, const CodeStream::CB &cb);
}   // namespace CodeGenerator;



#pragma once

// Standard C++ includes
#include <functional>

// PLOG includes
#include <plog/Log.h>

//----------------------------------------------------------------------------
// SpineMLSimulator::StateVar
//----------------------------------------------------------------------------
namespace SpineMLSimulator
{
template <typename T>
class StateVar
{
public:
    StateVar(const std::string &stateVarName, std::function<void*(const char*,bool)> getLibrarySymbolFunc)
    {
        // Get host statevar
        T **hostStateVar = reinterpret_cast<T**>(getLibrarySymbolFunc(stateVarName.c_str(), true));

        // If there is no host statevar, it has probably been optimised away so isn't accesible
        if(hostStateVar == nullptr) {
            m_Access = Access::None;
        }
        // Otherwise
        else {
            LOGD << "\t" << stateVarName;

            // If there is a function to get the current state of variable
            GetCurrentFunc getCurrentFunc = reinterpret_cast<GetCurrentFunc>(getLibrarySymbolFunc(("getCurrent" + stateVarName).c_str(), true));
            if(getCurrentFunc) {
                // Set access mode to indirect
                m_Access = Access::Indirect;

                // Populate 'indirect' structure
                m_Indirect.getFunc = getCurrentFunc;
                m_Indirect.pushFunc = reinterpret_cast<PushCurrentFunc>(getLibrarySymbolFunc(("pushCurrent" + stateVarName + "ToDevice").c_str(), false));
                m_PullFunc = reinterpret_cast<PullFunc>(getLibrarySymbolFunc(("pullCurrent" + stateVarName + "FromDevice").c_str(), false));

                LOGD << "\t\tIndirect with get function:" << m_Indirect.getFunc << ", push function:" << m_Indirect.pushFunc << ", pull function:" << m_PullFunc;
            }
            // Otherwise
            else {
                // Set access mode, to direct
                m_Access = Access::Direct;

                // Populate 'direct' structure
                m_Direct.hostStateVar = *hostStateVar;
                m_Direct.pushFunc = reinterpret_cast<PushFunc>(getLibrarySymbolFunc(("push" + stateVarName + "ToDevice").c_str(), false));
                m_PullFunc = reinterpret_cast<PullFunc>(getLibrarySymbolFunc(("pull" + stateVarName + "FromDevice").c_str(), false));

                LOGD << "\t\tDirect with host pointer:" << m_Direct.hostStateVar << ", push function:" << m_Direct.pushFunc << ", pull function:" << m_PullFunc;
            }
        }
    }

    bool isAccessible() const{ return (m_Access != Access::None); }

    void push() const
    {
        if(m_Access == Access::Indirect) {
            m_Indirect.pushFunc();
        }
        else if(m_Access == Access::Direct) {
            m_Direct.pushFunc(false);
        }
        else {
            throw std::runtime_error("Unable to push inaccessible variable");
        }
    }

    void pull() const
    {
        if(m_Access == Access::None) {
            throw std::runtime_error("Unable to pull inaccessible variable");
        }
        else {
            m_PullFunc();
        }
    }

    T *get()
    {
        if(m_Access == Access::Indirect) {
            return m_Indirect.getFunc();
        }
        else if(m_Access == Access::Direct) {
            return m_Direct.hostStateVar;
        }
        else {
            throw std::runtime_error("Unable to get inaccessible variable");
        }

    }
    const T *get() const
    {
        if(m_Access == Access::Indirect) {
            return m_Indirect.getFunc();
        }
        else if(m_Access == Access::Direct) {
            return m_Direct.hostStateVar;
        }
        else {
            throw std::runtime_error("Unable to get inaccessible variable");
        }
    }

private:
    //--------------------------------------------------------------------
    // Typedefines
    //--------------------------------------------------------------------
    typedef T *(*GetCurrentFunc)(void);
    typedef void (*PushFunc)(bool);
    typedef void (*PullFunc)(void);
    typedef void (*PushCurrentFunc)(void);

    //--------------------------------------------------------------------
    // Enumerations
    //--------------------------------------------------------------------
    //! Different means by which this struct can access variables
    enum class Access
    {
        None,
        Direct,
        Indirect,
    };

    //--------------------------------------------------------------------
    // Indirect
    //--------------------------------------------------------------------
    //! Struct containing pointers specific to variables accessed via
    //! Indirect getCurrentXXX methods
    struct Indirect
    {
        GetCurrentFunc getFunc;
        PushCurrentFunc pushFunc;
    };

    //--------------------------------------------------------------------
    // Direct
    //--------------------------------------------------------------------
    //! Struct containing pointers specific to variables
    //! accessed directly via state variable pointer
    struct Direct
    {
        T *hostStateVar;
        PushFunc pushFunc;
    };

    //--------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------
    PullFunc m_PullFunc;
    Access m_Access;

    union
    {
        Indirect m_Indirect;
        Direct m_Direct;
    };
};
}   // namespace SpineMLSimulator

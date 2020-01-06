#pragma once

// Standard C++ includes
#include <chrono>
#include <iostream>
#include <string>

// SpineML common includes
#include "spineMLLogging.h"

//------------------------------------------------------------------------
// SpineMLSimulator::Timer
//------------------------------------------------------------------------
namespace SpineMLSimulator
{
class Timer
{
public:
    Timer(const std::string &title) : m_Start(std::chrono::high_resolution_clock::now()), m_Title(title)
    {
    }

    ~Timer()
    {
        LOGI_SPINEML << m_Title << get() << std::endl;
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    double get() const
    {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = now - m_Start;
        return duration.count();
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::chrono::time_point<std::chrono::high_resolution_clock> m_Start;
    std::string m_Title;
};

//------------------------------------------------------------------------
// SpineMLSimulator::TimerAccumulate
//------------------------------------------------------------------------
class TimerAccumulate
{
public:
    TimerAccumulate(double &accumulator) : m_Start(std::chrono::high_resolution_clock::now()), m_Accumulator(accumulator)
    {
    }

    ~TimerAccumulate()
    {
        m_Accumulator += get();
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    double get() const
    {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = now - m_Start;
        return duration.count();
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::chrono::time_point<std::chrono::high_resolution_clock> m_Start;
    double &m_Accumulator;
};
}

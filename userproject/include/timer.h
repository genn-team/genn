#pragma once

// Standard C++ includes
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

//------------------------------------------------------------------------
// Timer
//------------------------------------------------------------------------
//! A generic timer which can give the current elapsed time
class Timer
{
public:
    //! Create a new Timer with the specified name and optionally a filename to append time to
    Timer(const std::string &message, const std::string &filename = "")
    :   m_Start(std::chrono::high_resolution_clock::now()), m_Message(message), m_Filename(filename)
    {
    }

    //! Stop the timer and print current elapsed time to terminal
    ~Timer()
    {
        const double duration = get();
        std::cout << m_Message << duration << " seconds" << std::endl;

        // If we specified a filename, write duration
        if(!m_Filename.empty()) {
            std::ofstream output(m_Filename, std::ios::app);
            output << duration << std::endl;
        }
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Get the elapsed time since this object was created
    double get() const
    {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = now - m_Start;
        return duration.count();
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const std::chrono::time_point<std::chrono::high_resolution_clock> m_Start;
    const std::string m_Message;
    const std::string m_Filename;
};


//------------------------------------------------------------------------
// TimerAccumulate
//------------------------------------------------------------------------
//! A timer which adds its elapsed time to an accumulator variable on destruction
class TimerAccumulate
{
public:
    TimerAccumulate(double &accumulator) : m_Start(std::chrono::high_resolution_clock::now()), m_Accumulator(accumulator)
    {}

    ~TimerAccumulate()
    {
        m_Accumulator += get();
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Get the elapsed time since this object was created
    double get() const
    {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = now - m_Start;
        return duration.count();
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::chrono::time_point<std::chrono::high_resolution_clock> m_Start;
    double &m_Accumulator;
};

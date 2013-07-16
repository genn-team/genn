//--------------------------------------------------------------------------
/*! \file hr_time.h

\brief This header file contains the definition of the CStopWatch class that implements a simple timing tool using the system clock.
*/
//--------------------------------------------------------------------------

#ifndef __HR_TIME_H
#define __HR_TIME_H

#ifdef _WIN32
#include <windows.h>

typedef struct {
    LARGE_INTEGER start;
    LARGE_INTEGER stop;
} stopWatch;

class CStopWatch {

private:
	stopWatch timer;
	LARGE_INTEGER frequency;
	double LIToSecs( LARGE_INTEGER & L);
public:
	CStopWatch();
	void startTimer( );
	void stopTimer( );
	double getElapsedTime();
};

#else
#include <sys/time.h>

typedef struct {
	timeval start;
	timeval stop;
} stopWatch;

class CStopWatch {

private:
	stopWatch timer;
public:
	CStopWatch() {};
	void startTimer( );
	void stopTimer( );
	double getElapsedTime();
};

#endif

#endif

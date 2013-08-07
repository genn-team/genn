//--------------------------------------------------------------------------
/*! \file hr_time.cpp

\brief This  file contains the implementation of the CStopWatch class that provides a simple timing tool based on the system clock.
*/
//--------------------------------------------------------------------------


#include <cstdio> // NULL
#include "hr_time.h"

#ifdef _WIN32
#include <windows.h>

double CStopWatch::LIToSecs( LARGE_INTEGER & L) {
	return ((double)L.QuadPart /(double)frequency.QuadPart);
}

CStopWatch::CStopWatch(){
	timer.start.QuadPart=0;
	timer.stop.QuadPart=0;	
	QueryPerformanceFrequency( &frequency );
}

void CStopWatch::startTimer( ) {
    QueryPerformanceCounter(&timer.start);
}

void CStopWatch::stopTimer( ) {
    QueryPerformanceCounter(&timer.stop);
}


double CStopWatch::getElapsedTime() {
	LARGE_INTEGER time;
	time.QuadPart = timer.stop.QuadPart - timer.start.QuadPart;
    return LIToSecs( time) ;
}
#else

//--------------------------------------------------------------------------
/*! \brief This method starts the timer
 */
//--------------------------------------------------------------------------

void CStopWatch::startTimer( ) {
	gettimeofday(&(timer.start),NULL);
}

//--------------------------------------------------------------------------
/*! \brief This method stops the timer
 */
//--------------------------------------------------------------------------

void CStopWatch::stopTimer( ) {
	gettimeofday(&(timer.stop),NULL);
}

//--------------------------------------------------------------------------
/*! \brief This method returns the time elapsed between start and stop of the timer in seconds.
 */
//--------------------------------------------------------------------------

double CStopWatch::getElapsedTime() {	
	timeval res;
	timersub(&(timer.stop),&(timer.start),&res);
	return res.tv_sec + res.tv_usec/1000000.0; // 10^6 uSec per second
}

#endif

//--------------------------------------------------------------------------
/*! \brief Template function for string conversion 
 */
//--------------------------------------------------------------------------

template<typename T> std::string toString(T t)
{
  std::stringstream s;
  s << t;
  return s.str();
} 

//--------------------------------------------------------------------------
/*! \brief function to detect the architecture of the windows operating 
  system that is being used
 */
//--------------------------------------------------------------------------

#ifdef _WIN32

string detectWindowsArch()
{
    string archChoice;
    string arch= tS(getenv("PROCESSOR_ARCHITECTURE"));
    if (arch == tS("AMD64")) {
	archChoice= tS("x64");
    }
    else {
	arch= tS(getenv("PROCESSOR_ARCHITEW6432"));
	if (arch == tS("AMD64")) {
	    archChoice= tS("x64");
	}
    }
    else {
	archChoice= tS("x86"):
    }
    return archChoice;
}

#endif

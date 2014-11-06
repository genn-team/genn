
#include "toString.h"

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
	else {
	  archChoice= tS("x86");
	}
    }
    return archChoice;
}

string ensureCompilerEnvironmentCmd()
{
  string cmd;
  cmd= tS("where /Q nmake.exe");
  int res= system(cmd.c_str());
  if (res != 0) {
    string archChoice= detectWindowsArch();
    cmd = tS("\"")+tS(getenv("VS_PATH"))+tS("\\VC\\vcvarsall.bat\" ")+archChoice+tS(" && ");
  }
  return cmd;
}

#endif

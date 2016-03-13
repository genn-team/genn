  // sets gennPath and modelRoot(for non-Windows build)
  string gennPath;
#ifndef _WIN32
  string modelRoot;
  {
    int i;
    modelRoot = (string)realpath(argv[0], NULL);
    i = modelRoot.length() - 1;
    while (modelRoot[i--] != '/');
    modelRoot.erase(i+1, modelRoot.length() - i);
  }
#endif

  try {
      gennPath += (string)getenv("GENN_PATH");
  }
  catch(std::logic_error &ex) {
      cerr << "Environment variable GENN_PATH not found..." << endl;
#ifdef _WIN32
      cerr << "Exiting..." << endl;
      exit(1);
#else
      gennPath = modelRoot + "/../..";
      gennPath = (string)realpath(gennPath.c_str(), NULL);
      cout << "Autodetecting GENN_PATH=" << gennPath << endl;
#endif
  }

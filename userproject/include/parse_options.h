  // parse options
  unsigned int dbgMode= 0;
  string ftype= "FLOAT";
  unsigned int fixsynapse= 0; 
  unsigned int cpu_only= 0;
  string option;
  for (int i= argStart; i < argc; i++) {
      if (extract_option(argv[i],option) != 0) {
	  cerr << "Unknown option '" << argv[i] << "'." << endl;
	  exit(1);
      }
      else {
	  if (option == "DEBUG") {
	      if (extract_bool_value(argv[i],dbgMode) != 0) {
		  cerr << "illegal value for 'DEBUG' option." << endl;
		  exit(1);
	      }
	  }
	  else if (option == "FTYPE") {
	      extract_string_value(argv[i],ftype);
	      if ((ftype != "FLOAT") && (ftype != "DOUBLE")) {
		  cerr << "illegal value " << ftype << " of 'FTYPE' option." <<endl;
		  exit(1);
	      }
	  }
	  else if (option == "REUSE") {
	      if (extract_bool_value(argv[i], fixsynapse) != 0) {
		  cerr << "illegal value for 'REUSE' option." << endl;
		  exit(1);
	      }
	  }
	  else if (option == "CPU_ONLY") {
	      if (extract_bool_value(argv[i], cpu_only) != 0) {
		  cerr << "illegal value for 'CPU_ONLY' option." << endl;
		  exit(1);
	      }
	  }
      }
  }
  if (cpu_only && (which == 1)) {
      cerr << "You cannot use a GPU in CPU_only mode." << endl;
      exit(1);
  }

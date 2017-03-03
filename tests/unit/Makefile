

CXX = g++
CXXFLAGS = -std=c++11 -I "$(GTEST_DIR)" -isystem "$(GTEST_DIR)/include" -I "$(GENN_PATH)/lib/include" -I "$(GENN_PATH)/userproject/include" -DCPU_ONLY
LINK_FLAGS = -lpthread

SOURCES = codeGenUtils.cc $(GTEST_DIR)/src/gtest-all.cc $(GTEST_DIR)/src/gtest_main.cc $(GENN_PATH)/lib/src/codeGenUtils.cc
OBJECTS =$(foreach obj,$(basename $(SOURCES)),$(obj).o)

%.o: %.cc
	$(CXX)  -c -o $@ $< $(CXXFLAGS)

test: $(OBJECTS) 
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LINK_FLAGS)
	
clean:
	rm -f *.o test
CXXFLAGS	+=-std=c++11 -Wall -Wpedantic -Wextra -I $(GTEST_DIR) -isystem $(GTEST_DIR)/include
OS_UPPER	:=$(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
DARWIN   	:=$(strip $(findstring DARWIN,$(OS_UPPER)))

ifeq ($(DARWIN),DARWIN)
    CXXFLAGS	+=-Wno-return-type-c-linkage
endif

.PHONY: all clean generated_code

all: test

test: test.cc generated_code
	$(CXX) $(CXXFLAGS) test.cc $(GTEST_DIR)/src/gtest-all.cc $(GTEST_DIR)/src/gtest_main.cc -o test -L$(SIM_CODE) -pthread -lrunner -Wl,-rpath $(SIM_CODE)
	
generated_code: 
	$(MAKE) -C $(SIM_CODE)

clean:
	@rm -f test $(SIM_CODE)/librunner.so $(SIM_CODE)/*.o $(SIM_CODE)/*.d default.profraw

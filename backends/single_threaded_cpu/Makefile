SOURCES		:= $(wildcard src/*.cc)
OBJECTS		:=$(SOURCES:.cc=.o)
DEPS		:=$(SOURCES:.cc=.d)
#ARFLAGS		:= -rcs

# **TODO** debug/MPI targets + CLang
CXXFLAGS += -std=c++11 -Wall -Wpedantic -Wextra -MMD -MP -Iinclude -I../../lib_genn/include -I../../third_party

.PHONY: all clean

all: libsingle_threaded_cpu_backend.a

libsingle_threaded_cpu_backend.a: $(OBJECTS)
	$(AR) $(ARFLAGS) $@ $(OBJECTS)

-include $(DEPS)

%.o: %.cc %.d
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.d: ;

clean:
	rm -f $(OBJECTS) $(DEPS) libsingle_threaded_cpu_backend.a

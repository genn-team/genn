FRONTEND_SOURCES	:= $(wildcard src/*.cc)
BACKEND_SOURCES		:= $(wildcard src/code_generator/*.cc)
SOURCES			:= $(FRONTEND_SOURCES) $(BACKEND_SOURCES) 
OBJECTS		:=$(SOURCES:.cc=.o)
DEPS		:=$(SOURCES:.cc=.d)
#ARFLAGS		:= -rcs

# **TODO** debug/MPI targets + CLang
CXXFLAGS += -std=c++11 -Wall -Wpedantic -Wextra -MMD -MP -Iinclude -I../third_party

.PHONY: all clean

all: libgenn.a

libgenn.a: $(OBJECTS)
	$(AR) $(ARFLAGS) $@ $(OBJECTS)

-include $(DEPS)

%.o: %.cc %.d
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.d: ;

clean:
	rm -f $(OBJECTS) $(DEPS) libantworld.a

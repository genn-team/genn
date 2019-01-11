# Include common makefile
include ../MakefileCommon

# Add include paths to CXX flags
CXXFLAGS		+= -I../lib_genn/include -I../third_party

# BUILD_MODEL_INCLUDE contains a colon-seperated list of additional include paths.
# substitute :s for spaces and then prepend each path with -I so it gets turned into an include directory
CXXFLAGS		+=$(patsubst %,-I%,$(subst :, ,$(BUILD_MODEL_INCLUDE)))

CXXFLAGS += -I../backends/$(BACKEND_NAME)/include -DMODEL=\"$(MODEL)\" -DBACKEND_NAMESPACE=$(BACKEND_NAMESPACE)

# Add prefix to generator name
GENERATOR		:=generator$(GENN_PREFIX).a

.PHONY: all clean libgenn backend

all: $(GENERATOR)

$(GENERATOR): generator.cc libgenn backend
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) generator.cc -o $@ -L../lib -lgenn$(GENN_PREFIX) -l$(BACKEND_NAME)_backend$(GENN_PREFIX)

libgenn:
	$(MAKE) -C ../lib_genn

backend:
	$(MAKE) -C ../backends/$(BACKEND_NAME)

clean:
	rm -f generator.o $(GENERATOR)

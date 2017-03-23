EXECUTABLE      := test
SOURCES         := test.cc $(GTEST_DIR)/src/gtest-all.cc $(GTEST_DIR)/src/gtest_main.cc

INCLUDE_FLAGS	:= -I $(GTEST_DIR) -isystem $(GTEST_DIR)/include 
LINK_FLAGS	:= -lpthread

ifdef SIM_CODE
	INCLUDE_FLAGS += -DMODEL_NAME=$(subst _CODE,,$(SIM_CODE)) -DDEFINITIONS_HEADER='"$(SIM_CODE)/definitions.h"'
endif

include $(GENN_PATH)/userproject/include/makefile_common_gnu.mk

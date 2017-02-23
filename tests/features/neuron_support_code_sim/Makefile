EXECUTABLE      := test
SOURCES         := test.cu $(GTEST_DIR)/src/gtest-all.cc $(GTEST_DIR)/src/gtest_main.cc

INCLUDE_FLAGS	:= -I $(GTEST_DIR) -isystem $(GTEST_DIR)/include
LINK_FLAGS	:= -lpthread

include $(GENN_PATH)/userproject/include/makefile_common_gnu.mk

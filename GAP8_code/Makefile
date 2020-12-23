# User Test
#------------------------------------

# include common/common.mk

override CONFIG_OPT += hyperflash hyperflash/fs/files=$(CURDIR)/HeySnips/quant_model_L3_Flash_Const.dat
RAM_FLASH_TYPE ?= HYPER
APP_CFLAGS += -DUSE_HYPER
MODEL_L3_EXEC=hram
MODEL_L3_CONST=hflash

QUANT_BITS?=8

MODEL_SUFFIX = _SQ8BIT

include common/model_decl.mk

CLUSTER_STACK_SIZE=4028
CLUSTER_SLAVE_STACK_SIZE=1024
TOTAL_STACK_SIZE=$(shell expr $(CLUSTER_STACK_SIZE) \+ $(CLUSTER_SLAVE_STACK_SIZE) \* 7)
MODEL_L1_MEMORY=$(shell expr 60000 \- $(TOTAL_STACK_SIZE))
MODEL_L2_MEMORY=250000
MODEL_L3_MEMORY=8388608

CNN_AT_PATH = $(TILER_CNN_GENERATOR_PATH)

PULP_APP              += test
PULP_APP_SRCS        += main.c mpr_math.c mpr_const_structs.c mpr_common_tables.c HeySnips/quant_modelKernels.c $(MODEL_LIB_SQ8)#$(TILER_LIB)
PULP_CFLAGS      += -g -O3 -mno-memcpy -fno-tree-loop-distribute-patterns 


READFS_FILES= $(abspath HeySnips/quant_model_L3_Flash_Const.dat)
PLPBRIDGE_FLAGS += -f -hyper

include $(RULES_DIR)/pulp_rules.mk 
INC	        += $(MODEL_COMMON_INC) $(TILER_EMU_INC) $(TILER_INC) $(CNN_LIB_INCLUDE) $(NNTOOL_KERNELS_PATH) ./HeySnips/outputs ./HeySnips . $(CNN_AT_PATH) $(MODEL_LIB_INCLUDE_SQ8) $(TILER_CNN_KERNEL_PATH_SQ8)


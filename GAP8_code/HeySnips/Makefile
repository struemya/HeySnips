# Copyright (C) 2017 GreenWaves Technologies
# All rights reserved.

# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

# APP = quantModel
RM=rm -f

#To Enable Bridge functions to read/write files from host 
#(this function will be replaced by openocd semihosting in next SDK release)

USE_BRIDGE=0

ifeq ($(USE_BRIDGE),1)
APP_CFLAGS += -DENABLE_BRIDGE
PLPBRIDGE_FLAGS += -fileIO
#To test new bridge features:
#comment the line above and use this two instead
#export GAP_USE_OPENOCD=1
#io=host
endif

CNN_AT_PATH = $(TILER_CNN_GENERATOR_PATH)

MODEL_LIB_SQ8 += $(TILER_CNN_KERNEL_PATH_SQ8)/CNN_BasicKernels_SQ8.h
MODEL_LIB_SQ8 += $(TILER_CNN_KERNEL_PATH_SQ8)/CNN_Activation_SQ8.c
MODEL_LIB_SQ8 += $(TILER_CNN_KERNEL_PATH_SQ8)/CNN_Bias_Linear_SQ8.c
MODEL_LIB_SQ8 += $(TILER_CNN_KERNEL_PATH_SQ8)/CNN_Conv_SQ8.c
MODEL_LIB_SQ8 += $(TILER_CNN_KERNEL_PATH_SQ8)/CNN_Pooling_SQ8.c
MODEL_LIB_SQ8 += $(TILER_CNN_KERNEL_PATH_SQ8)/CNN_Conv_DW_SQ8.c
MODEL_LIB_SQ8 += $(TILER_CNN_KERNEL_PATH_SQ8)/CNN_MatAlgebra_SQ8.c
MODEL_LIB_SQ8 += $(TILER_CNN_KERNEL_PATH_SQ8)/CNN_SoftMax_SQ8.c
MODEL_LIB_SQ8 += $(TILER_CNN_KERNEL_PATH_SQ8)/CNN_AT_Misc.c
MODEL_LIB_SQ8 += $(TILER_CNN_KERNEL_PATH_SQ8)/RNN_SQ8.c
MODEL_LIB_SQ8 += $(NNTOOL_KERNELS_PATH)/norm_transpose.c
MODEL_LIB_INCLUDE_SQ8 = -I$(TILER_CNN_KERNEL_PATH) -I$(TILER_CNN_KERNEL_PATH_SQ8) -I$(NNTOOL_KERNELS_PATH)
MODEL_GEN_SQ8 += $(TILER_CNN_GENERATOR_PATH)/CNN_Generator_Util.c
MODEL_GEN_SQ8 += $(TILER_CNN_GENERATOR_PATH_SQ8)/CNN_Generators_SQ8.c
MODEL_GEN_SQ8 += $(TILER_CNN_GENERATOR_PATH_SQ8)/RNN_Generators_SQ8.c
MODEL_GEN_SQ8 += $(NNTOOL_GENERATOR_PATH)/nntool_extra_generators.c
MODEL_GEN_INCLUDE_SQ8 = -I$(TILER_CNN_GENERATOR_PATH) -I$(TILER_CNN_GENERATOR_PATH_SQ8) -I$(NNTOOL_GENERATOR_PATH)

#Uncomment to use freertos
#PMSIS_OS ?= freerto
USE_PMSIS_BSP = 1

MODEL_PATH=./
MODEL_NAME=quant_model

L1_SPACE=48000

# The double colon allows us to force this to occur before the imported all target
# Link model generation to all step
# all: model

# Build the code generator
GenTile:
	gcc -o Gen$(MODEL_NAME) $(MODEL_LIB_INCLUDE_SQ8) $(MODEL_GEN_INCLUDE_SQ8) -I$(TILER_INC) -I$(TILER_EMU_INC) -I$(TILER_CNN_KERNEL_PATH_SQ8)-I$(CNN_AT_PATH) $(MODEL_GEN_SQ8) $(MODEL_NAME).c $(TILER_LIB) 

# Run the code generator
$(MODEL_NAME)Kernels.c: GenTile
	./Gen$(MODEL_NAME) --L1 $(L1_SPACE)
	sed -i 's/quant_model.h/main.h/g' quant_modelKernels.h

model: run_nntool gen_nntool $(MODEL_NAME)Kernels.c

run_nntool: 
	nntool -s ../nntool_script -q $(MODEL_PATH)$(MODEL_NAME).tflite

gen_nntool: 
	nntool -g $(MODEL_NAME).json -M . -m $(MODEL_NAME).c -T ./tensors --header_file $(MODEL_NAME)_info.h

clean::
	$(RM) Gen$(MODEL_NAME) $(MODEL_NAME)Kernels.* $(MODEL_NAME)_L3_Flash_Const.dat
	$(RM) -r tensors $(MODEL_NAME)_info.h $(MODEL_NAME).c $(MODEL_NAME).json $(MODEL_NAME).nnparam

.PHONY: run_nntool gen_nntool model clean

include $(GAP_SDK_HOME)/tools/rules/pmsis_rules.mk

#include <stdint.h>
#include <stdio.h>
#include "AutoTilerLib.h"
#include "CNN_Generators_SQ8.h"
#include "RNN_Generators_SQ8.h"

#include "nntool_extra_generators.h"





void quant_modelModel(unsigned int L1Memory, unsigned int L2Memory, unsigned int L3Memory, unsigned int L3Flash)
{
    KernelOper_T Cop = KOP_CONV;

    // SetKernelOpts(KER_OPT_NONE, KER_OPT_BUFFER_PROMOTE);
    SetSymbolDynamics();

    SetUsedFilesNames(0, 3, "nntool_extra_kernels.h", "CNN_BasicKernels_SQ8.h", "quant_model.h");
    SetGeneratedFilesNames("quant_modelKernels.c", "quant_modelKernels.h");
    AT_SetGraphCtrl(AT_GRAPH_MONITOR_CYCLES, AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_PRODUCE_NODE_NAMES, AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_PRODUCE_OPERINFOS, AT_OPT_ON);

    SetMemoryDeviceInfos(4,
        AT_MEM_L1, L1Memory, "quant_model_L1_Memory", 0, 0,
        AT_MEM_L2, L2Memory, "quant_model_L2_Memory", 0, 1,
        AT_MEM_L3_HRAM, L3Memory, "quant_model_L3_Memory", 0, 0,
        AT_MEM_L3_HFLASH, L3Flash, "quant_model_L3_Flash", "quant_model_L3_Flash_Const.dat", 0
    );

    LoadCNN_SQ8_Library();
    Load_RNN_SQ8_Library();

    LoadNNTools_Extra_Library();

    // generator for CONV_2D_0_3
    CNN_GenControl_T gen_ctrl_S2_Conv2d_16x20x1x3;
    CNN_InitGenCtrl(&gen_ctrl_S2_Conv2d_16x20x1x3);
    CNN_SetGenCtrl(&gen_ctrl_S2_Conv2d_16x20x1x3, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S2_Conv2d_16x20x1x3", &gen_ctrl_S2_Conv2d_16x20x1x3, 4, 1, 20, 16, 299, 1,
        KOP_CONV, 3, 1, 1, 1, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_NONE);
    // generator for CONV_2D_0_5_fusion
    CNN_GenControl_T gen_ctrl_S3_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S3_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S3_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S3_Conv2d_16x16x1x3_Relu", &gen_ctrl_S3_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 1, 1, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for CONV_2D_0_7_fusion
    CNN_GenControl_T gen_ctrl_S4_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S4_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S4_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S4_Conv2d_16x16x1x3_Relu", &gen_ctrl_S4_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 1, 1, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for ADD_0_8fusion
    CNN_MatAddAct_SQ8("S5_MatAdd_16x1x299_Relu", 0, 16, 1, 299, KOP_MATADD, KOP_RELU);
    // generator for CONV_2D_0_10_fusion
    CNN_GenControl_T gen_ctrl_S6_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S6_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S6_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S6_Conv2d_16x16x1x3_Relu", &gen_ctrl_S6_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 2, 2, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for CONV_2D_0_12_fusion
    CNN_GenControl_T gen_ctrl_S7_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S7_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S7_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S7_Conv2d_16x16x1x3_Relu", &gen_ctrl_S7_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 2, 2, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for ADD_0_13
    CNN_MatAddAct_SQ8("S8_MatAdd_16x1x299", 0, 16, 1, 299, KOP_MATADD, KOP_NONE);
    // generator for ADD_0_14
    CNN_MatAddAct_SQ8("S9_MatAdd_16x1x299", 0, 16, 1, 299, KOP_MATADD, KOP_NONE);
    // generator for CONV_2D_0_16_fusion
    CNN_GenControl_T gen_ctrl_S10_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S10_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S10_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S10_Conv2d_16x16x1x3_Relu", &gen_ctrl_S10_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 4, 4, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for CONV_2D_0_18_fusion
    CNN_GenControl_T gen_ctrl_S11_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S11_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S11_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S11_Conv2d_16x16x1x3_Relu", &gen_ctrl_S11_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 4, 4, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for ADD_0_19
    CNN_MatAddAct_SQ8("S12_MatAdd_16x1x299", 0, 16, 1, 299, KOP_MATADD, KOP_NONE);
    // generator for ADD_0_20
    CNN_MatAddAct_SQ8("S13_MatAdd_16x1x299", 0, 16, 1, 299, KOP_MATADD, KOP_NONE);
    // generator for CONV_2D_0_22_fusion
    CNN_GenControl_T gen_ctrl_S14_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S14_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S14_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S14_Conv2d_16x16x1x3_Relu", &gen_ctrl_S14_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 8, 8, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for CONV_2D_0_24_fusion
    CNN_GenControl_T gen_ctrl_S15_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S15_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S15_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S15_Conv2d_16x16x1x3_Relu", &gen_ctrl_S15_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 8, 8, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for ADD_0_25
    CNN_MatAddAct_SQ8("S16_MatAdd_16x1x299", 0, 16, 1, 299, KOP_MATADD, KOP_NONE);
    // generator for ADD_0_26
    CNN_MatAddAct_SQ8("S17_MatAdd_16x1x299", 0, 16, 1, 299, KOP_MATADD, KOP_NONE);
    // generator for CONV_2D_0_28_fusion
    CNN_GenControl_T gen_ctrl_S18_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S18_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S18_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S18_Conv2d_16x16x1x3_Relu", &gen_ctrl_S18_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 16, 16, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for CONV_2D_0_30_fusion
    CNN_GenControl_T gen_ctrl_S19_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S19_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S19_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S19_Conv2d_16x16x1x3_Relu", &gen_ctrl_S19_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 16, 16, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for ADD_0_31
    CNN_MatAddAct_SQ8("S20_MatAdd_16x1x299", 0, 16, 1, 299, KOP_MATADD, KOP_NONE);
    // generator for ADD_0_32
    CNN_MatAddAct_SQ8("S21_MatAdd_16x1x299", 0, 16, 1, 299, KOP_MATADD, KOP_NONE);
    // generator for CONV_2D_0_34_fusion
    CNN_GenControl_T gen_ctrl_S22_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S22_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S22_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S22_Conv2d_16x16x1x3_Relu", &gen_ctrl_S22_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 1, 1, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for CONV_2D_0_36_fusion
    CNN_GenControl_T gen_ctrl_S23_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S23_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S23_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S23_Conv2d_16x16x1x3_Relu", &gen_ctrl_S23_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 1, 1, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for ADD_0_37
    CNN_MatAddAct_SQ8("S24_MatAdd_16x1x299", 0, 16, 1, 299, KOP_MATADD, KOP_NONE);
    // generator for ADD_0_38
    CNN_MatAddAct_SQ8("S25_MatAdd_16x1x299", 0, 16, 1, 299, KOP_MATADD, KOP_NONE);
    // generator for CONV_2D_0_40_fusion
    CNN_GenControl_T gen_ctrl_S26_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S26_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S26_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S26_Conv2d_16x16x1x3_Relu", &gen_ctrl_S26_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 2, 2, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for CONV_2D_0_42_fusion
    CNN_GenControl_T gen_ctrl_S27_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S27_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S27_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S27_Conv2d_16x16x1x3_Relu", &gen_ctrl_S27_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 2, 2, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for ADD_0_43
    CNN_MatAddAct_SQ8("S28_MatAdd_16x1x299", 0, 16, 1, 299, KOP_MATADD, KOP_NONE);
    // generator for ADD_0_44
    CNN_MatAddAct_SQ8("S29_MatAdd_16x1x299", 0, 16, 1, 299, KOP_MATADD, KOP_NONE);
    // generator for CONV_2D_0_46_fusion
    CNN_GenControl_T gen_ctrl_S30_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S30_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S30_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S30_Conv2d_16x16x1x3_Relu", &gen_ctrl_S30_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 4, 4, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for CONV_2D_0_48_fusion
    CNN_GenControl_T gen_ctrl_S31_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S31_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S31_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S31_Conv2d_16x16x1x3_Relu", &gen_ctrl_S31_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 4, 4, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for ADD_0_49
    CNN_MatAddAct_SQ8("S32_MatAdd_16x1x299", 0, 16, 1, 299, KOP_MATADD, KOP_NONE);
    // generator for ADD_0_50
    CNN_MatAddAct_SQ8("S33_MatAdd_16x1x299", 0, 16, 1, 299, KOP_MATADD, KOP_NONE);
    // generator for CONV_2D_0_52_fusion
    CNN_GenControl_T gen_ctrl_S34_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S34_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S34_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S34_Conv2d_16x16x1x3_Relu", &gen_ctrl_S34_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 8, 8, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for CONV_2D_0_54_fusion
    CNN_GenControl_T gen_ctrl_S35_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S35_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S35_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S35_Conv2d_16x16x1x3_Relu", &gen_ctrl_S35_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 8, 8, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for ADD_0_55
    CNN_MatAddAct_SQ8("S36_MatAdd_16x1x299", 0, 16, 1, 299, KOP_MATADD, KOP_NONE);
    // generator for ADD_0_56
    CNN_MatAddAct_SQ8("S37_MatAdd_16x1x299", 0, 16, 1, 299, KOP_MATADD, KOP_NONE);
    // generator for CONV_2D_0_58_fusion
    CNN_GenControl_T gen_ctrl_S38_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S38_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S38_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S38_Conv2d_16x16x1x3_Relu", &gen_ctrl_S38_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 16, 16, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for CONV_2D_0_60_fusion
    CNN_GenControl_T gen_ctrl_S39_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S39_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S39_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S39_Conv2d_16x16x1x3_Relu", &gen_ctrl_S39_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 16, 16, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for ADD_0_61
    CNN_MatAddAct_SQ8("S40_MatAdd_16x1x299", 0, 16, 1, 299, KOP_MATADD, KOP_NONE);
    // generator for ADD_0_62
    CNN_MatAddAct_SQ8("S41_MatAdd_16x1x299", 0, 16, 1, 299, KOP_MATADD, KOP_NONE);
    // generator for CONV_2D_0_64_fusion
    CNN_GenControl_T gen_ctrl_S42_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S42_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S42_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S42_Conv2d_16x16x1x3_Relu", &gen_ctrl_S42_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 1, 1, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for CONV_2D_0_66_fusion
    CNN_GenControl_T gen_ctrl_S43_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S43_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S43_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S43_Conv2d_16x16x1x3_Relu", &gen_ctrl_S43_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 1, 1, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for ADD_0_67
    CNN_MatAddAct_SQ8("S44_MatAdd_16x1x299", 0, 16, 1, 299, KOP_MATADD, KOP_NONE);
    // generator for ADD_0_68
    CNN_MatAddAct_SQ8("S45_MatAdd_16x1x299", 0, 16, 1, 299, KOP_MATADD, KOP_NONE);
    // generator for CONV_2D_0_70_fusion
    CNN_GenControl_T gen_ctrl_S46_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S46_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S46_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S46_Conv2d_16x16x1x3_Relu", &gen_ctrl_S46_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 2, 2, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for CONV_2D_0_72_fusion
    CNN_GenControl_T gen_ctrl_S47_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S47_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S47_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S47_Conv2d_16x16x1x3_Relu", &gen_ctrl_S47_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 2, 2, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for ADD_0_73
    CNN_MatAddAct_SQ8("S48_MatAdd_16x1x299", 0, 16, 1, 299, KOP_MATADD, KOP_NONE);
    // generator for ADD_0_74
    CNN_MatAddAct_SQ8("S49_MatAdd_16x1x299", 0, 16, 1, 299, KOP_MATADD, KOP_NONE);
    // generator for CONV_2D_0_76_fusion
    CNN_GenControl_T gen_ctrl_S50_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S50_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S50_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S50_Conv2d_16x16x1x3_Relu", &gen_ctrl_S50_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 4, 4, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for CONV_2D_0_78_fusion
    CNN_GenControl_T gen_ctrl_S51_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S51_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S51_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S51_Conv2d_16x16x1x3_Relu", &gen_ctrl_S51_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 4, 4, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for ADD_0_79
    CNN_MatAddAct_SQ8("S52_MatAdd_16x1x299", 0, 16, 1, 299, KOP_MATADD, KOP_NONE);
    // generator for ADD_0_80
    CNN_MatAddAct_SQ8("S53_MatAdd_16x1x299", 0, 16, 1, 299, KOP_MATADD, KOP_NONE);
    // generator for CONV_2D_0_82_fusion
    CNN_GenControl_T gen_ctrl_S54_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S54_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S54_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S54_Conv2d_16x16x1x3_Relu", &gen_ctrl_S54_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 8, 8, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for CONV_2D_0_84_fusion
    CNN_GenControl_T gen_ctrl_S55_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S55_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S55_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S55_Conv2d_16x16x1x3_Relu", &gen_ctrl_S55_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 8, 8, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for ADD_0_85
    CNN_MatAddAct_SQ8("S56_MatAdd_16x1x299", 0, 16, 1, 299, KOP_MATADD, KOP_NONE);
    // generator for ADD_0_86
    CNN_MatAddAct_SQ8("S57_MatAdd_16x1x299", 0, 16, 1, 299, KOP_MATADD, KOP_NONE);
    // generator for CONV_2D_0_88_fusion
    CNN_GenControl_T gen_ctrl_S58_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S58_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S58_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S58_Conv2d_16x16x1x3_Relu", &gen_ctrl_S58_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 16, 16, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for CONV_2D_0_90_fusion
    CNN_GenControl_T gen_ctrl_S59_Conv2d_16x16x1x3_Relu;
    CNN_InitGenCtrl(&gen_ctrl_S59_Conv2d_16x16x1x3_Relu);
    CNN_SetGenCtrl(&gen_ctrl_S59_Conv2d_16x16x1x3_Relu, "PADTYPE", AT_OPT_VAL(0));
    CNN_ConvolutionPoolAct_SQ8("S59_Conv2d_16x16x1x3_Relu", &gen_ctrl_S59_Conv2d_16x16x1x3_Relu, 4, 1, 16, 16, 299, 1,
        KOP_CONV, 3, 1, 16, 16, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
        KOP_RELU);
    // generator for ADD_0_91
    CNN_MatAddAct_SQ8("S60_MatAdd_16x1x299", 0, 16, 1, 299, KOP_MATADD, KOP_NONE);

#define GRAPH
#ifdef GRAPH
    CreateGraph("quant_modelCNN",
        /* Arguments either passed or globals */
            CArgs(185,
                TCArgInfo("signed char * __restrict__", "Input_1", ARG_SCOPE_ARG, ARG_DIR_IN, AT_MEM_L2, AT_MEM_L2, 0),
                // BiasQ: 0
                TCArgInfo("signed char * __restrict__", "S2_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S2_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S2_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S2_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S2_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S2_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S2_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S2_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S2_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S2_Mul_shift.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S3_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S3_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S3_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S3_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S3_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S3_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S3_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S3_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S3_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S3_Mul_shift.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S4_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S4_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S4_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S4_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S4_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S4_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S4_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S4_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S4_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S4_Mul_shift.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S5_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S5_Infos.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S6_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S6_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S6_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S6_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S6_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S6_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S6_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S6_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S6_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S6_Mul_shift.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S7_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S7_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S7_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S7_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S7_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S7_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S7_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S7_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S7_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S7_Mul_shift.tensor", 1, 1, 1, 0)),
                // In1Scale: 123 In1ScaleN: 6 OutScale: 133 OutScaleN: 8
                TCArgInfo("signed char * __restrict__", "S8_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S8_Infos.tensor", 1, 1, 1, 0)),
                // In1Scale: 135 In1ScaleN: 4 OutScale: 231 OutScaleN: 11
                TCArgInfo("signed char * __restrict__", "S9_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S9_Infos.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S10_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S10_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S10_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S10_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S10_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S10_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S10_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S10_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S10_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S10_Mul_shift.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S11_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S11_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S11_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S11_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S11_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S11_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S11_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S11_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S11_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S11_Mul_shift.tensor", 1, 1, 1, 0)),
                // In1Scale: 145 In1ScaleN: 6 OutScale: 225 OutScaleN: 9
                TCArgInfo("signed char * __restrict__", "S12_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S12_Infos.tensor", 1, 1, 1, 0)),
                // In1Scale: 249 In1ScaleN: 6 OutScale: 131 OutScaleN: 9
                TCArgInfo("signed char * __restrict__", "S13_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S13_Infos.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S14_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S14_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S14_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S14_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S14_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S14_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S14_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S14_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S14_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S14_Mul_shift.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S15_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S15_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S15_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S15_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S15_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S15_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S15_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S15_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S15_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S15_Mul_shift.tensor", 1, 1, 1, 0)),
                // In1Scale: 41 In1ScaleN: 5 OutScale: 99 OutScaleN: 7
                TCArgInfo("signed char * __restrict__", "S16_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S16_Infos.tensor", 1, 1, 1, 0)),
                // In1Scale: 159 In1ScaleN: 5 OutScale: 3 OutScaleN: 4
                TCArgInfo("signed char * __restrict__", "S17_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S17_Infos.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S18_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S18_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S18_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S18_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S18_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S18_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S18_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S18_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S18_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S18_Mul_shift.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S19_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S19_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S19_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S19_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S19_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S19_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S19_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S19_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S19_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S19_Mul_shift.tensor", 1, 1, 1, 0)),
                // In1Scale: 133 In1ScaleN: 6 OutScale: 237 OutScaleN: 9
                TCArgInfo("signed char * __restrict__", "S20_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S20_Infos.tensor", 1, 1, 1, 0)),
                // In1Scale: 17 In1ScaleN: 1 OutScale: 239 OutScaleN: 11
                TCArgInfo("signed char * __restrict__", "S21_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S21_Infos.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S22_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S22_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S22_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S22_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S22_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S22_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S22_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S22_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S22_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S22_Mul_shift.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S23_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S23_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S23_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S23_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S23_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S23_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S23_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S23_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S23_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S23_Mul_shift.tensor", 1, 1, 1, 0)),
                // In1Scale: 37 In1ScaleN: 4 OutScale: 99 OutScaleN: 8
                TCArgInfo("signed char * __restrict__", "S24_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S24_Infos.tensor", 1, 1, 1, 0)),
                // In1Scale: 73 In1ScaleN: 3 OutScale: 219 OutScaleN: 11
                TCArgInfo("signed char * __restrict__", "S25_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S25_Infos.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S26_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S26_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S26_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S26_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S26_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S26_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S26_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S26_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S26_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S26_Mul_shift.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S27_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S27_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S27_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S27_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S27_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S27_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S27_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S27_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S27_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S27_Mul_shift.tensor", 1, 1, 1, 0)),
                // In1Scale: 137 In1ScaleN: 5 OutScale: 113 OutScaleN: 9
                TCArgInfo("signed char * __restrict__", "S28_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S28_Infos.tensor", 1, 1, 1, 0)),
                // In1Scale: 31 In1ScaleN: 1 OutScale: 33 OutScaleN: 9
                TCArgInfo("signed char * __restrict__", "S29_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S29_Infos.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S30_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S30_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S30_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S30_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S30_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S30_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S30_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S30_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S30_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S30_Mul_shift.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S31_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S31_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S31_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S31_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S31_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S31_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S31_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S31_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S31_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S31_Mul_shift.tensor", 1, 1, 1, 0)),
                // In1Scale: 15 In1ScaleN: 2 OutScale: 133 OutScaleN: 9
                TCArgInfo("signed char * __restrict__", "S32_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S32_Infos.tensor", 1, 1, 1, 0)),
                // In1Scale: 205 In1ScaleN: 4 OutScale: 5 OutScaleN: 6
                TCArgInfo("signed char * __restrict__", "S33_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S33_Infos.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S34_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S34_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S34_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S34_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S34_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S34_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S34_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S34_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S34_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S34_Mul_shift.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S35_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S35_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S35_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S35_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S35_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S35_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S35_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S35_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S35_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S35_Mul_shift.tensor", 1, 1, 1, 0)),
                // In1Scale: 197 In1ScaleN: 6 OutScale: 167 OutScaleN: 9
                TCArgInfo("signed char * __restrict__", "S36_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S36_Infos.tensor", 1, 1, 1, 0)),
                // In1Scale: 41 In1ScaleN: 2 OutScale: 199 OutScaleN: 11
                TCArgInfo("signed char * __restrict__", "S37_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S37_Infos.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S38_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S38_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S38_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S38_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S38_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S38_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S38_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S38_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S38_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S38_Mul_shift.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S39_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S39_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S39_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S39_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S39_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S39_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S39_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S39_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S39_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S39_Mul_shift.tensor", 1, 1, 1, 0)),
                // In1Scale: 57 In1ScaleN: 5 OutScale: 237 OutScaleN: 9
                TCArgInfo("signed char * __restrict__", "S40_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S40_Infos.tensor", 1, 1, 1, 0)),
                // In1Scale: 95 In1ScaleN: 4 OutScale: 167 OutScaleN: 10
                TCArgInfo("signed char * __restrict__", "S41_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S41_Infos.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S42_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S42_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S42_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S42_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S42_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S42_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S42_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S42_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S42_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S42_Mul_shift.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S43_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S43_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S43_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S43_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S43_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S43_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S43_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S43_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S43_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S43_Mul_shift.tensor", 1, 1, 1, 0)),
                // In1Scale: 115 In1ScaleN: 5 OutScale: 59 OutScaleN: 8
                TCArgInfo("signed char * __restrict__", "S44_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S44_Infos.tensor", 1, 1, 1, 0)),
                // In1Scale: 163 In1ScaleN: 4 OutScale: 201 OutScaleN: 11
                TCArgInfo("signed char * __restrict__", "S45_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S45_Infos.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S46_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S46_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S46_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S46_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S46_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S46_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S46_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S46_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S46_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S46_Mul_shift.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S47_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S47_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S47_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S47_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S47_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S47_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S47_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S47_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S47_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S47_Mul_shift.tensor", 1, 1, 1, 0)),
                // In1Scale: 137 In1ScaleN: 4 OutScale: 109 OutScaleN: 10
                TCArgInfo("signed char * __restrict__", "S48_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S48_Infos.tensor", 1, 1, 1, 0)),
                // In1Scale: 161 In1ScaleN: 3 OutScale: 25 OutScaleN: 9
                TCArgInfo("signed char * __restrict__", "S49_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S49_Infos.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S50_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S50_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S50_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S50_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S50_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S50_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S50_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S50_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S50_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S50_Mul_shift.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S51_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S51_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S51_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S51_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S51_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S51_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S51_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S51_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S51_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S51_Mul_shift.tensor", 1, 1, 1, 0)),
                // In1Scale: 179 In1ScaleN: 5 OutScale: 175 OutScaleN: 10
                TCArgInfo("signed char * __restrict__", "S52_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S52_Infos.tensor", 1, 1, 1, 0)),
                // In1Scale: 195 In1ScaleN: 4 OutScale: 41 OutScaleN: 9
                TCArgInfo("signed char * __restrict__", "S53_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S53_Infos.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S54_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S54_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S54_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S54_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S54_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S54_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S54_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S54_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S54_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S54_Mul_shift.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S55_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S55_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S55_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S55_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S55_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S55_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S55_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S55_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S55_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S55_Mul_shift.tensor", 1, 1, 1, 0)),
                // In1Scale: 239 In1ScaleN: 6 OutScale: 137 OutScaleN: 9
                TCArgInfo("signed char * __restrict__", "S56_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S56_Infos.tensor", 1, 1, 1, 0)),
                // In1Scale: 255 In1ScaleN: 5 OutScale: 1 OutScaleN: 3
                TCArgInfo("signed char * __restrict__", "S57_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S57_Infos.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S58_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S58_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S58_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S58_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S58_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S58_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S58_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S58_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S58_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S58_Mul_shift.tensor", 1, 1, 1, 0)),
                // all 0
                TCArgInfo("signed char * __restrict__", "S59_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S59_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S59_Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S59_Weights.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed int * __restrict__", "S59_Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S59_Biases.tensor", 1, 1, 4, 0)),
                TCArgInfo("unsigned char * __restrict__", "S59_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S59_Mul_scale.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "S59_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S59_Mul_shift.tensor", 1, 1, 1, 0)),
                // In1Scale: 125 In1ScaleN: 5 OutScale: 119 OutScaleN: 9
                TCArgInfo("signed char * __restrict__", "S60_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/S60_Infos.tensor", 1, 1, 1, 0)),
                TCArgInfo("signed char * __restrict__", "Output_1", ARG_SCOPE_ARG, ARG_DIR_OUT, AT_MEM_L2, AT_MEM_L2, 0)
            ),
        /* Locals, allocated dynamically */
        CArgs(58,
            TCArgInfo("signed char * __restrict__", "S2_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S3_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S4_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S5_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S6_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S7_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S8_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S9_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S10_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S11_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S12_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S13_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S14_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S15_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S16_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S17_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S18_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S19_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S20_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S21_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S22_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S23_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S24_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S25_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S26_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S27_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S28_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S29_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S30_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S31_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S32_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S33_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S34_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S35_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S36_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S37_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S38_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S39_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S40_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S41_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S42_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S43_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S44_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S45_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S46_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S47_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S48_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S49_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S50_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S51_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S52_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S53_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S54_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S55_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S56_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S57_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S58_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S59_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0)
        )
    );

    /* Stacked tensors - Concats */
    // no concats in graph so not stacked tensors created

    // Node S2_Conv2d_16x20x1x3 inq -22.48<i8*0.17558888<22.30 weightsq chan<i8*chan<chan outq -28.26<i8*0.22078258<28.04 biasesq i32*chan
    AddNode("S2_Conv2d_16x20x1x3", Bindings(7, GNodeArg(GNA_IN, "Input_1", 0), GNodeArg(GNA_IN, "S2_Weights", 0), GNodeArg(GNA_IN, "S2_Biases", 0), GNodeArg(GNA_OUT, "S2_Output", 0), GNodeArg(GNA_IN, "S2_Mul_scale", 0), GNodeArg(GNA_IN, "S2_Mul_shift", 0), GNodeArg(GNA_IN, "S2_Infos", 0)));
    // Node S3_Conv2d_16x16x1x3_Relu inq -28.26<i8*0.22078258<28.04 weightsq chan<i8*chan<chan outq -2.75<i8*0.02148511<2.73 biasesq i32*chan
    AddNode("S3_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S2_Output", 0), GNodeArg(GNA_IN, "S3_Weights", 0), GNodeArg(GNA_IN, "S3_Biases", 0), GNodeArg(GNA_OUT, "S3_Output", 0), GNodeArg(GNA_IN, "S3_Mul_scale", 0), GNodeArg(GNA_IN, "S3_Mul_shift", 0), GNodeArg(GNA_IN, "S3_Infos", 0)));
    // Node S4_Conv2d_16x16x1x3_Relu inq -2.75<i8*0.02148511<2.73 weightsq chan<i8*chan<chan outq -1.83<i8*0.01428443<1.81 biasesq i32*chan
    AddNode("S4_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S3_Output", 0), GNodeArg(GNA_IN, "S4_Weights", 0), GNodeArg(GNA_IN, "S4_Biases", 0), GNodeArg(GNA_OUT, "S4_Output", 0), GNodeArg(GNA_IN, "S4_Mul_scale", 0), GNodeArg(GNA_IN, "S4_Mul_shift", 0), GNodeArg(GNA_IN, "S4_Infos", 0)));
    // Node S5_MatAdd_16x1x299_Relu in1q -28.26<i8*0.22078258<28.04 in2q -1.83<i8*0.01428443<1.81 outq -29.72<i8*0.23222047<29.49
    AddNode("S5_MatAdd_16x1x299_Relu", Bindings(4, GNodeArg(GNA_IN, "S2_Output", 0), GNodeArg(GNA_IN, "S4_Output", 0), GNodeArg(GNA_OUT, "S5_Output", 0), GNodeArg(GNA_IN, "S5_Infos", 0)));
    // Node S6_Conv2d_16x16x1x3_Relu inq -29.72<i8*0.23222047<29.49 weightsq chan<i8*chan<chan outq -2.15<i8*0.01677836<2.13 biasesq i32*chan
    AddNode("S6_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S5_Output", 0), GNodeArg(GNA_IN, "S6_Weights", 0), GNodeArg(GNA_IN, "S6_Biases", 0), GNodeArg(GNA_OUT, "S6_Output", 0), GNodeArg(GNA_IN, "S6_Mul_scale", 0), GNodeArg(GNA_IN, "S6_Mul_shift", 0), GNodeArg(GNA_IN, "S6_Infos", 0)));
    // Node S7_Conv2d_16x16x1x3_Relu inq -2.15<i8*0.01677836<2.13 weightsq chan<i8*chan<chan outq -3.52<i8*0.02749346<3.49 biasesq i32*chan
    AddNode("S7_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S6_Output", 0), GNodeArg(GNA_IN, "S7_Weights", 0), GNodeArg(GNA_IN, "S7_Biases", 0), GNodeArg(GNA_OUT, "S7_Output", 0), GNodeArg(GNA_IN, "S7_Mul_scale", 0), GNodeArg(GNA_IN, "S7_Mul_shift", 0), GNodeArg(GNA_IN, "S7_Infos", 0)));
    // Node S8_MatAdd_16x1x299 in1q -3.52<i8*0.02749346<3.49 in2q -1.83<i8*0.01428443<1.81 outq -3.52<i8*0.02749346<3.49
    AddNode("S8_MatAdd_16x1x299", Bindings(4, GNodeArg(GNA_IN, "S7_Output", 0), GNodeArg(GNA_IN, "S4_Output", 0), GNodeArg(GNA_OUT, "S8_Output", 0), GNodeArg(GNA_IN, "S8_Infos", 0)));
    // Node S9_MatAdd_16x1x299 in1q -29.72<i8*0.23222047<29.49 in2q -3.52<i8*0.02749346<3.49 outq -31.14<i8*0.24328883<30.90
    AddNode("S9_MatAdd_16x1x299", Bindings(4, GNodeArg(GNA_IN, "S5_Output", 0), GNodeArg(GNA_IN, "S7_Output", 0), GNodeArg(GNA_OUT, "S9_Output", 0), GNodeArg(GNA_IN, "S9_Infos", 0)));
    // Node S10_Conv2d_16x16x1x3_Relu inq -31.14<i8*0.24328883<30.90 weightsq chan<i8*chan<chan outq -4.38<i8*0.03418038<4.34 biasesq i32*chan
    AddNode("S10_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S9_Output", 0), GNodeArg(GNA_IN, "S10_Weights", 0), GNodeArg(GNA_IN, "S10_Biases", 0), GNodeArg(GNA_OUT, "S10_Output", 0), GNodeArg(GNA_IN, "S10_Mul_scale", 0), GNodeArg(GNA_IN, "S10_Mul_shift", 0), GNodeArg(GNA_IN, "S10_Infos", 0)));
    // Node S11_Conv2d_16x16x1x3_Relu inq -4.38<i8*0.03418038<4.34 weightsq chan<i8*chan<chan outq -7.99<i8*0.06244475<7.93 biasesq i32*chan
    AddNode("S11_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S10_Output", 0), GNodeArg(GNA_IN, "S11_Weights", 0), GNodeArg(GNA_IN, "S11_Biases", 0), GNodeArg(GNA_OUT, "S11_Output", 0), GNodeArg(GNA_IN, "S11_Mul_scale", 0), GNodeArg(GNA_IN, "S11_Mul_shift", 0), GNodeArg(GNA_IN, "S11_Infos", 0)));
    // Node S12_MatAdd_16x1x299 in1q -7.99<i8*0.06244475<7.93 in2q -3.52<i8*0.02749346<3.49 outq -7.99<i8*0.06244475<7.93
    AddNode("S12_MatAdd_16x1x299", Bindings(4, GNodeArg(GNA_IN, "S11_Output", 0), GNodeArg(GNA_IN, "S8_Output", 0), GNodeArg(GNA_OUT, "S12_Output", 0), GNodeArg(GNA_IN, "S12_Infos", 0)));
    // Node S13_MatAdd_16x1x299 in1q -31.14<i8*0.24328883<30.90 in2q -7.99<i8*0.06244475<7.93 outq -31.14<i8*0.24328883<30.90
    AddNode("S13_MatAdd_16x1x299", Bindings(4, GNodeArg(GNA_IN, "S9_Output", 0), GNodeArg(GNA_IN, "S11_Output", 0), GNodeArg(GNA_OUT, "S13_Output", 0), GNodeArg(GNA_IN, "S13_Infos", 0)));
    // Node S14_Conv2d_16x16x1x3_Relu inq -31.14<i8*0.24328883<30.90 weightsq chan<i8*chan<chan outq -3.36<i8*0.02628114<3.34 biasesq i32*chan
    AddNode("S14_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S13_Output", 0), GNodeArg(GNA_IN, "S14_Weights", 0), GNodeArg(GNA_IN, "S14_Biases", 0), GNodeArg(GNA_OUT, "S14_Output", 0), GNodeArg(GNA_IN, "S14_Mul_scale", 0), GNodeArg(GNA_IN, "S14_Mul_shift", 0), GNodeArg(GNA_IN, "S14_Infos", 0)));
    // Node S15_Conv2d_16x16x1x3_Relu inq -3.36<i8*0.02628114<3.34 weightsq chan<i8*chan<chan outq -6.25<i8*0.04883727<6.20 biasesq i32*chan
    AddNode("S15_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S14_Output", 0), GNodeArg(GNA_IN, "S15_Weights", 0), GNodeArg(GNA_IN, "S15_Biases", 0), GNodeArg(GNA_OUT, "S15_Output", 0), GNodeArg(GNA_IN, "S15_Mul_scale", 0), GNodeArg(GNA_IN, "S15_Mul_shift", 0), GNodeArg(GNA_IN, "S15_Infos", 0)));
    // Node S16_MatAdd_16x1x299 in1q -7.99<i8*0.06244475<7.93 in2q -6.25<i8*0.04883727<6.20 outq -8.10<i8*0.06328034<8.04
    AddNode("S16_MatAdd_16x1x299", Bindings(4, GNodeArg(GNA_IN, "S12_Output", 0), GNodeArg(GNA_IN, "S15_Output", 0), GNodeArg(GNA_OUT, "S16_Output", 0), GNodeArg(GNA_IN, "S16_Infos", 0)));
    // Node S17_MatAdd_16x1x299 in1q -31.14<i8*0.24328883<30.90 in2q -6.25<i8*0.04883727<6.20 outq -33.31<i8*0.26024291<33.05
    AddNode("S17_MatAdd_16x1x299", Bindings(4, GNodeArg(GNA_IN, "S13_Output", 0), GNodeArg(GNA_IN, "S15_Output", 0), GNodeArg(GNA_OUT, "S17_Output", 0), GNodeArg(GNA_IN, "S17_Infos", 0)));
    // Node S18_Conv2d_16x16x1x3_Relu inq -33.31<i8*0.26024291<33.05 weightsq chan<i8*chan<chan outq -3.41<i8*0.02660332<3.38 biasesq i32*chan
    AddNode("S18_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S17_Output", 0), GNodeArg(GNA_IN, "S18_Weights", 0), GNodeArg(GNA_IN, "S18_Biases", 0), GNodeArg(GNA_OUT, "S18_Output", 0), GNodeArg(GNA_IN, "S18_Mul_scale", 0), GNodeArg(GNA_IN, "S18_Mul_shift", 0), GNodeArg(GNA_IN, "S18_Infos", 0)));
    // Node S19_Conv2d_16x16x1x3_Relu inq -3.41<i8*0.02660332<3.38 weightsq chan<i8*chan<chan outq -3.91<i8*0.03053966<3.88 biasesq i32*chan
    AddNode("S19_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S18_Output", 0), GNodeArg(GNA_IN, "S19_Weights", 0), GNodeArg(GNA_IN, "S19_Biases", 0), GNodeArg(GNA_OUT, "S19_Output", 0), GNodeArg(GNA_IN, "S19_Mul_scale", 0), GNodeArg(GNA_IN, "S19_Mul_shift", 0), GNodeArg(GNA_IN, "S19_Infos", 0)));
    // Node S20_MatAdd_16x1x299 in1q -8.10<i8*0.06328034<8.04 in2q -3.91<i8*0.03053966<3.88 outq -8.44<i8*0.06596538<8.38
    AddNode("S20_MatAdd_16x1x299", Bindings(4, GNodeArg(GNA_IN, "S16_Output", 0), GNodeArg(GNA_IN, "S19_Output", 0), GNodeArg(GNA_OUT, "S20_Output", 0), GNodeArg(GNA_IN, "S20_Infos", 0)));
    // Node S21_MatAdd_16x1x299 in1q -33.31<i8*0.26024291<33.05 in2q -3.91<i8*0.03053966<3.88 outq -33.45<i8*0.26133761<33.19
    AddNode("S21_MatAdd_16x1x299", Bindings(4, GNodeArg(GNA_IN, "S17_Output", 0), GNodeArg(GNA_IN, "S19_Output", 0), GNodeArg(GNA_OUT, "S21_Output", 0), GNodeArg(GNA_IN, "S21_Infos", 0)));
    // Node S22_Conv2d_16x16x1x3_Relu inq -33.45<i8*0.26133761<33.19 weightsq chan<i8*chan<chan outq -2.73<i8*0.02131508<2.71 biasesq i32*chan
    AddNode("S22_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S21_Output", 0), GNodeArg(GNA_IN, "S22_Weights", 0), GNodeArg(GNA_IN, "S22_Biases", 0), GNodeArg(GNA_OUT, "S22_Output", 0), GNodeArg(GNA_IN, "S22_Mul_scale", 0), GNodeArg(GNA_IN, "S22_Mul_shift", 0), GNodeArg(GNA_IN, "S22_Infos", 0)));
    // Node S23_Conv2d_16x16x1x3_Relu inq -2.73<i8*0.02131508<2.71 weightsq chan<i8*chan<chan outq -3.66<i8*0.02856687<3.63 biasesq i32*chan
    AddNode("S23_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S22_Output", 0), GNodeArg(GNA_IN, "S23_Weights", 0), GNodeArg(GNA_IN, "S23_Biases", 0), GNodeArg(GNA_OUT, "S23_Output", 0), GNodeArg(GNA_IN, "S23_Mul_scale", 0), GNodeArg(GNA_IN, "S23_Mul_shift", 0), GNodeArg(GNA_IN, "S23_Infos", 0)));
    // Node S24_MatAdd_16x1x299 in1q -8.44<i8*0.06596538<8.38 in2q -3.66<i8*0.02856687<3.63 outq -9.43<i8*0.07370432<9.36
    AddNode("S24_MatAdd_16x1x299", Bindings(4, GNodeArg(GNA_IN, "S20_Output", 0), GNodeArg(GNA_IN, "S23_Output", 0), GNodeArg(GNA_OUT, "S24_Output", 0), GNodeArg(GNA_IN, "S24_Infos", 0)));
    // Node S25_MatAdd_16x1x299 in1q -33.45<i8*0.26133761<33.19 in2q -3.66<i8*0.02856687<3.63 outq -34.16<i8*0.26686081<33.89
    AddNode("S25_MatAdd_16x1x299", Bindings(4, GNodeArg(GNA_IN, "S21_Output", 0), GNodeArg(GNA_IN, "S23_Output", 0), GNodeArg(GNA_OUT, "S25_Output", 0), GNodeArg(GNA_IN, "S25_Infos", 0)));
    // Node S26_Conv2d_16x16x1x3_Relu inq -34.16<i8*0.26686081<33.89 weightsq chan<i8*chan<chan outq -2.35<i8*0.01835034<2.33 biasesq i32*chan
    AddNode("S26_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S25_Output", 0), GNodeArg(GNA_IN, "S26_Weights", 0), GNodeArg(GNA_IN, "S26_Biases", 0), GNodeArg(GNA_OUT, "S26_Output", 0), GNodeArg(GNA_IN, "S26_Mul_scale", 0), GNodeArg(GNA_IN, "S26_Mul_shift", 0), GNodeArg(GNA_IN, "S26_Infos", 0)));
    // Node S27_Conv2d_16x16x1x3_Relu inq -2.35<i8*0.01835034<2.33 weightsq chan<i8*chan<chan outq -2.20<i8*0.01719413<2.18 biasesq i32*chan
    AddNode("S27_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S26_Output", 0), GNodeArg(GNA_IN, "S27_Weights", 0), GNodeArg(GNA_IN, "S27_Biases", 0), GNodeArg(GNA_OUT, "S27_Output", 0), GNodeArg(GNA_IN, "S27_Mul_scale", 0), GNodeArg(GNA_IN, "S27_Mul_shift", 0), GNodeArg(GNA_IN, "S27_Infos", 0)));
    // Node S28_MatAdd_16x1x299 in1q -9.43<i8*0.07370432<9.36 in2q -2.20<i8*0.01719413<2.18 outq -9.97<i8*0.07791241<9.89
    AddNode("S28_MatAdd_16x1x299", Bindings(4, GNodeArg(GNA_IN, "S24_Output", 0), GNodeArg(GNA_IN, "S27_Output", 0), GNodeArg(GNA_OUT, "S28_Output", 0), GNodeArg(GNA_IN, "S28_Infos", 0)));
    // Node S29_MatAdd_16x1x299 in1q -34.16<i8*0.26686081<33.89 in2q -2.20<i8*0.01719413<2.18 outq -34.16<i8*0.26686081<33.89
    AddNode("S29_MatAdd_16x1x299", Bindings(4, GNodeArg(GNA_IN, "S25_Output", 0), GNodeArg(GNA_IN, "S27_Output", 0), GNodeArg(GNA_OUT, "S29_Output", 0), GNodeArg(GNA_IN, "S29_Infos", 0)));
    // Node S30_Conv2d_16x16x1x3_Relu inq -34.16<i8*0.26686081<33.89 weightsq chan<i8*chan<chan outq -2.83<i8*0.02213331<2.81 biasesq i32*chan
    AddNode("S30_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S29_Output", 0), GNodeArg(GNA_IN, "S30_Weights", 0), GNodeArg(GNA_IN, "S30_Biases", 0), GNodeArg(GNA_OUT, "S30_Output", 0), GNodeArg(GNA_IN, "S30_Mul_scale", 0), GNodeArg(GNA_IN, "S30_Mul_shift", 0), GNodeArg(GNA_IN, "S30_Infos", 0)));
    // Node S31_Conv2d_16x16x1x3_Relu inq -2.83<i8*0.02213331<2.81 weightsq chan<i8*chan<chan outq -2.66<i8*0.02079234<2.64 biasesq i32*chan
    AddNode("S31_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S30_Output", 0), GNodeArg(GNA_IN, "S31_Weights", 0), GNodeArg(GNA_IN, "S31_Biases", 0), GNodeArg(GNA_OUT, "S31_Output", 0), GNodeArg(GNA_IN, "S31_Mul_scale", 0), GNodeArg(GNA_IN, "S31_Mul_shift", 0), GNodeArg(GNA_IN, "S31_Infos", 0)));
    // Node S32_MatAdd_16x1x299 in1q -9.97<i8*0.07791241<9.89 in2q -2.66<i8*0.02079234<2.64 outq -10.21<i8*0.07980008<10.13
    AddNode("S32_MatAdd_16x1x299", Bindings(4, GNodeArg(GNA_IN, "S28_Output", 0), GNodeArg(GNA_IN, "S31_Output", 0), GNodeArg(GNA_OUT, "S32_Output", 0), GNodeArg(GNA_IN, "S32_Infos", 0)));
    // Node S33_MatAdd_16x1x299 in1q -34.16<i8*0.26686081<33.89 in2q -2.66<i8*0.02079234<2.64 outq -34.16<i8*0.26686081<33.89
    AddNode("S33_MatAdd_16x1x299", Bindings(4, GNodeArg(GNA_IN, "S29_Output", 0), GNodeArg(GNA_IN, "S31_Output", 0), GNodeArg(GNA_OUT, "S33_Output", 0), GNodeArg(GNA_IN, "S33_Infos", 0)));
    // Node S34_Conv2d_16x16x1x3_Relu inq -34.16<i8*0.26686081<33.89 weightsq chan<i8*chan<chan outq -3.07<i8*0.02398712<3.05 biasesq i32*chan
    AddNode("S34_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S33_Output", 0), GNodeArg(GNA_IN, "S34_Weights", 0), GNodeArg(GNA_IN, "S34_Biases", 0), GNodeArg(GNA_OUT, "S34_Output", 0), GNodeArg(GNA_IN, "S34_Mul_scale", 0), GNodeArg(GNA_IN, "S34_Mul_shift", 0), GNodeArg(GNA_IN, "S34_Infos", 0)));
    // Node S35_Conv2d_16x16x1x3_Relu inq -3.07<i8*0.02398712<3.05 weightsq chan<i8*chan<chan outq -3.32<i8*0.02596876<3.30 biasesq i32*chan
    AddNode("S35_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S34_Output", 0), GNodeArg(GNA_IN, "S35_Weights", 0), GNodeArg(GNA_IN, "S35_Biases", 0), GNodeArg(GNA_OUT, "S35_Output", 0), GNodeArg(GNA_IN, "S35_Mul_scale", 0), GNodeArg(GNA_IN, "S35_Mul_shift", 0), GNodeArg(GNA_IN, "S35_Infos", 0)));
    // Node S36_MatAdd_16x1x299 in1q -10.21<i8*0.07980008<10.13 in2q -3.32<i8*0.02596876<3.30 outq -10.21<i8*0.07980008<10.13
    AddNode("S36_MatAdd_16x1x299", Bindings(4, GNodeArg(GNA_IN, "S32_Output", 0), GNodeArg(GNA_IN, "S35_Output", 0), GNodeArg(GNA_OUT, "S36_Output", 0), GNodeArg(GNA_IN, "S36_Infos", 0)));
    // Node S37_MatAdd_16x1x299 in1q -34.16<i8*0.26686081<33.89 in2q -3.32<i8*0.02596876<3.30 outq -34.16<i8*0.26686081<33.89
    AddNode("S37_MatAdd_16x1x299", Bindings(4, GNodeArg(GNA_IN, "S33_Output", 0), GNodeArg(GNA_IN, "S35_Output", 0), GNodeArg(GNA_OUT, "S37_Output", 0), GNodeArg(GNA_IN, "S37_Infos", 0)));
    // Node S38_Conv2d_16x16x1x3_Relu inq -34.16<i8*0.26686081<33.89 weightsq chan<i8*chan<chan outq -3.32<i8*0.02594501<3.30 biasesq i32*chan
    AddNode("S38_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S37_Output", 0), GNodeArg(GNA_IN, "S38_Weights", 0), GNodeArg(GNA_IN, "S38_Biases", 0), GNodeArg(GNA_OUT, "S38_Output", 0), GNodeArg(GNA_IN, "S38_Mul_scale", 0), GNodeArg(GNA_IN, "S38_Mul_shift", 0), GNodeArg(GNA_IN, "S38_Infos", 0)));
    // Node S39_Conv2d_16x16x1x3_Relu inq -3.32<i8*0.02594501<3.30 weightsq chan<i8*chan<chan outq -5.74<i8*0.04486069<5.70 biasesq i32*chan
    AddNode("S39_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S38_Output", 0), GNodeArg(GNA_IN, "S39_Weights", 0), GNodeArg(GNA_IN, "S39_Biases", 0), GNodeArg(GNA_OUT, "S39_Output", 0), GNodeArg(GNA_IN, "S39_Mul_scale", 0), GNodeArg(GNA_IN, "S39_Mul_shift", 0), GNodeArg(GNA_IN, "S39_Infos", 0)));
    // Node S40_MatAdd_16x1x299 in1q -10.21<i8*0.07980008<10.13 in2q -5.74<i8*0.04486069<5.70 outq -12.38<i8*0.09673376<12.29
    AddNode("S40_MatAdd_16x1x299", Bindings(4, GNodeArg(GNA_IN, "S36_Output", 0), GNodeArg(GNA_IN, "S39_Output", 0), GNodeArg(GNA_OUT, "S40_Output", 0), GNodeArg(GNA_IN, "S40_Infos", 0)));
    // Node S41_MatAdd_16x1x299 in1q -34.16<i8*0.26686081<33.89 in2q -5.74<i8*0.04486069<5.70 outq -35.13<i8*0.27444515<34.85
    AddNode("S41_MatAdd_16x1x299", Bindings(4, GNodeArg(GNA_IN, "S37_Output", 0), GNodeArg(GNA_IN, "S39_Output", 0), GNodeArg(GNA_OUT, "S41_Output", 0), GNodeArg(GNA_IN, "S41_Infos", 0)));
    // Node S42_Conv2d_16x16x1x3_Relu inq -35.13<i8*0.27444515<34.85 weightsq chan<i8*chan<chan outq -3.36<i8*0.02622865<3.33 biasesq i32*chan
    AddNode("S42_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S41_Output", 0), GNodeArg(GNA_IN, "S42_Weights", 0), GNodeArg(GNA_IN, "S42_Biases", 0), GNodeArg(GNA_OUT, "S42_Output", 0), GNodeArg(GNA_IN, "S42_Mul_scale", 0), GNodeArg(GNA_IN, "S42_Mul_shift", 0), GNodeArg(GNA_IN, "S42_Infos", 0)));
    // Node S43_Conv2d_16x16x1x3_Relu inq -3.36<i8*0.02622865<3.33 weightsq chan<i8*chan<chan outq -3.45<i8*0.02693235<3.42 biasesq i32*chan
    AddNode("S43_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S42_Output", 0), GNodeArg(GNA_IN, "S43_Weights", 0), GNodeArg(GNA_IN, "S43_Biases", 0), GNodeArg(GNA_OUT, "S43_Output", 0), GNodeArg(GNA_IN, "S43_Mul_scale", 0), GNodeArg(GNA_IN, "S43_Mul_shift", 0), GNodeArg(GNA_IN, "S43_Infos", 0)));
    // Node S44_MatAdd_16x1x299 in1q -12.38<i8*0.09673376<12.29 in2q -3.45<i8*0.02693235<3.42 outq -14.95<i8*0.11676898<14.83
    AddNode("S44_MatAdd_16x1x299", Bindings(4, GNodeArg(GNA_IN, "S40_Output", 0), GNodeArg(GNA_IN, "S43_Output", 0), GNodeArg(GNA_OUT, "S44_Output", 0), GNodeArg(GNA_IN, "S44_Infos", 0)));
    // Node S45_MatAdd_16x1x299 in1q -35.13<i8*0.27444515<34.85 in2q -3.45<i8*0.02693235<3.42 outq -35.13<i8*0.27444515<34.85
    AddNode("S45_MatAdd_16x1x299", Bindings(4, GNodeArg(GNA_IN, "S41_Output", 0), GNodeArg(GNA_IN, "S43_Output", 0), GNodeArg(GNA_OUT, "S45_Output", 0), GNodeArg(GNA_IN, "S45_Infos", 0)));
    // Node S46_Conv2d_16x16x1x3_Relu inq -35.13<i8*0.27444515<34.85 weightsq chan<i8*chan<chan outq -2.85<i8*0.02224313<2.82 biasesq i32*chan
    AddNode("S46_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S45_Output", 0), GNodeArg(GNA_IN, "S46_Weights", 0), GNodeArg(GNA_IN, "S46_Biases", 0), GNodeArg(GNA_OUT, "S46_Output", 0), GNodeArg(GNA_IN, "S46_Mul_scale", 0), GNodeArg(GNA_IN, "S46_Mul_shift", 0), GNodeArg(GNA_IN, "S46_Infos", 0)));
    // Node S47_Conv2d_16x16x1x3_Relu inq -2.85<i8*0.02224313<2.82 weightsq chan<i8*chan<chan outq -1.75<i8*0.01365883<1.73 biasesq i32*chan
    AddNode("S47_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S46_Output", 0), GNodeArg(GNA_IN, "S47_Weights", 0), GNodeArg(GNA_IN, "S47_Biases", 0), GNodeArg(GNA_OUT, "S47_Output", 0), GNodeArg(GNA_IN, "S47_Mul_scale", 0), GNodeArg(GNA_IN, "S47_Mul_shift", 0), GNodeArg(GNA_IN, "S47_Infos", 0)));
    // Node S48_MatAdd_16x1x299 in1q -14.95<i8*0.11676898<14.83 in2q -1.75<i8*0.01365883<1.73 outq -16.44<i8*0.12847272<16.32
    AddNode("S48_MatAdd_16x1x299", Bindings(4, GNodeArg(GNA_IN, "S44_Output", 0), GNodeArg(GNA_IN, "S47_Output", 0), GNodeArg(GNA_OUT, "S48_Output", 0), GNodeArg(GNA_IN, "S48_Infos", 0)));
    // Node S49_MatAdd_16x1x299 in1q -35.13<i8*0.27444515<34.85 in2q -1.75<i8*0.01365883<1.73 outq -35.84<i8*0.28000027<35.56
    AddNode("S49_MatAdd_16x1x299", Bindings(4, GNodeArg(GNA_IN, "S45_Output", 0), GNodeArg(GNA_IN, "S47_Output", 0), GNodeArg(GNA_OUT, "S49_Output", 0), GNodeArg(GNA_IN, "S49_Infos", 0)));
    // Node S50_Conv2d_16x16x1x3_Relu inq -35.84<i8*0.28000027<35.56 weightsq chan<i8*chan<chan outq -2.71<i8*0.02117695<2.69 biasesq i32*chan
    AddNode("S50_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S49_Output", 0), GNodeArg(GNA_IN, "S50_Weights", 0), GNodeArg(GNA_IN, "S50_Biases", 0), GNodeArg(GNA_OUT, "S50_Output", 0), GNodeArg(GNA_IN, "S50_Mul_scale", 0), GNodeArg(GNA_IN, "S50_Mul_shift", 0), GNodeArg(GNA_IN, "S50_Infos", 0)));
    // Node S51_Conv2d_16x16x1x3_Relu inq -2.71<i8*0.02117695<2.69 weightsq chan<i8*chan<chan outq -2.94<i8*0.02299252<2.92 biasesq i32*chan
    AddNode("S51_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S50_Output", 0), GNodeArg(GNA_IN, "S51_Weights", 0), GNodeArg(GNA_IN, "S51_Biases", 0), GNodeArg(GNA_OUT, "S51_Output", 0), GNodeArg(GNA_IN, "S51_Mul_scale", 0), GNodeArg(GNA_IN, "S51_Mul_shift", 0), GNodeArg(GNA_IN, "S51_Infos", 0)));
    // Node S52_MatAdd_16x1x299 in1q -16.44<i8*0.12847272<16.32 in2q -2.94<i8*0.02299252<2.92 outq -17.26<i8*0.13481569<17.12
    AddNode("S52_MatAdd_16x1x299", Bindings(4, GNodeArg(GNA_IN, "S48_Output", 0), GNodeArg(GNA_IN, "S51_Output", 0), GNodeArg(GNA_OUT, "S52_Output", 0), GNodeArg(GNA_IN, "S52_Infos", 0)));
    // Node S53_MatAdd_16x1x299 in1q -35.84<i8*0.28000027<35.56 in2q -2.94<i8*0.02299252<2.92 outq -36.75<i8*0.28712147<36.46
    AddNode("S53_MatAdd_16x1x299", Bindings(4, GNodeArg(GNA_IN, "S49_Output", 0), GNodeArg(GNA_IN, "S51_Output", 0), GNodeArg(GNA_OUT, "S53_Output", 0), GNodeArg(GNA_IN, "S53_Infos", 0)));
    // Node S54_Conv2d_16x16x1x3_Relu inq -36.75<i8*0.28712147<36.46 weightsq chan<i8*chan<chan outq -3.14<i8*0.02454505<3.12 biasesq i32*chan
    AddNode("S54_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S53_Output", 0), GNodeArg(GNA_IN, "S54_Weights", 0), GNodeArg(GNA_IN, "S54_Biases", 0), GNodeArg(GNA_OUT, "S54_Output", 0), GNodeArg(GNA_IN, "S54_Mul_scale", 0), GNodeArg(GNA_IN, "S54_Mul_shift", 0), GNodeArg(GNA_IN, "S54_Infos", 0)));
    // Node S55_Conv2d_16x16x1x3_Relu inq -3.14<i8*0.02454505<3.12 weightsq chan<i8*chan<chan outq -4.61<i8*0.03603386<4.58 biasesq i32*chan
    AddNode("S55_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S54_Output", 0), GNodeArg(GNA_IN, "S55_Weights", 0), GNodeArg(GNA_IN, "S55_Biases", 0), GNodeArg(GNA_OUT, "S55_Output", 0), GNodeArg(GNA_IN, "S55_Mul_scale", 0), GNodeArg(GNA_IN, "S55_Mul_shift", 0), GNodeArg(GNA_IN, "S55_Infos", 0)));
    // Node S56_MatAdd_16x1x299 in1q -17.26<i8*0.13481569<17.12 in2q -4.61<i8*0.03603386<4.58 outq -17.26<i8*0.13481569<17.12
    AddNode("S56_MatAdd_16x1x299", Bindings(4, GNodeArg(GNA_IN, "S52_Output", 0), GNodeArg(GNA_IN, "S55_Output", 0), GNodeArg(GNA_OUT, "S56_Output", 0), GNodeArg(GNA_IN, "S56_Infos", 0)));
    // Node S57_MatAdd_16x1x299 in1q -36.75<i8*0.28712147<36.46 in2q -4.61<i8*0.03603386<4.58 outq -36.79<i8*0.28742731<36.50
    AddNode("S57_MatAdd_16x1x299", Bindings(4, GNodeArg(GNA_IN, "S53_Output", 0), GNodeArg(GNA_IN, "S55_Output", 0), GNodeArg(GNA_OUT, "S57_Output", 0), GNodeArg(GNA_IN, "S57_Infos", 0)));
    // Node S58_Conv2d_16x16x1x3_Relu inq -36.79<i8*0.28742731<36.50 weightsq chan<i8*chan<chan outq -2.42<i8*0.01893183<2.40 biasesq i32*chan
    AddNode("S58_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S57_Output", 0), GNodeArg(GNA_IN, "S58_Weights", 0), GNodeArg(GNA_IN, "S58_Biases", 0), GNodeArg(GNA_OUT, "S58_Output", 0), GNodeArg(GNA_IN, "S58_Mul_scale", 0), GNodeArg(GNA_IN, "S58_Mul_shift", 0), GNodeArg(GNA_IN, "S58_Infos", 0)));
    // Node S59_Conv2d_16x16x1x3_Relu inq -2.42<i8*0.01893183<2.40 weightsq chan<i8*chan<chan outq -4.42<i8*0.03455089<4.39 biasesq i32*chan
    AddNode("S59_Conv2d_16x16x1x3_Relu", Bindings(7, GNodeArg(GNA_IN, "S58_Output", 0), GNodeArg(GNA_IN, "S59_Weights", 0), GNodeArg(GNA_IN, "S59_Biases", 0), GNodeArg(GNA_OUT, "S59_Output", 0), GNodeArg(GNA_IN, "S59_Mul_scale", 0), GNodeArg(GNA_IN, "S59_Mul_shift", 0), GNodeArg(GNA_IN, "S59_Infos", 0)));
    // Node S60_MatAdd_16x1x299 in1q -17.26<i8*0.13481569<17.12 in2q -4.42<i8*0.03455089<4.39 outq -19.04<i8*0.14875674<18.89
    AddNode("S60_MatAdd_16x1x299", Bindings(4, GNodeArg(GNA_IN, "S56_Output", 0), GNodeArg(GNA_IN, "S59_Output", 0), GNodeArg(GNA_OUT, "Output_1", 0), GNodeArg(GNA_IN, "S60_Infos", 0)));
    CloseGraph();
#endif
}

int main(int argc, char **argv)

{
    if (TilerParseOptions(argc, argv)) {
            printf("Failed to initialize or incorrect output arguments directory.\n"); return 1;
    }
    quant_modelModel(52000, 300*1024, 8*1024*1024, 20*1024*1024);
    GenerateTilingCode();
    return 0;
}

#include "quant_modelKernels.h"
L1_CL_MEM AT_L1_POINTER quant_model_L1_Memory;
L2_MEM AT_L2_POINTER quant_model_L2_Memory;
static AT_HYPERFLASH_FS_T HyperFlash;
void S2_Conv2d_16x20x1x3(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 30968 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 20, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 960 [D1, [0 x 960, 960]][D0, [0 x 960, 960]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 960, 960]][D0, [0 x 960, 960]]
		Tile0: [0, 960, 960], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 5980 [D0, [0 x 5980, 5980]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 5980, 5980]]
		Tile0: [0, 5980, 5980], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+11820);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+5980);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (20);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (20);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+6076);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+11820);
	KerArg1->Pad = (v4s) ((v4s){2,0,0,0});
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+11820);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+7036);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+6044);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+6060);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+30956);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5980), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+6044), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+6060), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+6076), 960, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 5980, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+30956), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+30956))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConv3x1Stride1x1_SQ8, (void *) KerArg1);
				__CALL(KerParConv3x1Stride1x1_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+7036), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S3_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){2,0,0,0});
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConv3x1Stride1x1_SQ8, (void *) KerArg1);
				__CALL(KerParConv3x1Stride1x1_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S4_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){2,0,0,0});
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConv3x1Stride1x1_SQ8, (void *) KerArg1);
				__CALL(KerParConv3x1Stride1x1_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S5_MatAdd_16x1x299_Relu(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14364 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (quant_model_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (quant_model_L1_Memory+4784);
	KerArg0->Out = (signed char *__restrict__) (quant_model_L1_Memory+9568);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (299);
	KerArg0->DoScale = (unsigned char) (0);
	KerArg0->Infos = (signed char *__restrict__) (quant_model_L1_Memory+14352);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 4784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+14352), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_ReLU_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_ReLU_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+9568), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S6_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){4,0,0,0});
	KerArg1->N = (unsigned char) (3);
	KerArg1->S = (unsigned char) (1);
	KerArg1->D = (unsigned char) (2);
	KerArg1->Ny = (unsigned char) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg1->Dy = (unsigned char) (2);
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConvNxMDxDyStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMDxDyStrideSxSy_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S7_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){4,0,0,0});
	KerArg1->N = (unsigned char) (3);
	KerArg1->S = (unsigned char) (1);
	KerArg1->D = (unsigned char) (2);
	KerArg1->Ny = (unsigned char) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg1->Dy = (unsigned char) (2);
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConvNxMDxDyStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMDxDyStrideSxSy_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S8_MatAdd_16x1x299(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14364 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (quant_model_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (quant_model_L1_Memory+4784);
	KerArg0->Out = (signed char *__restrict__) (quant_model_L1_Memory+9568);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (299);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (quant_model_L1_Memory+14352);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 4784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+14352), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+9568), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S9_MatAdd_16x1x299(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14364 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (quant_model_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (quant_model_L1_Memory+4784);
	KerArg0->Out = (signed char *__restrict__) (quant_model_L1_Memory+9568);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (299);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (quant_model_L1_Memory+14352);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 4784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+14352), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+9568), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S10_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){8,0,0,0});
	KerArg1->N = (unsigned char) (3);
	KerArg1->S = (unsigned char) (1);
	KerArg1->D = (unsigned char) (4);
	KerArg1->Ny = (unsigned char) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg1->Dy = (unsigned char) (4);
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConvNxMDxDyStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMDxDyStrideSxSy_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S11_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){8,0,0,0});
	KerArg1->N = (unsigned char) (3);
	KerArg1->S = (unsigned char) (1);
	KerArg1->D = (unsigned char) (4);
	KerArg1->Ny = (unsigned char) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg1->Dy = (unsigned char) (4);
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConvNxMDxDyStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMDxDyStrideSxSy_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S12_MatAdd_16x1x299(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14364 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (quant_model_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (quant_model_L1_Memory+4784);
	KerArg0->Out = (signed char *__restrict__) (quant_model_L1_Memory+9568);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (299);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (quant_model_L1_Memory+14352);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 4784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+14352), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+9568), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S13_MatAdd_16x1x299(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14364 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (quant_model_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (quant_model_L1_Memory+4784);
	KerArg0->Out = (signed char *__restrict__) (quant_model_L1_Memory+9568);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (299);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (quant_model_L1_Memory+14352);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 4784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+14352), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+9568), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S14_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){16,0,0,0});
	KerArg1->N = (unsigned char) (3);
	KerArg1->S = (unsigned char) (1);
	KerArg1->D = (unsigned char) (8);
	KerArg1->Ny = (unsigned char) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg1->Dy = (unsigned char) (8);
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConvNxMDxDyStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMDxDyStrideSxSy_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S15_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){16,0,0,0});
	KerArg1->N = (unsigned char) (3);
	KerArg1->S = (unsigned char) (1);
	KerArg1->D = (unsigned char) (8);
	KerArg1->Ny = (unsigned char) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg1->Dy = (unsigned char) (8);
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConvNxMDxDyStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMDxDyStrideSxSy_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S16_MatAdd_16x1x299(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14364 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (quant_model_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (quant_model_L1_Memory+4784);
	KerArg0->Out = (signed char *__restrict__) (quant_model_L1_Memory+9568);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (299);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (quant_model_L1_Memory+14352);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 4784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+14352), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+9568), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S17_MatAdd_16x1x299(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14364 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (quant_model_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (quant_model_L1_Memory+4784);
	KerArg0->Out = (signed char *__restrict__) (quant_model_L1_Memory+9568);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (299);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (quant_model_L1_Memory+14352);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 4784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+14352), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+9568), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S18_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){32,0,0,0});
	KerArg1->N = (unsigned char) (3);
	KerArg1->S = (unsigned char) (1);
	KerArg1->D = (unsigned char) (16);
	KerArg1->Ny = (unsigned char) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg1->Dy = (unsigned char) (16);
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConvNxMDxDyStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMDxDyStrideSxSy_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S19_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){32,0,0,0});
	KerArg1->N = (unsigned char) (3);
	KerArg1->S = (unsigned char) (1);
	KerArg1->D = (unsigned char) (16);
	KerArg1->Ny = (unsigned char) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg1->Dy = (unsigned char) (16);
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConvNxMDxDyStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMDxDyStrideSxSy_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S20_MatAdd_16x1x299(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14364 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (quant_model_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (quant_model_L1_Memory+4784);
	KerArg0->Out = (signed char *__restrict__) (quant_model_L1_Memory+9568);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (299);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (quant_model_L1_Memory+14352);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 4784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+14352), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+9568), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S21_MatAdd_16x1x299(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14364 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (quant_model_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (quant_model_L1_Memory+4784);
	KerArg0->Out = (signed char *__restrict__) (quant_model_L1_Memory+9568);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (299);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (quant_model_L1_Memory+14352);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 4784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+14352), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+9568), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S22_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){2,0,0,0});
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConv3x1Stride1x1_SQ8, (void *) KerArg1);
				__CALL(KerParConv3x1Stride1x1_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S23_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){2,0,0,0});
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConv3x1Stride1x1_SQ8, (void *) KerArg1);
				__CALL(KerParConv3x1Stride1x1_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S24_MatAdd_16x1x299(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14364 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (quant_model_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (quant_model_L1_Memory+4784);
	KerArg0->Out = (signed char *__restrict__) (quant_model_L1_Memory+9568);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (299);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (quant_model_L1_Memory+14352);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 4784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+14352), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+9568), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S25_MatAdd_16x1x299(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14364 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (quant_model_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (quant_model_L1_Memory+4784);
	KerArg0->Out = (signed char *__restrict__) (quant_model_L1_Memory+9568);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (299);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (quant_model_L1_Memory+14352);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 4784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+14352), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+9568), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S26_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){4,0,0,0});
	KerArg1->N = (unsigned char) (3);
	KerArg1->S = (unsigned char) (1);
	KerArg1->D = (unsigned char) (2);
	KerArg1->Ny = (unsigned char) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg1->Dy = (unsigned char) (2);
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConvNxMDxDyStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMDxDyStrideSxSy_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S27_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){4,0,0,0});
	KerArg1->N = (unsigned char) (3);
	KerArg1->S = (unsigned char) (1);
	KerArg1->D = (unsigned char) (2);
	KerArg1->Ny = (unsigned char) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg1->Dy = (unsigned char) (2);
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConvNxMDxDyStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMDxDyStrideSxSy_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S28_MatAdd_16x1x299(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14364 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (quant_model_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (quant_model_L1_Memory+4784);
	KerArg0->Out = (signed char *__restrict__) (quant_model_L1_Memory+9568);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (299);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (quant_model_L1_Memory+14352);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 4784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+14352), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+9568), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S29_MatAdd_16x1x299(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14364 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (quant_model_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (quant_model_L1_Memory+4784);
	KerArg0->Out = (signed char *__restrict__) (quant_model_L1_Memory+9568);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (299);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (quant_model_L1_Memory+14352);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 4784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+14352), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+9568), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S30_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){8,0,0,0});
	KerArg1->N = (unsigned char) (3);
	KerArg1->S = (unsigned char) (1);
	KerArg1->D = (unsigned char) (4);
	KerArg1->Ny = (unsigned char) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg1->Dy = (unsigned char) (4);
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConvNxMDxDyStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMDxDyStrideSxSy_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S31_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){8,0,0,0});
	KerArg1->N = (unsigned char) (3);
	KerArg1->S = (unsigned char) (1);
	KerArg1->D = (unsigned char) (4);
	KerArg1->Ny = (unsigned char) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg1->Dy = (unsigned char) (4);
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConvNxMDxDyStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMDxDyStrideSxSy_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S32_MatAdd_16x1x299(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14364 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (quant_model_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (quant_model_L1_Memory+4784);
	KerArg0->Out = (signed char *__restrict__) (quant_model_L1_Memory+9568);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (299);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (quant_model_L1_Memory+14352);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 4784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+14352), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+9568), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S33_MatAdd_16x1x299(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14364 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (quant_model_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (quant_model_L1_Memory+4784);
	KerArg0->Out = (signed char *__restrict__) (quant_model_L1_Memory+9568);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (299);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (quant_model_L1_Memory+14352);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 4784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+14352), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+9568), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S34_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){16,0,0,0});
	KerArg1->N = (unsigned char) (3);
	KerArg1->S = (unsigned char) (1);
	KerArg1->D = (unsigned char) (8);
	KerArg1->Ny = (unsigned char) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg1->Dy = (unsigned char) (8);
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConvNxMDxDyStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMDxDyStrideSxSy_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S35_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){16,0,0,0});
	KerArg1->N = (unsigned char) (3);
	KerArg1->S = (unsigned char) (1);
	KerArg1->D = (unsigned char) (8);
	KerArg1->Ny = (unsigned char) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg1->Dy = (unsigned char) (8);
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConvNxMDxDyStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMDxDyStrideSxSy_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S36_MatAdd_16x1x299(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14364 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (quant_model_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (quant_model_L1_Memory+4784);
	KerArg0->Out = (signed char *__restrict__) (quant_model_L1_Memory+9568);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (299);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (quant_model_L1_Memory+14352);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 4784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+14352), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+9568), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S37_MatAdd_16x1x299(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14364 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (quant_model_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (quant_model_L1_Memory+4784);
	KerArg0->Out = (signed char *__restrict__) (quant_model_L1_Memory+9568);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (299);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (quant_model_L1_Memory+14352);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 4784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+14352), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+9568), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S38_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){32,0,0,0});
	KerArg1->N = (unsigned char) (3);
	KerArg1->S = (unsigned char) (1);
	KerArg1->D = (unsigned char) (16);
	KerArg1->Ny = (unsigned char) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg1->Dy = (unsigned char) (16);
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConvNxMDxDyStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMDxDyStrideSxSy_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S39_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){32,0,0,0});
	KerArg1->N = (unsigned char) (3);
	KerArg1->S = (unsigned char) (1);
	KerArg1->D = (unsigned char) (16);
	KerArg1->Ny = (unsigned char) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg1->Dy = (unsigned char) (16);
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConvNxMDxDyStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMDxDyStrideSxSy_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S40_MatAdd_16x1x299(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14364 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (quant_model_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (quant_model_L1_Memory+4784);
	KerArg0->Out = (signed char *__restrict__) (quant_model_L1_Memory+9568);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (299);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (quant_model_L1_Memory+14352);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 4784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+14352), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+9568), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S41_MatAdd_16x1x299(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14364 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (quant_model_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (quant_model_L1_Memory+4784);
	KerArg0->Out = (signed char *__restrict__) (quant_model_L1_Memory+9568);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (299);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (quant_model_L1_Memory+14352);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 4784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+14352), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+9568), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S42_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){2,0,0,0});
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConv3x1Stride1x1_SQ8, (void *) KerArg1);
				__CALL(KerParConv3x1Stride1x1_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S43_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){2,0,0,0});
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConv3x1Stride1x1_SQ8, (void *) KerArg1);
				__CALL(KerParConv3x1Stride1x1_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S44_MatAdd_16x1x299(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14364 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (quant_model_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (quant_model_L1_Memory+4784);
	KerArg0->Out = (signed char *__restrict__) (quant_model_L1_Memory+9568);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (299);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (quant_model_L1_Memory+14352);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 4784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+14352), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+9568), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S45_MatAdd_16x1x299(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14364 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (quant_model_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (quant_model_L1_Memory+4784);
	KerArg0->Out = (signed char *__restrict__) (quant_model_L1_Memory+9568);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (299);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (quant_model_L1_Memory+14352);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 4784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+14352), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+9568), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S46_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){4,0,0,0});
	KerArg1->N = (unsigned char) (3);
	KerArg1->S = (unsigned char) (1);
	KerArg1->D = (unsigned char) (2);
	KerArg1->Ny = (unsigned char) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg1->Dy = (unsigned char) (2);
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConvNxMDxDyStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMDxDyStrideSxSy_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S47_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){4,0,0,0});
	KerArg1->N = (unsigned char) (3);
	KerArg1->S = (unsigned char) (1);
	KerArg1->D = (unsigned char) (2);
	KerArg1->Ny = (unsigned char) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg1->Dy = (unsigned char) (2);
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConvNxMDxDyStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMDxDyStrideSxSy_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S48_MatAdd_16x1x299(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14364 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (quant_model_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (quant_model_L1_Memory+4784);
	KerArg0->Out = (signed char *__restrict__) (quant_model_L1_Memory+9568);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (299);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (quant_model_L1_Memory+14352);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 4784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+14352), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+9568), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S49_MatAdd_16x1x299(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14364 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (quant_model_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (quant_model_L1_Memory+4784);
	KerArg0->Out = (signed char *__restrict__) (quant_model_L1_Memory+9568);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (299);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (quant_model_L1_Memory+14352);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 4784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+14352), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+9568), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S50_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){8,0,0,0});
	KerArg1->N = (unsigned char) (3);
	KerArg1->S = (unsigned char) (1);
	KerArg1->D = (unsigned char) (4);
	KerArg1->Ny = (unsigned char) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg1->Dy = (unsigned char) (4);
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConvNxMDxDyStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMDxDyStrideSxSy_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S51_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){8,0,0,0});
	KerArg1->N = (unsigned char) (3);
	KerArg1->S = (unsigned char) (1);
	KerArg1->D = (unsigned char) (4);
	KerArg1->Ny = (unsigned char) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg1->Dy = (unsigned char) (4);
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConvNxMDxDyStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMDxDyStrideSxSy_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S52_MatAdd_16x1x299(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14364 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (quant_model_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (quant_model_L1_Memory+4784);
	KerArg0->Out = (signed char *__restrict__) (quant_model_L1_Memory+9568);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (299);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (quant_model_L1_Memory+14352);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 4784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+14352), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+9568), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S53_MatAdd_16x1x299(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14364 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (quant_model_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (quant_model_L1_Memory+4784);
	KerArg0->Out = (signed char *__restrict__) (quant_model_L1_Memory+9568);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (299);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (quant_model_L1_Memory+14352);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 4784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+14352), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+9568), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S54_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){16,0,0,0});
	KerArg1->N = (unsigned char) (3);
	KerArg1->S = (unsigned char) (1);
	KerArg1->D = (unsigned char) (8);
	KerArg1->Ny = (unsigned char) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg1->Dy = (unsigned char) (8);
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConvNxMDxDyStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMDxDyStrideSxSy_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S55_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){16,0,0,0});
	KerArg1->N = (unsigned char) (3);
	KerArg1->S = (unsigned char) (1);
	KerArg1->D = (unsigned char) (8);
	KerArg1->Ny = (unsigned char) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg1->Dy = (unsigned char) (8);
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConvNxMDxDyStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMDxDyStrideSxSy_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S56_MatAdd_16x1x299(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14364 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (quant_model_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (quant_model_L1_Memory+4784);
	KerArg0->Out = (signed char *__restrict__) (quant_model_L1_Memory+9568);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (299);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (quant_model_L1_Memory+14352);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 4784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+14352), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+9568), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S57_MatAdd_16x1x299(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14364 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (quant_model_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (quant_model_L1_Memory+4784);
	KerArg0->Out = (signed char *__restrict__) (quant_model_L1_Memory+9568);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (299);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (quant_model_L1_Memory+14352);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 4784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+14352), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+9568), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S58_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){32,0,0,0});
	KerArg1->N = (unsigned char) (3);
	KerArg1->S = (unsigned char) (1);
	KerArg1->D = (unsigned char) (16);
	KerArg1->Ny = (unsigned char) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg1->Dy = (unsigned char) (16);
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConvNxMDxDyStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMDxDyStrideSxSy_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S59_Conv2d_16x16x1x3_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		unsigned char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 29580 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Last;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 16, Tiled: 1]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 19136 [D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 19136, 19136]][Tile0, 1:[299x1], 4]
		Tile0: [0, 19136, 19136], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 16 [D1, [0 x 16, 16]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 16, 16]]
		Tile0: [0, 16, 16], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0], T1: [D1: 0], T2: [D1: 0]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 768 [D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 768, 768]][D0, [0 x 768, 768]]
		Tile0: [0, 768, 768], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][D0: 0], T1: [D1: 0][D0: 0], T2: [D1: 0][D0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D1: 0][Tile0: 0], T1: [D1: 0][Tile0: 0], T2: [D1: 0][Tile0: 0]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[299x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[299x1], 1][D0, [0 x 4784, 4784]]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg0->W = (unsigned short int) (299);
	KerArg0->H = (unsigned short int) (1);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->Bias = (void * __restrict__) (quant_model_L1_Memory+4784);
	KerArg1->In = (signed char * __restrict__) (quant_model_L1_Memory+0);
	KerArg1->W = (unsigned short int) (299);
	KerArg1->UsedW = (unsigned short int) (299);
	KerArg1->H = (unsigned short int) (1);
	KerArg1->InFeatures = (unsigned short int) (16);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->TotalInFeatures = (unsigned short int) (16);
	KerArg1->Filter = (signed char * __restrict__) (quant_model_L1_Memory+4880);
	KerArg1->Out = (int * __restrict__) (quant_model_L1_Memory+10432);
	KerArg1->Pad = (v4s) ((v4s){32,0,0,0});
	KerArg1->N = (unsigned char) (3);
	KerArg1->S = (unsigned char) (1);
	KerArg1->D = (unsigned char) (16);
	KerArg1->Ny = (unsigned char) (1);
	KerArg1->Sy = (unsigned char) (1);
	KerArg1->Dy = (unsigned char) (16);
	KerArg2->In = (int *__restrict__) (quant_model_L1_Memory+10432);
	KerArg2->Out = (void *__restrict__) (quant_model_L1_Memory+5648);
	KerArg2->Feat = (unsigned short int) (16);
	KerArg2->W = (unsigned short int) (299);
	KerArg2->H = (unsigned short int) (1);
	KerArg2->Scale = (unsigned char *__restrict__) (quant_model_L1_Memory+4848);
	KerArg2->ScaleN = (unsigned char *__restrict__) (quant_model_L1_Memory+4864);
	KerArg2->Infos = (signed char *__restrict__) (quant_model_L1_Memory+29568);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4848), 16, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4864), 16, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4880), 768, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+29568), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(quant_model_L1_Memory+29568))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->UsedH = (unsigned short int) (1);
				AT_FORK(gap_ncore(), (void *) KerParConvNxMDxDyStrideSxSy_SQ8, (void *) KerArg1);
				__CALL(KerParConvNxMDxDyStrideSxSy_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReduct_CC_ReLU_SQ8, (void *) KerArg2);
			__CALL(KerParReduct_CC_ReLU_SQ8, KerArg2);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+5648), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S60_MatAdd_16x1x299(
		signed char * __restrict__ In1,
		signed char * __restrict__ In2,
		signed char * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 14364 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaW_Evt1;
	KerMat3_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 16, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In1, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: In2, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 4784 [D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 4784, 4784]][Tile0, 1:[1x299], 1]
		Tile0: [0, 4784, 4784], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [D0: 0][Tile0: 0], T1: [D0: 0][Tile0: 0], T2: [D0: 0][Tile0: 0]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 0, 0], Tile2; [0, 0, 0]
		T0: [Tile0: 0], T1: [Tile0: 0], T2: [Tile0: 0]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In1 = (signed char *__restrict__) (quant_model_L1_Memory+0);
	KerArg0->In2 = (signed char *__restrict__) (quant_model_L1_Memory+4784);
	KerArg0->Out = (signed char *__restrict__) (quant_model_L1_Memory+9568);
	KerArg0->Feat = (unsigned short int) (16);
	KerArg0->W = (unsigned short int) (1);
	KerArg0->H = (unsigned short int) (299);
	KerArg0->DoScale = (unsigned char) (1);
	KerArg0->Infos = (signed char *__restrict__) (quant_model_L1_Memory+14352);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In1+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+0), 4784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In1 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In2+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+4784), 4784, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read In2 */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+14352), 9, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParMatAdd_SQ8, (void *) KerArg0);
			__CALL(KerParMatAdd_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) quant_model_L1_Memory+9568), 4784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
int quant_modelCNN_Construct()

{
	AT_HYPERFLASH_FS_FC_EVENT UchanHF1;
	AT_HYPERFLASH_FS_CONF_T HyperFlashConf;
	int Error;
	AT_HYPERFLASH_FS_CONF_INIT(&HyperFlashConf, AT_MEM_L3_HFLASH, 0);
	AT_HYPERFLASH_FS_OPEN(&HyperFlash, &HyperFlashConf, "quant_model_L3_Flash_Const.dat", &Error);
	if (Error) return 1;
	quant_model_L2_Memory = (AT_L2_POINTER) AT_L2_ALLOC(0, 56385);
	if (quant_model_L2_Memory == 0) return 3;
	quant_model_L1_Memory = (AT_L1_POINTER) AT_L1_ALLOC(0, 30968);
	if (quant_model_L1_Memory == 0) return 4;
	/* Moving S2_Infos, size 9 from HyperFlash at 26976 to (size 9) L2 at 26976..26984 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26976), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26976), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S2_Weights, size 960 from HyperFlash at 0 to (size 960) L2 at 0..959 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 0), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 0), 960, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S2_Biases, size 64 from HyperFlash at 24000 to (size 64) L2 at 24000..24063 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 24000), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 24000), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S2_Mul_scale, size 16 from HyperFlash at 25984 to (size 16) L2 at 25984..25999 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 25984), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 25984), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S2_Mul_shift, size 16 from HyperFlash at 26000 to (size 16) L2 at 26000..26015 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26000), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26000), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S3_Infos, size 9 from HyperFlash at 26988 to (size 9) L2 at 26988..26996 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26988), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26988), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S3_Weights, size 768 from HyperFlash at 960 to (size 768) L2 at 960..1727 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 960), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 960), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S3_Biases, size 64 from HyperFlash at 24064 to (size 64) L2 at 24064..24127 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 24064), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 24064), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S3_Mul_scale, size 16 from HyperFlash at 26016 to (size 16) L2 at 26016..26031 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26016), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26016), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S3_Mul_shift, size 16 from HyperFlash at 26032 to (size 16) L2 at 26032..26047 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26032), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26032), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S4_Infos, size 9 from HyperFlash at 27000 to (size 9) L2 at 27000..27008 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27000), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27000), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S4_Weights, size 768 from HyperFlash at 1728 to (size 768) L2 at 1728..2495 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 1728), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 1728), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S4_Biases, size 64 from HyperFlash at 24128 to (size 64) L2 at 24128..24191 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 24128), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 24128), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S4_Mul_scale, size 16 from HyperFlash at 26048 to (size 16) L2 at 26048..26063 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26048), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26048), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S4_Mul_shift, size 16 from HyperFlash at 26064 to (size 16) L2 at 26064..26079 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26064), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26064), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S5_Infos, size 9 from HyperFlash at 27012 to (size 9) L2 at 27012..27020 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27012), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27012), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S6_Infos, size 9 from HyperFlash at 27024 to (size 9) L2 at 27024..27032 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27024), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27024), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S6_Weights, size 768 from HyperFlash at 2496 to (size 768) L2 at 2496..3263 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 2496), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 2496), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S6_Biases, size 64 from HyperFlash at 24192 to (size 64) L2 at 24192..24255 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 24192), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 24192), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S6_Mul_scale, size 16 from HyperFlash at 26080 to (size 16) L2 at 26080..26095 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26080), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26080), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S6_Mul_shift, size 16 from HyperFlash at 26096 to (size 16) L2 at 26096..26111 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26096), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26096), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S7_Infos, size 9 from HyperFlash at 27036 to (size 9) L2 at 27036..27044 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27036), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27036), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S7_Weights, size 768 from HyperFlash at 3264 to (size 768) L2 at 3264..4031 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 3264), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 3264), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S7_Biases, size 64 from HyperFlash at 24256 to (size 64) L2 at 24256..24319 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 24256), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 24256), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S7_Mul_scale, size 16 from HyperFlash at 26112 to (size 16) L2 at 26112..26127 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26112), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26112), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S7_Mul_shift, size 16 from HyperFlash at 26128 to (size 16) L2 at 26128..26143 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26128), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26128), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S8_Infos, size 9 from HyperFlash at 27048 to (size 9) L2 at 27048..27056 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27048), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27048), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S9_Infos, size 9 from HyperFlash at 27060 to (size 9) L2 at 27060..27068 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27060), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27060), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S10_Infos, size 9 from HyperFlash at 27072 to (size 9) L2 at 27072..27080 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27072), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27072), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S10_Weights, size 768 from HyperFlash at 4032 to (size 768) L2 at 4032..4799 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 4032), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 4032), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S10_Biases, size 64 from HyperFlash at 24320 to (size 64) L2 at 24320..24383 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 24320), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 24320), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S10_Mul_scale, size 16 from HyperFlash at 26144 to (size 16) L2 at 26144..26159 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26144), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26144), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S10_Mul_shift, size 16 from HyperFlash at 26160 to (size 16) L2 at 26160..26175 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26160), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26160), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S11_Infos, size 9 from HyperFlash at 27084 to (size 9) L2 at 27084..27092 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27084), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27084), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S11_Weights, size 768 from HyperFlash at 4800 to (size 768) L2 at 4800..5567 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 4800), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 4800), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S11_Biases, size 64 from HyperFlash at 24384 to (size 64) L2 at 24384..24447 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 24384), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 24384), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S11_Mul_scale, size 16 from HyperFlash at 26176 to (size 16) L2 at 26176..26191 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26176), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26176), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S11_Mul_shift, size 16 from HyperFlash at 26192 to (size 16) L2 at 26192..26207 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26192), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26192), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S12_Infos, size 9 from HyperFlash at 27096 to (size 9) L2 at 27096..27104 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27096), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27096), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S13_Infos, size 9 from HyperFlash at 27108 to (size 9) L2 at 27108..27116 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27108), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27108), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S14_Infos, size 9 from HyperFlash at 27120 to (size 9) L2 at 27120..27128 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27120), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27120), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S14_Weights, size 768 from HyperFlash at 5568 to (size 768) L2 at 5568..6335 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 5568), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 5568), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S14_Biases, size 64 from HyperFlash at 24448 to (size 64) L2 at 24448..24511 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 24448), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 24448), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S14_Mul_scale, size 16 from HyperFlash at 26208 to (size 16) L2 at 26208..26223 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26208), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26208), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S14_Mul_shift, size 16 from HyperFlash at 26224 to (size 16) L2 at 26224..26239 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26224), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26224), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S15_Infos, size 9 from HyperFlash at 27132 to (size 9) L2 at 27132..27140 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27132), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27132), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S15_Weights, size 768 from HyperFlash at 6336 to (size 768) L2 at 6336..7103 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 6336), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 6336), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S15_Biases, size 64 from HyperFlash at 24512 to (size 64) L2 at 24512..24575 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 24512), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 24512), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S15_Mul_scale, size 16 from HyperFlash at 26240 to (size 16) L2 at 26240..26255 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26240), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26240), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S15_Mul_shift, size 16 from HyperFlash at 26256 to (size 16) L2 at 26256..26271 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26256), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26256), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S16_Infos, size 9 from HyperFlash at 27144 to (size 9) L2 at 27144..27152 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27144), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27144), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S17_Infos, size 9 from HyperFlash at 27156 to (size 9) L2 at 27156..27164 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27156), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27156), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S18_Infos, size 9 from HyperFlash at 27168 to (size 9) L2 at 27168..27176 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27168), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27168), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S18_Weights, size 768 from HyperFlash at 7104 to (size 768) L2 at 7104..7871 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 7104), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 7104), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S18_Biases, size 64 from HyperFlash at 24576 to (size 64) L2 at 24576..24639 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 24576), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 24576), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S18_Mul_scale, size 16 from HyperFlash at 26272 to (size 16) L2 at 26272..26287 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26272), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26272), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S18_Mul_shift, size 16 from HyperFlash at 26288 to (size 16) L2 at 26288..26303 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26288), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26288), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S19_Infos, size 9 from HyperFlash at 27180 to (size 9) L2 at 27180..27188 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27180), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27180), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S19_Weights, size 768 from HyperFlash at 7872 to (size 768) L2 at 7872..8639 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 7872), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 7872), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S19_Biases, size 64 from HyperFlash at 24640 to (size 64) L2 at 24640..24703 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 24640), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 24640), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S19_Mul_scale, size 16 from HyperFlash at 26304 to (size 16) L2 at 26304..26319 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26304), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26304), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S19_Mul_shift, size 16 from HyperFlash at 26320 to (size 16) L2 at 26320..26335 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26320), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26320), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S20_Infos, size 9 from HyperFlash at 27192 to (size 9) L2 at 27192..27200 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27192), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27192), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S21_Infos, size 9 from HyperFlash at 27204 to (size 9) L2 at 27204..27212 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27204), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27204), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S22_Infos, size 9 from HyperFlash at 27216 to (size 9) L2 at 27216..27224 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27216), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27216), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S22_Weights, size 768 from HyperFlash at 8640 to (size 768) L2 at 8640..9407 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 8640), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 8640), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S22_Biases, size 64 from HyperFlash at 24704 to (size 64) L2 at 24704..24767 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 24704), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 24704), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S22_Mul_scale, size 16 from HyperFlash at 26336 to (size 16) L2 at 26336..26351 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26336), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26336), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S22_Mul_shift, size 16 from HyperFlash at 26352 to (size 16) L2 at 26352..26367 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26352), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26352), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S23_Infos, size 9 from HyperFlash at 27228 to (size 9) L2 at 27228..27236 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27228), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27228), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S23_Weights, size 768 from HyperFlash at 9408 to (size 768) L2 at 9408..10175 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 9408), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 9408), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S23_Biases, size 64 from HyperFlash at 24768 to (size 64) L2 at 24768..24831 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 24768), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 24768), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S23_Mul_scale, size 16 from HyperFlash at 26368 to (size 16) L2 at 26368..26383 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26368), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26368), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S23_Mul_shift, size 16 from HyperFlash at 26384 to (size 16) L2 at 26384..26399 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26384), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26384), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S24_Infos, size 9 from HyperFlash at 27240 to (size 9) L2 at 27240..27248 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27240), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27240), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S25_Infos, size 9 from HyperFlash at 27252 to (size 9) L2 at 27252..27260 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27252), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27252), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S26_Infos, size 9 from HyperFlash at 27264 to (size 9) L2 at 27264..27272 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27264), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27264), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S26_Weights, size 768 from HyperFlash at 10176 to (size 768) L2 at 10176..10943 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 10176), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 10176), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S26_Biases, size 64 from HyperFlash at 24832 to (size 64) L2 at 24832..24895 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 24832), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 24832), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S26_Mul_scale, size 16 from HyperFlash at 26400 to (size 16) L2 at 26400..26415 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26400), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26400), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S26_Mul_shift, size 16 from HyperFlash at 26416 to (size 16) L2 at 26416..26431 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26416), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26416), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S27_Infos, size 9 from HyperFlash at 27276 to (size 9) L2 at 27276..27284 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27276), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27276), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S27_Weights, size 768 from HyperFlash at 10944 to (size 768) L2 at 10944..11711 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 10944), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 10944), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S27_Biases, size 64 from HyperFlash at 24896 to (size 64) L2 at 24896..24959 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 24896), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 24896), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S27_Mul_scale, size 16 from HyperFlash at 26432 to (size 16) L2 at 26432..26447 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26432), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26432), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S27_Mul_shift, size 16 from HyperFlash at 26448 to (size 16) L2 at 26448..26463 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26448), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26448), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S28_Infos, size 9 from HyperFlash at 27288 to (size 9) L2 at 27288..27296 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27288), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27288), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S29_Infos, size 9 from HyperFlash at 27300 to (size 9) L2 at 27300..27308 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27300), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27300), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S30_Infos, size 9 from HyperFlash at 27312 to (size 9) L2 at 27312..27320 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27312), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27312), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S30_Weights, size 768 from HyperFlash at 11712 to (size 768) L2 at 11712..12479 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 11712), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 11712), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S30_Biases, size 64 from HyperFlash at 24960 to (size 64) L2 at 24960..25023 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 24960), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 24960), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S30_Mul_scale, size 16 from HyperFlash at 26464 to (size 16) L2 at 26464..26479 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26464), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26464), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S30_Mul_shift, size 16 from HyperFlash at 26480 to (size 16) L2 at 26480..26495 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26480), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26480), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S31_Infos, size 9 from HyperFlash at 27324 to (size 9) L2 at 27324..27332 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27324), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27324), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S31_Weights, size 768 from HyperFlash at 12480 to (size 768) L2 at 12480..13247 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 12480), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 12480), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S31_Biases, size 64 from HyperFlash at 25024 to (size 64) L2 at 25024..25087 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 25024), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 25024), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S31_Mul_scale, size 16 from HyperFlash at 26496 to (size 16) L2 at 26496..26511 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26496), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26496), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S31_Mul_shift, size 16 from HyperFlash at 26512 to (size 16) L2 at 26512..26527 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26512), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26512), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S32_Infos, size 9 from HyperFlash at 27336 to (size 9) L2 at 27336..27344 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27336), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27336), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S33_Infos, size 9 from HyperFlash at 27348 to (size 9) L2 at 27348..27356 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27348), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27348), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S34_Infos, size 9 from HyperFlash at 27360 to (size 9) L2 at 27360..27368 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27360), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27360), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S34_Weights, size 768 from HyperFlash at 13248 to (size 768) L2 at 13248..14015 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 13248), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 13248), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S34_Biases, size 64 from HyperFlash at 25088 to (size 64) L2 at 25088..25151 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 25088), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 25088), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S34_Mul_scale, size 16 from HyperFlash at 26528 to (size 16) L2 at 26528..26543 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26528), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26528), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S34_Mul_shift, size 16 from HyperFlash at 26544 to (size 16) L2 at 26544..26559 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26544), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26544), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S35_Infos, size 9 from HyperFlash at 27372 to (size 9) L2 at 27372..27380 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27372), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27372), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S35_Weights, size 768 from HyperFlash at 14016 to (size 768) L2 at 14016..14783 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 14016), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 14016), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S35_Biases, size 64 from HyperFlash at 25152 to (size 64) L2 at 25152..25215 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 25152), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 25152), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S35_Mul_scale, size 16 from HyperFlash at 26560 to (size 16) L2 at 26560..26575 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26560), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26560), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S35_Mul_shift, size 16 from HyperFlash at 26576 to (size 16) L2 at 26576..26591 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26576), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26576), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S36_Infos, size 9 from HyperFlash at 27384 to (size 9) L2 at 27384..27392 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27384), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27384), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S37_Infos, size 9 from HyperFlash at 27396 to (size 9) L2 at 27396..27404 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27396), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27396), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S38_Infos, size 9 from HyperFlash at 27408 to (size 9) L2 at 27408..27416 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27408), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27408), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S38_Weights, size 768 from HyperFlash at 14784 to (size 768) L2 at 14784..15551 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 14784), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 14784), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S38_Biases, size 64 from HyperFlash at 25216 to (size 64) L2 at 25216..25279 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 25216), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 25216), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S38_Mul_scale, size 16 from HyperFlash at 26592 to (size 16) L2 at 26592..26607 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26592), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26592), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S38_Mul_shift, size 16 from HyperFlash at 26608 to (size 16) L2 at 26608..26623 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26608), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26608), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S39_Infos, size 9 from HyperFlash at 27420 to (size 9) L2 at 27420..27428 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27420), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27420), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S39_Weights, size 768 from HyperFlash at 15552 to (size 768) L2 at 15552..16319 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 15552), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 15552), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S39_Biases, size 64 from HyperFlash at 25280 to (size 64) L2 at 25280..25343 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 25280), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 25280), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S39_Mul_scale, size 16 from HyperFlash at 26624 to (size 16) L2 at 26624..26639 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26624), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26624), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S39_Mul_shift, size 16 from HyperFlash at 26640 to (size 16) L2 at 26640..26655 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26640), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26640), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S40_Infos, size 9 from HyperFlash at 27432 to (size 9) L2 at 27432..27440 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27432), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27432), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S41_Infos, size 9 from HyperFlash at 27444 to (size 9) L2 at 27444..27452 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27444), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27444), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S42_Infos, size 9 from HyperFlash at 27456 to (size 9) L2 at 27456..27464 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27456), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27456), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S42_Weights, size 768 from HyperFlash at 16320 to (size 768) L2 at 16320..17087 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 16320), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 16320), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S42_Biases, size 64 from HyperFlash at 25344 to (size 64) L2 at 25344..25407 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 25344), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 25344), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S42_Mul_scale, size 16 from HyperFlash at 26656 to (size 16) L2 at 26656..26671 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26656), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26656), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S42_Mul_shift, size 16 from HyperFlash at 26672 to (size 16) L2 at 26672..26687 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26672), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26672), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S43_Infos, size 9 from HyperFlash at 27468 to (size 9) L2 at 27468..27476 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27468), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27468), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S43_Weights, size 768 from HyperFlash at 17088 to (size 768) L2 at 17088..17855 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 17088), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 17088), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S43_Biases, size 64 from HyperFlash at 25408 to (size 64) L2 at 25408..25471 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 25408), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 25408), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S43_Mul_scale, size 16 from HyperFlash at 26688 to (size 16) L2 at 26688..26703 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26688), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26688), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S43_Mul_shift, size 16 from HyperFlash at 26704 to (size 16) L2 at 26704..26719 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26704), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26704), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S44_Infos, size 9 from HyperFlash at 27480 to (size 9) L2 at 27480..27488 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27480), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27480), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S45_Infos, size 9 from HyperFlash at 27492 to (size 9) L2 at 27492..27500 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27492), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27492), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S46_Infos, size 9 from HyperFlash at 27504 to (size 9) L2 at 27504..27512 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27504), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27504), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S46_Weights, size 768 from HyperFlash at 17856 to (size 768) L2 at 17856..18623 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 17856), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 17856), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S46_Biases, size 64 from HyperFlash at 25472 to (size 64) L2 at 25472..25535 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 25472), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 25472), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S46_Mul_scale, size 16 from HyperFlash at 26720 to (size 16) L2 at 26720..26735 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26720), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26720), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S46_Mul_shift, size 16 from HyperFlash at 26736 to (size 16) L2 at 26736..26751 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26736), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26736), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S47_Infos, size 9 from HyperFlash at 27516 to (size 9) L2 at 27516..27524 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27516), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27516), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S47_Weights, size 768 from HyperFlash at 18624 to (size 768) L2 at 18624..19391 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 18624), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 18624), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S47_Biases, size 64 from HyperFlash at 25536 to (size 64) L2 at 25536..25599 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 25536), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 25536), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S47_Mul_scale, size 16 from HyperFlash at 26752 to (size 16) L2 at 26752..26767 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26752), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26752), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S47_Mul_shift, size 16 from HyperFlash at 26768 to (size 16) L2 at 26768..26783 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26768), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26768), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S48_Infos, size 9 from HyperFlash at 27528 to (size 9) L2 at 27528..27536 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27528), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27528), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S49_Infos, size 9 from HyperFlash at 27540 to (size 9) L2 at 27540..27548 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27540), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27540), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S50_Infos, size 9 from HyperFlash at 27552 to (size 9) L2 at 27552..27560 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27552), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27552), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S50_Weights, size 768 from HyperFlash at 19392 to (size 768) L2 at 19392..20159 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 19392), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 19392), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S50_Biases, size 64 from HyperFlash at 25600 to (size 64) L2 at 25600..25663 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 25600), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 25600), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S50_Mul_scale, size 16 from HyperFlash at 26784 to (size 16) L2 at 26784..26799 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26784), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26784), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S50_Mul_shift, size 16 from HyperFlash at 26800 to (size 16) L2 at 26800..26815 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26800), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26800), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S51_Infos, size 9 from HyperFlash at 27564 to (size 9) L2 at 27564..27572 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27564), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27564), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S51_Weights, size 768 from HyperFlash at 20160 to (size 768) L2 at 20160..20927 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 20160), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 20160), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S51_Biases, size 64 from HyperFlash at 25664 to (size 64) L2 at 25664..25727 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 25664), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 25664), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S51_Mul_scale, size 16 from HyperFlash at 26816 to (size 16) L2 at 26816..26831 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26816), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26816), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S51_Mul_shift, size 16 from HyperFlash at 26832 to (size 16) L2 at 26832..26847 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26832), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26832), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S52_Infos, size 9 from HyperFlash at 27576 to (size 9) L2 at 27576..27584 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27576), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27576), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S53_Infos, size 9 from HyperFlash at 27588 to (size 9) L2 at 27588..27596 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27588), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27588), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S54_Infos, size 9 from HyperFlash at 27600 to (size 9) L2 at 27600..27608 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27600), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27600), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S54_Weights, size 768 from HyperFlash at 20928 to (size 768) L2 at 20928..21695 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 20928), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 20928), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S54_Biases, size 64 from HyperFlash at 25728 to (size 64) L2 at 25728..25791 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 25728), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 25728), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S54_Mul_scale, size 16 from HyperFlash at 26848 to (size 16) L2 at 26848..26863 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26848), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26848), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S54_Mul_shift, size 16 from HyperFlash at 26864 to (size 16) L2 at 26864..26879 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26864), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26864), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S55_Infos, size 9 from HyperFlash at 27612 to (size 9) L2 at 27612..27620 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27612), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27612), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S55_Weights, size 768 from HyperFlash at 21696 to (size 768) L2 at 21696..22463 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 21696), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 21696), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S55_Biases, size 64 from HyperFlash at 25792 to (size 64) L2 at 25792..25855 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 25792), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 25792), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S55_Mul_scale, size 16 from HyperFlash at 26880 to (size 16) L2 at 26880..26895 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26880), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26880), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S55_Mul_shift, size 16 from HyperFlash at 26896 to (size 16) L2 at 26896..26911 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26896), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26896), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S56_Infos, size 9 from HyperFlash at 27624 to (size 9) L2 at 27624..27632 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27624), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27624), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S57_Infos, size 9 from HyperFlash at 27636 to (size 9) L2 at 27636..27644 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27636), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27636), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S58_Infos, size 9 from HyperFlash at 27648 to (size 9) L2 at 27648..27656 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27648), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27648), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S58_Weights, size 768 from HyperFlash at 22464 to (size 768) L2 at 22464..23231 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 22464), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 22464), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S58_Biases, size 64 from HyperFlash at 25856 to (size 64) L2 at 25856..25919 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 25856), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 25856), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S58_Mul_scale, size 16 from HyperFlash at 26912 to (size 16) L2 at 26912..26927 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26912), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26912), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S58_Mul_shift, size 16 from HyperFlash at 26928 to (size 16) L2 at 26928..26943 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26928), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26928), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S59_Infos, size 9 from HyperFlash at 27660 to (size 9) L2 at 27660..27668 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27660), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27660), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S59_Weights, size 768 from HyperFlash at 23232 to (size 768) L2 at 23232..23999 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 23232), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 23232), 768, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S59_Biases, size 64 from HyperFlash at 25920 to (size 64) L2 at 25920..25983 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 25920), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 25920), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S59_Mul_scale, size 16 from HyperFlash at 26944 to (size 16) L2 at 26944..26959 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26944), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26944), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S59_Mul_shift, size 16 from HyperFlash at 26960 to (size 16) L2 at 26960..26975 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 26960), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 26960), 16, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S60_Infos, size 9 from HyperFlash at 27672 to (size 9) L2 at 27672..27680 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) quant_model_L3_Flash + 27672), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) quant_model_L2_Memory + 27672), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	return 0;
}
int quant_modelCNN_Destruct()

{
	AT_L2_FREE(0, quant_model_L2_Memory, 56385);
	AT_L1_FREE(0, quant_model_L1_Memory, 30968);
	AT_HYPERFLASH_FS_CLOSE(&HyperFlash);
	return 0;
}
unsigned int AT_GraphPerf[59];
unsigned int AT_GraphOperInfosNames[59] = {
	287040,
	234416,
	234416,
	4784,
	234416,
	234416,
	4784,
	4784,
	234416,
	234416,
	4784,
	4784,
	234416,
	234416,
	4784,
	4784,
	234416,
	234416,
	4784,
	4784,
	234416,
	234416,
	4784,
	4784,
	234416,
	234416,
	4784,
	4784,
	234416,
	234416,
	4784,
	4784,
	234416,
	234416,
	4784,
	4784,
	234416,
	234416,
	4784,
	4784,
	234416,
	234416,
	4784,
	4784,
	234416,
	234416,
	4784,
	4784,
	234416,
	234416,
	4784,
	4784,
	234416,
	234416,
	4784,
	4784,
	234416,
	234416,
	4784,
};
char *AT_GraphNodeNames[59] = {
	"S2_Conv2d_16x20x1x3",
	"S3_Conv2d_16x16x1x3_Relu",
	"S4_Conv2d_16x16x1x3_Relu",
	"S5_MatAdd_16x1x299_Relu",
	"S6_Conv2d_16x16x1x3_Relu",
	"S7_Conv2d_16x16x1x3_Relu",
	"S8_MatAdd_16x1x299",
	"S9_MatAdd_16x1x299",
	"S10_Conv2d_16x16x1x3_Relu",
	"S11_Conv2d_16x16x1x3_Relu",
	"S12_MatAdd_16x1x299",
	"S13_MatAdd_16x1x299",
	"S14_Conv2d_16x16x1x3_Relu",
	"S15_Conv2d_16x16x1x3_Relu",
	"S16_MatAdd_16x1x299",
	"S17_MatAdd_16x1x299",
	"S18_Conv2d_16x16x1x3_Relu",
	"S19_Conv2d_16x16x1x3_Relu",
	"S20_MatAdd_16x1x299",
	"S21_MatAdd_16x1x299",
	"S22_Conv2d_16x16x1x3_Relu",
	"S23_Conv2d_16x16x1x3_Relu",
	"S24_MatAdd_16x1x299",
	"S25_MatAdd_16x1x299",
	"S26_Conv2d_16x16x1x3_Relu",
	"S27_Conv2d_16x16x1x3_Relu",
	"S28_MatAdd_16x1x299",
	"S29_MatAdd_16x1x299",
	"S30_Conv2d_16x16x1x3_Relu",
	"S31_Conv2d_16x16x1x3_Relu",
	"S32_MatAdd_16x1x299",
	"S33_MatAdd_16x1x299",
	"S34_Conv2d_16x16x1x3_Relu",
	"S35_Conv2d_16x16x1x3_Relu",
	"S36_MatAdd_16x1x299",
	"S37_MatAdd_16x1x299",
	"S38_Conv2d_16x16x1x3_Relu",
	"S39_Conv2d_16x16x1x3_Relu",
	"S40_MatAdd_16x1x299",
	"S41_MatAdd_16x1x299",
	"S42_Conv2d_16x16x1x3_Relu",
	"S43_Conv2d_16x16x1x3_Relu",
	"S44_MatAdd_16x1x299",
	"S45_MatAdd_16x1x299",
	"S46_Conv2d_16x16x1x3_Relu",
	"S47_Conv2d_16x16x1x3_Relu",
	"S48_MatAdd_16x1x299",
	"S49_MatAdd_16x1x299",
	"S50_Conv2d_16x16x1x3_Relu",
	"S51_Conv2d_16x16x1x3_Relu",
	"S52_MatAdd_16x1x299",
	"S53_MatAdd_16x1x299",
	"S54_Conv2d_16x16x1x3_Relu",
	"S55_Conv2d_16x16x1x3_Relu",
	"S56_MatAdd_16x1x299",
	"S57_MatAdd_16x1x299",
	"S58_Conv2d_16x16x1x3_Relu",
	"S59_Conv2d_16x16x1x3_Relu",
	"S60_MatAdd_16x1x299",
};
int quant_modelCNN(
		signed char * __restrict__ Input_1,
		signed char * __restrict__ Output_1)

{
	AT_GraphPerf[0] = gap_cl_readhwtimer();
	S2_Conv2d_16x20x1x3(
		((signed char * __restrict__) Input_1), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+0)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+24000)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+25984)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26000)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+26976)) /* Infos */
	);
	AT_GraphPerf[0] = gap_cl_readhwtimer() - AT_GraphPerf[0];
	AT_GraphPerf[1] = gap_cl_readhwtimer();
	S3_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+960)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+24064)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26016)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26032)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+26988)) /* Infos */
	);
	AT_GraphPerf[1] = gap_cl_readhwtimer() - AT_GraphPerf[1];
	AT_GraphPerf[2] = gap_cl_readhwtimer();
	S4_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+1728)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+24128)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+46820)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26048)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26064)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27000)) /* Infos */
	);
	AT_GraphPerf[2] = gap_cl_readhwtimer() - AT_GraphPerf[2];
	AT_GraphPerf[3] = gap_cl_readhwtimer();
	S5_MatAdd_16x1x299_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* In1 */
		((signed char * __restrict__) (quant_model_L2_Memory+46820)), /* In2 */
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* Out */
		((signed char * __restrict__) (quant_model_L2_Memory+27012)) /* Infos */
	);
	AT_GraphPerf[3] = gap_cl_readhwtimer() - AT_GraphPerf[3];
	AT_GraphPerf[4] = gap_cl_readhwtimer();
	S6_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+2496)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+24192)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26080)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26096)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27024)) /* Infos */
	);
	AT_GraphPerf[4] = gap_cl_readhwtimer() - AT_GraphPerf[4];
	AT_GraphPerf[5] = gap_cl_readhwtimer();
	S7_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+3264)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+24256)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26112)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26128)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27036)) /* Infos */
	);
	AT_GraphPerf[5] = gap_cl_readhwtimer() - AT_GraphPerf[5];
	AT_GraphPerf[6] = gap_cl_readhwtimer();
	S8_MatAdd_16x1x299(
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* In1 */
		((signed char * __restrict__) (quant_model_L2_Memory+46820)), /* In2 */
		((signed char * __restrict__) (quant_model_L2_Memory+42036)), /* Out */
		((signed char * __restrict__) (quant_model_L2_Memory+27048)) /* Infos */
	);
	AT_GraphPerf[6] = gap_cl_readhwtimer() - AT_GraphPerf[6];
	AT_GraphPerf[7] = gap_cl_readhwtimer();
	S9_MatAdd_16x1x299(
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* In1 */
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* In2 */
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* Out */
		((signed char * __restrict__) (quant_model_L2_Memory+27060)) /* Infos */
	);
	AT_GraphPerf[7] = gap_cl_readhwtimer() - AT_GraphPerf[7];
	AT_GraphPerf[8] = gap_cl_readhwtimer();
	S10_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+4032)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+24320)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26144)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26160)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27072)) /* Infos */
	);
	AT_GraphPerf[8] = gap_cl_readhwtimer() - AT_GraphPerf[8];
	AT_GraphPerf[9] = gap_cl_readhwtimer();
	S11_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+4800)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+24384)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26176)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26192)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27084)) /* Infos */
	);
	AT_GraphPerf[9] = gap_cl_readhwtimer() - AT_GraphPerf[9];
	AT_GraphPerf[10] = gap_cl_readhwtimer();
	S12_MatAdd_16x1x299(
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* In1 */
		((signed char * __restrict__) (quant_model_L2_Memory+42036)), /* In2 */
		((signed char * __restrict__) (quant_model_L2_Memory+46820)), /* Out */
		((signed char * __restrict__) (quant_model_L2_Memory+27096)) /* Infos */
	);
	AT_GraphPerf[10] = gap_cl_readhwtimer() - AT_GraphPerf[10];
	AT_GraphPerf[11] = gap_cl_readhwtimer();
	S13_MatAdd_16x1x299(
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* In1 */
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* In2 */
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* Out */
		((signed char * __restrict__) (quant_model_L2_Memory+27108)) /* Infos */
	);
	AT_GraphPerf[11] = gap_cl_readhwtimer() - AT_GraphPerf[11];
	AT_GraphPerf[12] = gap_cl_readhwtimer();
	S14_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+5568)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+24448)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26208)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26224)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27120)) /* Infos */
	);
	AT_GraphPerf[12] = gap_cl_readhwtimer() - AT_GraphPerf[12];
	AT_GraphPerf[13] = gap_cl_readhwtimer();
	S15_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+6336)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+24512)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26240)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26256)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27132)) /* Infos */
	);
	AT_GraphPerf[13] = gap_cl_readhwtimer() - AT_GraphPerf[13];
	AT_GraphPerf[14] = gap_cl_readhwtimer();
	S16_MatAdd_16x1x299(
		((signed char * __restrict__) (quant_model_L2_Memory+46820)), /* In1 */
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* In2 */
		((signed char * __restrict__) (quant_model_L2_Memory+42036)), /* Out */
		((signed char * __restrict__) (quant_model_L2_Memory+27144)) /* Infos */
	);
	AT_GraphPerf[14] = gap_cl_readhwtimer() - AT_GraphPerf[14];
	AT_GraphPerf[15] = gap_cl_readhwtimer();
	S17_MatAdd_16x1x299(
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* In1 */
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* In2 */
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* Out */
		((signed char * __restrict__) (quant_model_L2_Memory+27156)) /* Infos */
	);
	AT_GraphPerf[15] = gap_cl_readhwtimer() - AT_GraphPerf[15];
	AT_GraphPerf[16] = gap_cl_readhwtimer();
	S18_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+7104)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+24576)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26272)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26288)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27168)) /* Infos */
	);
	AT_GraphPerf[16] = gap_cl_readhwtimer() - AT_GraphPerf[16];
	AT_GraphPerf[17] = gap_cl_readhwtimer();
	S19_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+7872)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+24640)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26304)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26320)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27180)) /* Infos */
	);
	AT_GraphPerf[17] = gap_cl_readhwtimer() - AT_GraphPerf[17];
	AT_GraphPerf[18] = gap_cl_readhwtimer();
	S20_MatAdd_16x1x299(
		((signed char * __restrict__) (quant_model_L2_Memory+42036)), /* In1 */
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* In2 */
		((signed char * __restrict__) (quant_model_L2_Memory+46820)), /* Out */
		((signed char * __restrict__) (quant_model_L2_Memory+27192)) /* Infos */
	);
	AT_GraphPerf[18] = gap_cl_readhwtimer() - AT_GraphPerf[18];
	AT_GraphPerf[19] = gap_cl_readhwtimer();
	S21_MatAdd_16x1x299(
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* In1 */
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* In2 */
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* Out */
		((signed char * __restrict__) (quant_model_L2_Memory+27204)) /* Infos */
	);
	AT_GraphPerf[19] = gap_cl_readhwtimer() - AT_GraphPerf[19];
	AT_GraphPerf[20] = gap_cl_readhwtimer();
	S22_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+8640)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+24704)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26336)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26352)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27216)) /* Infos */
	);
	AT_GraphPerf[20] = gap_cl_readhwtimer() - AT_GraphPerf[20];
	AT_GraphPerf[21] = gap_cl_readhwtimer();
	S23_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+9408)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+24768)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26368)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26384)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27228)) /* Infos */
	);
	AT_GraphPerf[21] = gap_cl_readhwtimer() - AT_GraphPerf[21];
	AT_GraphPerf[22] = gap_cl_readhwtimer();
	S24_MatAdd_16x1x299(
		((signed char * __restrict__) (quant_model_L2_Memory+46820)), /* In1 */
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* In2 */
		((signed char * __restrict__) (quant_model_L2_Memory+51604)), /* Out */
		((signed char * __restrict__) (quant_model_L2_Memory+27240)) /* Infos */
	);
	AT_GraphPerf[22] = gap_cl_readhwtimer() - AT_GraphPerf[22];
	AT_GraphPerf[23] = gap_cl_readhwtimer();
	S25_MatAdd_16x1x299(
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* In1 */
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* In2 */
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* Out */
		((signed char * __restrict__) (quant_model_L2_Memory+27252)) /* Infos */
	);
	AT_GraphPerf[23] = gap_cl_readhwtimer() - AT_GraphPerf[23];
	AT_GraphPerf[24] = gap_cl_readhwtimer();
	S26_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+10176)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+24832)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26400)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26416)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27264)) /* Infos */
	);
	AT_GraphPerf[24] = gap_cl_readhwtimer() - AT_GraphPerf[24];
	AT_GraphPerf[25] = gap_cl_readhwtimer();
	S27_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+10944)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+24896)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+42036)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26432)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26448)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27276)) /* Infos */
	);
	AT_GraphPerf[25] = gap_cl_readhwtimer() - AT_GraphPerf[25];
	AT_GraphPerf[26] = gap_cl_readhwtimer();
	S28_MatAdd_16x1x299(
		((signed char * __restrict__) (quant_model_L2_Memory+51604)), /* In1 */
		((signed char * __restrict__) (quant_model_L2_Memory+42036)), /* In2 */
		((signed char * __restrict__) (quant_model_L2_Memory+46820)), /* Out */
		((signed char * __restrict__) (quant_model_L2_Memory+27288)) /* Infos */
	);
	AT_GraphPerf[26] = gap_cl_readhwtimer() - AT_GraphPerf[26];
	AT_GraphPerf[27] = gap_cl_readhwtimer();
	S29_MatAdd_16x1x299(
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* In1 */
		((signed char * __restrict__) (quant_model_L2_Memory+42036)), /* In2 */
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* Out */
		((signed char * __restrict__) (quant_model_L2_Memory+27300)) /* Infos */
	);
	AT_GraphPerf[27] = gap_cl_readhwtimer() - AT_GraphPerf[27];
	AT_GraphPerf[28] = gap_cl_readhwtimer();
	S30_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+11712)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+24960)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26464)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26480)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27312)) /* Infos */
	);
	AT_GraphPerf[28] = gap_cl_readhwtimer() - AT_GraphPerf[28];
	AT_GraphPerf[29] = gap_cl_readhwtimer();
	S31_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+12480)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+25024)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26496)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26512)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27324)) /* Infos */
	);
	AT_GraphPerf[29] = gap_cl_readhwtimer() - AT_GraphPerf[29];
	AT_GraphPerf[30] = gap_cl_readhwtimer();
	S32_MatAdd_16x1x299(
		((signed char * __restrict__) (quant_model_L2_Memory+46820)), /* In1 */
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* In2 */
		((signed char * __restrict__) (quant_model_L2_Memory+42036)), /* Out */
		((signed char * __restrict__) (quant_model_L2_Memory+27336)) /* Infos */
	);
	AT_GraphPerf[30] = gap_cl_readhwtimer() - AT_GraphPerf[30];
	AT_GraphPerf[31] = gap_cl_readhwtimer();
	S33_MatAdd_16x1x299(
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* In1 */
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* In2 */
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* Out */
		((signed char * __restrict__) (quant_model_L2_Memory+27348)) /* Infos */
	);
	AT_GraphPerf[31] = gap_cl_readhwtimer() - AT_GraphPerf[31];
	AT_GraphPerf[32] = gap_cl_readhwtimer();
	S34_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+13248)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+25088)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26528)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26544)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27360)) /* Infos */
	);
	AT_GraphPerf[32] = gap_cl_readhwtimer() - AT_GraphPerf[32];
	AT_GraphPerf[33] = gap_cl_readhwtimer();
	S35_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+14016)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+25152)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26560)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26576)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27372)) /* Infos */
	);
	AT_GraphPerf[33] = gap_cl_readhwtimer() - AT_GraphPerf[33];
	AT_GraphPerf[34] = gap_cl_readhwtimer();
	S36_MatAdd_16x1x299(
		((signed char * __restrict__) (quant_model_L2_Memory+42036)), /* In1 */
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* In2 */
		((signed char * __restrict__) (quant_model_L2_Memory+46820)), /* Out */
		((signed char * __restrict__) (quant_model_L2_Memory+27384)) /* Infos */
	);
	AT_GraphPerf[34] = gap_cl_readhwtimer() - AT_GraphPerf[34];
	AT_GraphPerf[35] = gap_cl_readhwtimer();
	S37_MatAdd_16x1x299(
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* In1 */
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* In2 */
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* Out */
		((signed char * __restrict__) (quant_model_L2_Memory+27396)) /* Infos */
	);
	AT_GraphPerf[35] = gap_cl_readhwtimer() - AT_GraphPerf[35];
	AT_GraphPerf[36] = gap_cl_readhwtimer();
	S38_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+14784)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+25216)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26592)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26608)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27408)) /* Infos */
	);
	AT_GraphPerf[36] = gap_cl_readhwtimer() - AT_GraphPerf[36];
	AT_GraphPerf[37] = gap_cl_readhwtimer();
	S39_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+15552)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+25280)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26624)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26640)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27420)) /* Infos */
	);
	AT_GraphPerf[37] = gap_cl_readhwtimer() - AT_GraphPerf[37];
	AT_GraphPerf[38] = gap_cl_readhwtimer();
	S40_MatAdd_16x1x299(
		((signed char * __restrict__) (quant_model_L2_Memory+46820)), /* In1 */
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* In2 */
		((signed char * __restrict__) (quant_model_L2_Memory+42036)), /* Out */
		((signed char * __restrict__) (quant_model_L2_Memory+27432)) /* Infos */
	);
	AT_GraphPerf[38] = gap_cl_readhwtimer() - AT_GraphPerf[38];
	AT_GraphPerf[39] = gap_cl_readhwtimer();
	S41_MatAdd_16x1x299(
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* In1 */
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* In2 */
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* Out */
		((signed char * __restrict__) (quant_model_L2_Memory+27444)) /* Infos */
	);
	AT_GraphPerf[39] = gap_cl_readhwtimer() - AT_GraphPerf[39];
	AT_GraphPerf[40] = gap_cl_readhwtimer();
	S42_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+16320)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+25344)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26656)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26672)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27456)) /* Infos */
	);
	AT_GraphPerf[40] = gap_cl_readhwtimer() - AT_GraphPerf[40];
	AT_GraphPerf[41] = gap_cl_readhwtimer();
	S43_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+17088)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+25408)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26688)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26704)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27468)) /* Infos */
	);
	AT_GraphPerf[41] = gap_cl_readhwtimer() - AT_GraphPerf[41];
	AT_GraphPerf[42] = gap_cl_readhwtimer();
	S44_MatAdd_16x1x299(
		((signed char * __restrict__) (quant_model_L2_Memory+42036)), /* In1 */
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* In2 */
		((signed char * __restrict__) (quant_model_L2_Memory+46820)), /* Out */
		((signed char * __restrict__) (quant_model_L2_Memory+27480)) /* Infos */
	);
	AT_GraphPerf[42] = gap_cl_readhwtimer() - AT_GraphPerf[42];
	AT_GraphPerf[43] = gap_cl_readhwtimer();
	S45_MatAdd_16x1x299(
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* In1 */
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* In2 */
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* Out */
		((signed char * __restrict__) (quant_model_L2_Memory+27492)) /* Infos */
	);
	AT_GraphPerf[43] = gap_cl_readhwtimer() - AT_GraphPerf[43];
	AT_GraphPerf[44] = gap_cl_readhwtimer();
	S46_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+17856)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+25472)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26720)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26736)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27504)) /* Infos */
	);
	AT_GraphPerf[44] = gap_cl_readhwtimer() - AT_GraphPerf[44];
	AT_GraphPerf[45] = gap_cl_readhwtimer();
	S47_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+18624)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+25536)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+42036)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26752)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26768)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27516)) /* Infos */
	);
	AT_GraphPerf[45] = gap_cl_readhwtimer() - AT_GraphPerf[45];
	AT_GraphPerf[46] = gap_cl_readhwtimer();
	S48_MatAdd_16x1x299(
		((signed char * __restrict__) (quant_model_L2_Memory+46820)), /* In1 */
		((signed char * __restrict__) (quant_model_L2_Memory+42036)), /* In2 */
		((signed char * __restrict__) (quant_model_L2_Memory+51604)), /* Out */
		((signed char * __restrict__) (quant_model_L2_Memory+27528)) /* Infos */
	);
	AT_GraphPerf[46] = gap_cl_readhwtimer() - AT_GraphPerf[46];
	AT_GraphPerf[47] = gap_cl_readhwtimer();
	S49_MatAdd_16x1x299(
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* In1 */
		((signed char * __restrict__) (quant_model_L2_Memory+42036)), /* In2 */
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* Out */
		((signed char * __restrict__) (quant_model_L2_Memory+27540)) /* Infos */
	);
	AT_GraphPerf[47] = gap_cl_readhwtimer() - AT_GraphPerf[47];
	AT_GraphPerf[48] = gap_cl_readhwtimer();
	S50_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+19392)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+25600)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26784)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26800)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27552)) /* Infos */
	);
	AT_GraphPerf[48] = gap_cl_readhwtimer() - AT_GraphPerf[48];
	AT_GraphPerf[49] = gap_cl_readhwtimer();
	S51_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+20160)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+25664)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26816)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26832)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27564)) /* Infos */
	);
	AT_GraphPerf[49] = gap_cl_readhwtimer() - AT_GraphPerf[49];
	AT_GraphPerf[50] = gap_cl_readhwtimer();
	S52_MatAdd_16x1x299(
		((signed char * __restrict__) (quant_model_L2_Memory+51604)), /* In1 */
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* In2 */
		((signed char * __restrict__) (quant_model_L2_Memory+42036)), /* Out */
		((signed char * __restrict__) (quant_model_L2_Memory+27576)) /* Infos */
	);
	AT_GraphPerf[50] = gap_cl_readhwtimer() - AT_GraphPerf[50];
	AT_GraphPerf[51] = gap_cl_readhwtimer();
	S53_MatAdd_16x1x299(
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* In1 */
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* In2 */
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* Out */
		((signed char * __restrict__) (quant_model_L2_Memory+27588)) /* Infos */
	);
	AT_GraphPerf[51] = gap_cl_readhwtimer() - AT_GraphPerf[51];
	AT_GraphPerf[52] = gap_cl_readhwtimer();
	S54_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+20928)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+25728)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26848)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26864)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27600)) /* Infos */
	);
	AT_GraphPerf[52] = gap_cl_readhwtimer() - AT_GraphPerf[52];
	AT_GraphPerf[53] = gap_cl_readhwtimer();
	S55_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+21696)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+25792)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26880)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26896)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27612)) /* Infos */
	);
	AT_GraphPerf[53] = gap_cl_readhwtimer() - AT_GraphPerf[53];
	AT_GraphPerf[54] = gap_cl_readhwtimer();
	S56_MatAdd_16x1x299(
		((signed char * __restrict__) (quant_model_L2_Memory+42036)), /* In1 */
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* In2 */
		((signed char * __restrict__) (quant_model_L2_Memory+46820)), /* Out */
		((signed char * __restrict__) (quant_model_L2_Memory+27624)) /* Infos */
	);
	AT_GraphPerf[54] = gap_cl_readhwtimer() - AT_GraphPerf[54];
	AT_GraphPerf[55] = gap_cl_readhwtimer();
	S57_MatAdd_16x1x299(
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* In1 */
		((signed char * __restrict__) (quant_model_L2_Memory+37252)), /* In2 */
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* Out */
		((signed char * __restrict__) (quant_model_L2_Memory+27636)) /* Infos */
	);
	AT_GraphPerf[55] = gap_cl_readhwtimer() - AT_GraphPerf[55];
	AT_GraphPerf[56] = gap_cl_readhwtimer();
	S58_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+22464)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+25856)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26912)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26928)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27648)) /* Infos */
	);
	AT_GraphPerf[56] = gap_cl_readhwtimer() - AT_GraphPerf[56];
	AT_GraphPerf[57] = gap_cl_readhwtimer();
	S59_Conv2d_16x16x1x3_Relu(
		((signed char * __restrict__) (quant_model_L2_Memory+27684)), /* In */
		((signed char * __restrict__) (quant_model_L2_Memory+23232)), /* Filter */
		((signed int * __restrict__) (quant_model_L2_Memory+25920)), /* Bias */
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* Out */
		((unsigned char * __restrict__) (quant_model_L2_Memory+26944)), /* Scale */
		((signed char * __restrict__) (quant_model_L2_Memory+26960)), /* ScaleN */
		((signed char * __restrict__) (quant_model_L2_Memory+27660)) /* Infos */
	);
	AT_GraphPerf[57] = gap_cl_readhwtimer() - AT_GraphPerf[57];
	AT_GraphPerf[58] = gap_cl_readhwtimer();
	S60_MatAdd_16x1x299(
		((signed char * __restrict__) (quant_model_L2_Memory+46820)), /* In1 */
		((signed char * __restrict__) (quant_model_L2_Memory+32468)), /* In2 */
		((signed char * __restrict__) Output_1), /* Out */
		((signed char * __restrict__) (quant_model_L2_Memory+27672)) /* Infos */
	);
	AT_GraphPerf[58] = gap_cl_readhwtimer() - AT_GraphPerf[58];
	return 0;
}

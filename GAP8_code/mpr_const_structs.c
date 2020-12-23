/*
* @Author: michael
* @Date:   2020-06-13 05:14:40
* @Last Modified by:   Michael Rogenmoser
* @Last Modified time: 2020-07-09 14:25:14
*/

#include "mpr_const_structs.h"

const plp_cfft_instance_q16 plp_cfft_sR_q16_len16 = {
	16, twiddleCoef_16_q16, plpBitRevIndexTable_fixed_16, PLPBITREVINDEXTABLE_FIXED_16_TABLE_LENGTH
};

RT_L1_DATA const plp_cfft_instance_q16 plp_cfft_sR_q16_len32 = {
	32, twiddleCoef_32_q16, plpBitRevIndexTable_fixed_32, PLPBITREVINDEXTABLE_FIXED_32_TABLE_LENGTH
};

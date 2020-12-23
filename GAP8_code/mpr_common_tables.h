#ifndef _MPR_COMMON_TABLES_H
#define _MPR_COMMON_TABLES_H

#include "mpr_math.h"

extern const int16_t twiddleCoef_16_q16[24];
RT_L1_DATA extern const int16_t twiddleCoef_32_q16[48];

#define PLPBITREVINDEXTABLE_FIXED_16_TABLE_LENGTH ((uint16_t)12)
#define PLPBITREVINDEXTABLE_FIXED_32_TABLE_LENGTH ((uint16_t)24)

extern const uint16_t plpBitRevIndexTable_fixed_16[PLPBITREVINDEXTABLE_FIXED_16_TABLE_LENGTH];
RT_L1_DATA extern const uint16_t plpBitRevIndexTable_fixed_32[PLPBITREVINDEXTABLE_FIXED_32_TABLE_LENGTH];

#endif /* _MPR_COMMON_TABLES_H */
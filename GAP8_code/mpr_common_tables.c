/*
* @Author: michael
* @Date:   2020-06-13 05:14:05
* @Last Modified by:   Michael Rogenmoser
* @Last Modified time: 2020-07-09 13:59:15
*/

#include "mpr_math.h"
#include "mpr_common_tables.h"

/* copied from ARM DSP Library */

/**
  @par
  Example code for q16 Twiddle factors Generation::
  @par
  <pre>fori = 0; i< 3N/4; i++)
  {
     twiddleCoefq16[2*i]   = cos(i * 2*PI/(float)N);
     twiddleCoefq16[2*i+1] = sin(i * 2*PI/(float)N);
  } </pre>
  @par
  where N = 16, PI = 3.14159265358979
  @par
  Cos and Sin values are interleaved fashion
  @par
  Convert Floating point to q15(Fixed point 1.15):
    round(twiddleCoefq16(i) * pow(2, 15))
 */
const int16_t twiddleCoef_16_q16[24] = {
    (int16_t)0x7FFF, (int16_t)0x0000,
    (int16_t)0x7641, (int16_t)0x30FB,
    (int16_t)0x5A82, (int16_t)0x5A82,
    (int16_t)0x30FB, (int16_t)0x7641,
    (int16_t)0x0000, (int16_t)0x7FFF,
    (int16_t)0xCF04, (int16_t)0x7641,
    (int16_t)0xA57D, (int16_t)0x5A82,
    (int16_t)0x89BE, (int16_t)0x30FB,
    (int16_t)0x8000, (int16_t)0x0000,
    (int16_t)0x89BE, (int16_t)0xCF04,
    (int16_t)0xA57D, (int16_t)0xA57D,
    (int16_t)0xCF04, (int16_t)0x89BE
};

/**
* \par
* Example code for q16 Twiddle factors Generation::
* \par
* <pre>for(i = 0; i< 3N/4; i++)
* {
*    twiddleCoefq16[2*i]= cos(i * 2*PI/(float)N);
*    twiddleCoefq16[2*i+1]= sin(i * 2*PI/(float)N);
* } </pre>
* \par
* where N = 32	and PI = 3.14159265358979
* \par
* Cos and Sin values are interleaved fashion
* \par
* Convert Floating point to q15(Fixed point 1.15):
*	round(twiddleCoefq16(i) * pow(2, 15))
*
*/
RT_L1_DATA const int16_t twiddleCoef_32_q16[48] = {
    (int16_t)0x7FFF, (int16_t)0x0000,
    (int16_t)0x7D8A, (int16_t)0x18F8,
    (int16_t)0x7641, (int16_t)0x30FB,
    (int16_t)0x6A6D, (int16_t)0x471C,
    (int16_t)0x5A82, (int16_t)0x5A82,
    (int16_t)0x471C, (int16_t)0x6A6D,
    (int16_t)0x30FB, (int16_t)0x7641,
    (int16_t)0x18F8, (int16_t)0x7D8A,
    (int16_t)0x0000, (int16_t)0x7FFF,
    (int16_t)0xE707, (int16_t)0x7D8A,
    (int16_t)0xCF04, (int16_t)0x7641,
    (int16_t)0xB8E3, (int16_t)0x6A6D,
    (int16_t)0xA57D, (int16_t)0x5A82,
    (int16_t)0x9592, (int16_t)0x471C,
    (int16_t)0x89BE, (int16_t)0x30FB,
    (int16_t)0x8275, (int16_t)0x18F8,
    (int16_t)0x8000, (int16_t)0x0000,
    (int16_t)0x8275, (int16_t)0xE707,
    (int16_t)0x89BE, (int16_t)0xCF04,
    (int16_t)0x9592, (int16_t)0xB8E3,
    (int16_t)0xA57D, (int16_t)0xA57D,
    (int16_t)0xB8E3, (int16_t)0x9592,
    (int16_t)0xCF04, (int16_t)0x89BE,
    (int16_t)0xE707, (int16_t)0x8275
};


const uint16_t plpBitRevIndexTable_fixed_16[PLPBITREVINDEXTABLE_FIXED_16_TABLE_LENGTH] =
{
   /* radix 4, size 12 */
   8,64, 16,32, 24,96, 40,80, 56,112, 88,104
};

RT_L1_DATA const uint16_t plpBitRevIndexTable_fixed_32[PLPBITREVINDEXTABLE_FIXED_32_TABLE_LENGTH] =
{
   /* 4x2, size 24 */
   8,128, 16,64, 24,192, 40,160, 48,96, 56,224, 72,144,
   88,208, 104,176, 120,240, 152,200, 184,232
};


/**
 * @defgroup   MATH mathematics
 *
 * @brief      This file implements mathematics.
 * 
 * Most of this was ported from arm_math library
 *
 * @author     Michael
 * @date       2020
 */

#ifndef _MPR_MATH_H
#define _MPR_MATH_H

#include <stdint.h>
#include "rt/rt_api.h"

/**
 * @defgroup math
 */

/**
 * @brief Instance structure for the fixed-point CFFT/CIFFT function.
 */
typedef struct {
	uint16_t fftLen;				/**< lenfth of the FFT. */
	const int16_t *pTwiddle;			/**< points to the Twiddle factor table. */
	const uint16_t *pBitRevTable;	/**< points to the bit reversal table. */
	uint16_t bitRevLength;			/**< bit reversal table length. */
} plp_cfft_instance_q16;

/**
 * @brief      		Processing function for the Q15 complex FFT
 * @param[in]  		S               points to an instance of the Q15 CFFT structure.
 * @param[in,out]	p1              points to the complex data buffer of size <code>2*fftLen</code>. Processing occurs in-place.
 * @param[in]  		ifftFlag        flag that selects forward (ifftFlag=0) or inverse (ifftFlag=1) transform.
 * @param[in]  		bitReverseFlag  flag that enables (bitReverseFlag=1) of disables (bitReverseFlag=0) bit reversal of output. 
 */
void plp_cfft_q16(
	const plp_cfft_instance_q16 * S,
		int16_t * p1,
		uint8_t ifftFlag,
		uint8_t bitReverseFlag);

void plp_sqrt_q16(
                           const int16_t * __restrict__ pSrc,
                           const uint32_t deciPoint,
                           int16_t * __restrict__ pRes);

void plp_cmplx_mag_q16(
	const int16_t * pSrc,
	const uint32_t deciPoint, 
	int16_t * pRes,
	uint32_t numSamples);

#endif /* _MPR_MATH_H */


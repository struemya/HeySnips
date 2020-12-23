/*
* @Author: michael
* @Date:   2020-06-13 05:13:06
* @Last Modified by:   Michael Rogenmoser
* @Last Modified time: 2020-07-10 10:42:42
*/
/**
 * @defgroup   MATH mathematics
 *
 * @brief      This file implements mathematics.
 *
 * @author     Michael
 * @date       2020
 */

/**
 * @ingroup math
 */

#define sqrt2 0b1011010100000100
#include "mpr_math.h"
#include <rt/rt_api.h>


extern void plp_cfft_radix4by2_q16(
	int16_t * pSrc,
	uint32_t fftLen,
	const int16_t * pCoef);

extern void plp_radix4_butterfly_q16(
	int16_t * pSrc16,
	uint32_t fftLen,
	int16_t * pCoef16,
	uint32_t twidCoefModifier);

/**
 * @brief      calculates the complex magnitude. 
 *
 * @param[in]  pSrc        The source
 * @param[in]  deciPoint   The decimal point. Fromat: Q(16-deciPoint).deciPoint
 * @param      pRes        The result
 * @param[in]  numSamples  The number of samples
 */

void plp_cmplx_mag_q16(
	const int16_t * pSrc,
	const uint32_t deciPoint, 
	int16_t * pRes,
	uint32_t numSamples){

	// Initial implementation, needs improvement
	int16_t real, cmplx, sqr;
	for (int i = 0; i < numSamples; i++) {

		real = (pSrc[2*i] * pSrc[2*i])>>16;
		cmplx = (pSrc[2*i + 1] * pSrc[2*i + 1])>>16;
		if (deciPoint % 2 == 0) {
			sqr = __CLIP(real + cmplx, 15);
		} else {
			sqr = __CLIP((real + cmplx) << 1, 15);
		}
		plp_sqrt_q16(&sqr, 16-deciPoint, &pRes[i]);
		pRes[i] = pRes[i] << ((16-deciPoint)/2);
	}
}

void plp_sqrt_q16(
                           const int16_t * __restrict__ pSrc,
                           const uint32_t deciPoint,
                           int16_t * __restrict__ pRes){

  int16_t number, temp1, intermediate_fixpoint, signBits, half;
  int32_t bits_val1;
  float temp_float1;
  union
  {
    int32_t fracval;
    float floatval;
  } tempconv;

  number = *pSrc;

  /* If the input is a positive number then compute the signBits. */
  if (number > 0)
    {
      signBits = __builtin_clz(number) - 17;

      /* Shift by the number of signBits */
      if ((signBits % 2) == 0)
        {
          number = number << signBits;
        }
      else
        {
          number = number << (signBits - 1);
        }

      /* Calculate half value of the number */
      half = number >> 1;
      /* Store the number for later use */
      temp1 = number;

      /* Convert to float */
      temp_float1 = number * 3.051757812500000e-005f;
      /*Store as integer */
      tempconv.floatval = temp_float1;
      bits_val1 = tempconv.fracval;
      /* Subtract the shifted value from the magic number to give intial guess */
      bits_val1 = 0x5f3759df - (bits_val1 >> 1);  /* gives initial guess */
      /* Store as float */
      tempconv.fracval = bits_val1;
      temp_float1 = tempconv.floatval;
      /* Convert to integer format */
      intermediate_fixpoint = (int32_t) (temp_float1 * 16384);

      intermediate_fixpoint = ((int16_t) ((int32_t) intermediate_fixpoint * (0x3000 -
                                       ((int16_t)
                                        ((((int16_t)
                                           (((int32_t) intermediate_fixpoint * intermediate_fixpoint) >> 15)) *
                                          (int32_t) half) >> 15))) >> 15)) << 2;
      
      intermediate_fixpoint = ((int16_t) ((int32_t) intermediate_fixpoint * (0x3000 -
                                       ((int16_t)
                                        ((((int16_t)
                                           (((int32_t) intermediate_fixpoint * intermediate_fixpoint) >> 15)) *
                                          (int32_t) half) >> 15))) >> 15)) << 2;
     
      intermediate_fixpoint = ((int16_t) ((int32_t) intermediate_fixpoint * (0x3000 -
                                       ((int16_t)
                                        ((((int16_t)
                                           (((int32_t) intermediate_fixpoint * intermediate_fixpoint) >> 15)) *
                                          (int32_t) half) >> 15))) >> 15)) << 2;

      
      intermediate_fixpoint = ((int16_t) (((int32_t) temp1 * intermediate_fixpoint) >> 15)) << 1;


      if(deciPoint > 1){
        intermediate_fixpoint = intermediate_fixpoint >> ((int32_t)(deciPoint)>>1);
        if(deciPoint%2==0){
          intermediate_fixpoint = ((int32_t)intermediate_fixpoint * sqrt2) >> 15;
        }
      }
      

      if ((signBits % 2) == 0)
        {
          intermediate_fixpoint = intermediate_fixpoint >> (signBits / 2);
        }
      else
        {
          intermediate_fixpoint = intermediate_fixpoint >> ((signBits - 1) / 2);
        }
      *pRes = intermediate_fixpoint;

    }

  else
    {
      *pRes = 0;
    }
}


/**
  @brief         In-place 16 bit reversal function.
  @param[in,out] pSrc        points to in-place buffer of unknown 16-bit data type
  @param[in]     bitRevLen   bit reversal table length
  @param[in]     pBitRevTab  points to bit reversal table
  @return        none
*/

void plp_bitreversal_16v_xpulpv2(
        uint16_t *pSrc,
  const uint16_t bitRevLen,
  const uint16_t *pBitRevTab)
{
  uint16_t i, tmp;

  v2s c;

  for (i = 0; i < bitRevLen; )
  {
     c = __SRA2(*(v2s *) & pBitRevTab[i], ((v2s){2,2}));

     //real
     tmp = pSrc[c[0]];
     pSrc[c[0]] = pSrc[c[1]];
     pSrc[c[1]] = tmp;

     //complex
     tmp = pSrc[c[0]+1];
     pSrc[c[0]+1] = pSrc[c[1]+1];
     pSrc[c[1]+1] = tmp;

    i += 2;
  }
}

/**
 * @brief      		Processing function for the q16 complex FFT
 * @param[in]  		S               points to an instance of the q16 CFFT structure.
 * @param[in,out]	p1              points to the complex data buffer of size <code>2*fftLen</code>. Processing occurs in-place.
 * @param[in]  		ifftFlag        flag that selects forward (ifftFlag=0) or inverse (ifftFlag=1) transform.
 * @param[in]  		bitReverseFlag  flag that enables (bitReverseFlag=1) of disables (bitReverseFlag=0) bit reversal of output. 
 */
void plp_cfft_q16(
	const plp_cfft_instance_q16 * S,
	int16_t * p1,
	uint8_t ifftFlag,
	uint8_t bitReverseFlag) {

	uint32_t L = S->fftLen;

	if (ifftFlag == 0) {
		switch(L){
		case 16:
        case 64:
        case 256:
        case 1024:
        case 4096:
        	plp_radix4_butterfly_q16(p1, L, (int16_t*)S->pTwiddle, 1);
        	break;
        case 32:
        case 128:
        case 512:
        case 2048:
        	plp_cfft_radix4by2_q16(p1, L, (int16_t*)S->pTwiddle);
        	break;
		}
	}
	if ( bitReverseFlag )
		plp_bitreversal_16v_xpulpv2 ((uint16_t*) p1, S->bitRevLength, S->pBitRevTable);
}

void plp_cfft_radix4by2_q16(
	int16_t * pSrc,
	uint32_t fftLen,
	const int16_t * pCoef) {

	uint32_t i;
	uint32_t n2;
	v2s pa, pb;

	uint32_t ia, l;
	int16_t xt, yt, cosVal, sinVal;
	v2s CoSi;
	v2s a, b, t;
	int16_t testa, testb;

	n2 = fftLen >> 1;

	ia = 0;
	for (i = 0; i < n2; i++)
    {
        CoSi = *(v2s *) &pCoef[ia * 2];

        ia++;

        l = i + n2;

        a = __SRA2(*(v2s *) &pSrc[2*i], ((v2s){1,1}));
        b = __SRA2(*(v2s *) &pSrc[2*l], ((v2s){1,1}));
        t = __SUB2(a,b);
        *((v2s *)&pSrc[i * 2]) = __ADD2DIV2(a, b);

        // xt = t[0];
        // yt = t[1];
        // cosVal = CoSi[0];
        // sinVal = CoSi[1];

        // pSrc[2U * l] = (((int16_t) (((int32_t) xt * cosVal) >> 16)) +
        //           ((int16_t) (((int32_t) yt * sinVal) >> 16)));

        // pSrc[2U * l + 1U] = (((int16_t) (((int32_t) yt * cosVal) >> 16)) -
        //                ((int16_t) (((int32_t) xt * sinVal) >> 16)));
        
        testa = (int16_t)(__DOTP2(t,CoSi)>>16);
        testb = (int16_t)(__DOTP2(t, __PACK2(-CoSi[1], CoSi[0]))>>16);

        *((v2s *)&pSrc[l * 2]) = __PACK2(testa, testb);
    }

    // first col
    plp_radix4_butterfly_q16( pSrc, n2, (int16_t*)pCoef, 2U);
    // second col
    plp_radix4_butterfly_q16( pSrc + fftLen, n2, (int16_t*)pCoef, 2U);

    for (i = 0; i < fftLen >> 1; i++)
    {
        pa = *(v2s *) &pSrc[4*i];
        pb = *(v2s *) &pSrc[4*i + 2];

        pa = __SLL2(pa, ((v2s){1,1}));
        pb = __SLL2(pb, ((v2s){1,1}));

        *((v2s *)&pSrc[4*i]) = pa;
        *((v2s *)&pSrc[4*i+2]) = pb;
    }
}

/*
* Radix-4 FFT algorithm used is :
*
* Input real and imaginary data:
* x(n) = xa + j * ya
* x(n+N/4 ) = xb + j * yb
* x(n+N/2 ) = xc + j * yc
* x(n+3N 4) = xd + j * yd
*
*
* Output real and imaginary data:
* x(4r) = xa'+ j * ya'
* x(4r+1) = xb'+ j * yb'
* x(4r+2) = xc'+ j * yc'
* x(4r+3) = xd'+ j * yd'
*
*
* Twiddle factors for radix-4 FFT:
* Wn = co1 + j * (- si1)
* W2n = co2 + j * (- si2)
* W3n = co3 + j * (- si3)

* The real and imaginary output values for the radix-4 butterfly are
* xa' = xa + xb + xc + xd
* ya' = ya + yb + yc + yd
* xb' = (xa+yb-xc-yd)* co1 + (ya-xb-yc+xd)* (si1)
* yb' = (ya-xb-yc+xd)* co1 - (xa+yb-xc-yd)* (si1)
* xc' = (xa-xb+xc-xd)* co2 + (ya-yb+yc-yd)* (si2)
* yc' = (ya-yb+yc-yd)* co2 - (xa-xb+xc-xd)* (si2)
* xd' = (xa-yb-xc+yd)* co3 + (ya+xb-yc-xd)* (si3)
* yd' = (ya+xb-yc-xd)* co3 - (xa-yb-xc+yd)* (si3)
*
*/

/**
 * @brief  Core function for the q16 CFFT butterfly process.
 * @param[in, out] *pSrc16          points to the in-place buffer of q16 data type.
 * @param[in]      fftLen           length of the FFT.
 * @param[in]      *pCoef16         points to twiddle coefficient buffer.
 * @param[in]      twidCoefModifier twiddle coefficient modifier that supports different size FFTs with the same twiddle factor table.
 * @return none.
 */

void plp_radix4_butterfly_q16(
	int16_t * pSrc16,
	uint32_t fftLen,
	int16_t * pCoef16,
	uint32_t twidCoefModifier)
{
	v2s R, S, T, U, V;
	v2s CoSi1, CoSi2, CoSi3, out;
	uint32_t n1, n2, ic, i0, i1, i2, i3, j, k;

	/* Total process is divided into three stages */

	/* process first stage, middle stages, & last stage */

	/* Initializations for the first stage */
	n2 = fftLen;
	n1 = n2;

	/* n2 = fftLen/4 */
	n2 >>= 2U;

	/*Index for twiddle coefficient */ 
	ic = 0U;

	/*Index for input read and output write */
	i0 = 0U;
	j = n2;

	/* Input is in 1.15(q15) format */

	/* start of first stage process */
	for(j = n2; j > 0; j--)//do
	{
		/* Butterfly implementation */

		/* index calculation for the input as, */
		/* pSrc16[i0 + 0], pSrc16[i0 + fftLen/4], pSrc16[i0 + fftLen/2], pSrc16[i0 + 3fftLen/4] */
		i1 = i0 + n2;
		i2 = i1 + n2;
		i3 = i2 + n2;

		/* Reading i0, i0+fftLen/2 inputs */

		/* input is down scale by 4 to avoid overflow */
		/* Read ya (real), xa (imag) input */
		T = __SRA2(*(v2s *) &pSrc16[i0 *2U], ((v2s){2,2}));

		/* input is down scale by 4 to avoid overflow */
    	/* Read yc (real), xc(imag) input */
		S = __SRA2(*(v2s *)&pSrc16[i2 * 2U], ((v2s){2,2}));

		/* R0 = (ya + yc) */
    	/* R1 = (xa + xc) */
		R = __ADD2(T, S);

    	/* S0 = (ya - yc) */
    	/* S1 = (xa - xc) */
		S = __SUB2(T, S);

    	/*  Reading i0+fftLen/4 , i0+3fftLen/4 inputs */
    	/* input is down scale by 4 to avoid overflow */
    	/* Read yb (real), xb(imag) input */
		T = __SRA2(*(v2s *) &pSrc16[i1 *2U], ((v2s){2,2}));

    	/* input is down scale by 4 to avoid overflow */
    	/* Read yd (real), xd(imag) input */
		U = __SRA2(*(v2s *) &pSrc16[i3 *2U], ((v2s){2,2}));

		/* T0 = (yb + yd) */
    	/* T1 = (xb + xd) */
		V = __ADD2(T, U);

    	/*  writing the butterfly processed i0 sample */
    	/* ya' = ya + yb + yc + yd */
    	/* xa' = xa + xb + xc + xd */
		*((v2s *)&pSrc16[i0 * 2U]) = __ADD2(__SRA2(R, ((v2s){1,1})), __SRA2(V, ((v2s){1,1})));

    	/* R0 = (ya + yc) - (yb + yd) */
    	/* R1 = (xa + xc) - (xb + xd) */
		R = __SUB2(R, V);

    	/* co2 & si2 are read from Coefficient pointer */
		CoSi2 = *(v2s *) &pCoef16[2U * ic * 2U];

		/*  Reading i0+fftLen/4 */
    	/* input is down scale by 4 to avoid overflow */
    	/* T0 = yb, T1 =  xb */
		// T = __SRA2(*(v2s *) &pSrc16[i1 *2U], ((v2s){2,2}));

    	/* xc' = (xa-xb+xc-xd)* co2 + (ya-yb+yc-yd)* (si2) */
    	/* yc' = (ya-yb+yc-yd)* co2 - (xa-xb+xc-xd)* (si2) */
		/* writing the butterfly processed i0 + fftLen/4 sample */
    	/* writing output(xc', yc') in little endian format */
    	*((v2s *)&pSrc16[i1 * 2U]) = __PACK2((int16_t) (__DOTP2(CoSi2, R) >> 16U), (int16_t) (__DOTP2(__PACK2(-CoSi2[1], CoSi2[0]), R) >> 16U));

    	/*  Butterfly calculations */
    	/* input is down scale by 4 to avoid overflow */
    	/* U0 = yd, U1 = xd */
		U = __SRA2(*(v2s *) &pSrc16[i3 *2U], ((v2s){2,2}));

    	/* T0 = yb-yd */
    	/* T1 = xb-xd */
		T = __SUB2(T, U);

    	/* R1 = (ya-yc) + (xb- xd),  R0 = (xa-xc) - (yb-yd)) */
		R = __ADD2(S, __PACK2(-T[1], T[0]));

    	/* S1 = (ya-yc) - (xb- xd), S0 = (xa-xc) + (yb-yd)) */
		S = __ADD2(S, __PACK2(T[1], -T[0]));

   		/* co1 & si1 are read from Coefficient pointer */
		CoSi1 = *(v2s *) &pCoef16[ ic * 2U];

   		/*  Butterfly process for the i0+fftLen/2 sample */
   		/* xb' = (xa+yb-xc-yd)* co1 + (ya-xb-yc+xd)* (si1) */
   		/* yb' = (ya-xb-yc+xd)* co1 - (xa+yb-xc-yd)* (si1) */
    	/* writing output(xb', yb') in little endian format */
		*((v2s *)&pSrc16[i2 * 2U]) = __PACK2((int16_t) (__DOTP2(CoSi1, S) >> 16U), (int16_t) (__DOTP2(__PACK2(-CoSi1[1], CoSi1[0]), S) >> 16U));

    	/* Co3 & si3 are read from Coefficient pointer */
		CoSi3 = *(v2s *) &pCoef16[3U * (ic * 2U)];

    	/*  Butterfly process for the i0+3fftLen/4 sample */
    	/* xd' = (xa-yb-xc+yd)* Co3 + (ya+xb-yc-xd)* (si3) */
    	/* yd' = (ya+xb-yc-xd)* Co3 - (xa-yb-xc+yd)* (si3) 
    	/* writing output(xd', yd') in little endian format */
		*((v2s *)&pSrc16[i3 * 2U]) = __PACK2((int16_t) (__DOTP2(CoSi3, R) >> 16U), (int16_t) (__DOTP2(__PACK2(-CoSi3[1], CoSi3[0]), R) >> 16U));

    	/*  Twiddle coefficients index modifier */
		ic = ic + twidCoefModifier;

    	/*  Updating input index */
		i0 = i0 + 1U;

	} //while (--j);
  	/* data is in 4.11(q11) format */

  	/* end of first stage process */


  	/* start of middle stage process */

  	/*  Twiddle coefficients index modifier */
	twidCoefModifier <<= 2U;

  	/*  Calculation of Middle stage */
	for (k = fftLen / 4U; k > 4U; k >>= 2U)
	{
    	/*  Initializations for the middle stage */
		n1 = n2;
		n2 >>= 2U;
		ic = 0U;

		for (j = 0U; j <= (n2 - 1U); j++)
		{
      		/*  index calculation for the coefficients */

			/*  index calculation for the coefficients */
			CoSi1 = *(v2s *) &pCoef16[ic * 2U];
			CoSi2 = *(v2s *) &pCoef16[2U * (ic * 2U)];
			CoSi3 = *(v2s *) &pCoef16[3U * (ic * 2U)];

      		/*  Twiddle coefficients index modifier */
			ic = ic + twidCoefModifier;

      		/*  Butterfly implementation */
			for (i0 = j; i0 < fftLen; i0 += n1)
			{
        		/*  index calculation for the input as, */
        		/*  pSrc16[i0 + 0], pSrc16[i0 + fftLen/4], pSrc16[i0 + fftLen/2], pSrc16[i0 + 3fftLen/4] */
				i1 = i0 + n2;
				i2 = i1 + n2;
				i3 = i2 + n2;

        		/*  Reading i0, i0+fftLen/2 inputs */
        		/* Read ya (real), xa(imag) input */
				T = *(v2s *) &pSrc16[i0 *2U];

        		/* Read yc (real), xc(imag) input */
				S = *(v2s *) &pSrc16[i2 *2U];

        		/* R0 = (ya + yc), R1 = (xa + xc) */
				R = __ADD2(T, S);

        		/* S0 = (ya - yc), S1 =(xa - xc) */
				S = __SUB2(T, S);

        		/*  Reading i0+fftLen/4 , i0+3fftLen/4 inputs */
        		/* Read yb (real), xb(imag) input */
				T = *(v2s *) &pSrc16[i1 *2U];

        		/* Read yd (real), xd(imag) input */
				U = *(v2s *) &pSrc16[i3 *2U];

        		/* T0 = (yb + yd), T1 = (xb + xd) */
				V = __ADD2(T, U);

        		/*  writing the butterfly processed i0 sample */

        		/* xa' = xa + xb + xc + xd */
        		/* ya' = ya + yb + yc + yd */
				*((v2s *)&pSrc16[i0 * 2U]) = __ADD2DIV2(__SRA2(R, ((v2s){1,1})), __SRA2(V, ((v2s){1,1})));

				/* R0 = (ya + yc) - (yb + yd), R1 = (xa + xc) - (xb + xd) */
				R = __SUB2(__SRA2(R, ((v2s){1,1})), __SRA2(V, ((v2s){1,1})));

				/*  Reading i0+3fftLen/4 */
        		/* Read yb (real), xb(imag) input */
				// T = *(v2s *) &pSrc16[i1 * 2U];

        		/* (ya-yb+yc-yd)* (si2) + (xa-xb+xc-xd)* co2 */
        		/* (ya-yb+yc-yd)* co2 - (xa-xb+xc-xd)* (si2) */
        		/*  writing the butterfly processed i0 + fftLen/4 sample */
        		/* xc' = (xa-xb+xc-xd)* co2 + (ya-yb+yc-yd)* (si2) */
        		/* yc' = (ya-yb+yc-yd)* co2 - (xa-xb+xc-xd)* (si2) */
				*((v2s *)&pSrc16[i1 * 2U]) = __PACK2((int16_t) (__DOTP2(CoSi2, R) >> 16U), (int16_t) (__DOTP2(__PACK2(-CoSi2[1], CoSi2[0]), R) >> 16U));

        		/*  Butterfly calculations */

       	 		/* Read yd (real), xd(imag) input */
				U = *(v2s *) &pSrc16[i3 * 2U];

        		/* T0 = yb-yd, T1 = xb-xd */
				T = __SRA2(__SUB2(T, U), ((v2s){1,1}));

        		/* R0 = (ya-yc) + (xb- xd), R1 = (xa-xc) - (yb-yd)) */
				R = __ADD2(__SRA2(S, ((v2s){1,1})), __PACK2(-T[1], T[0]));

        		/* S0 = (ya-yc) - (xb- xd), S1 = (xa-xc) + (yb-yd)) */
				S = __ADD2(__SRA2(S, ((v2s){1,1})), __PACK2(T[1], -T[0]));

        		/*  Butterfly process for the i0+fftLen/2 sample */
        		/* xb' = (xa+yb-xc-yd)* co1 + (ya-xb-yc+xd)* (si1) */
        		/* yb' = (ya-xb-yc+xd)* co1 - (xa+yb-xc-yd)* (si1) */
				*((v2s *)&pSrc16[i2 * 2U]) = __PACK2((int16_t) (__DOTP2(CoSi1, S) >> 16U), (int16_t) (__DOTP2(__PACK2(-CoSi1[1], CoSi1[0]), S) >> 16U));

        		/*  Butterfly process for the i0+3fftLen/4 sample */
        		/* xd' = (xa-yb-xc+yd)* Co3 + (ya+xb-yc-xd)* (si3) */
        		/* yd' = (ya+xb-yc-xd)* Co3 - (xa-yb-xc+yd)* (si3) */
				*((v2s *)&pSrc16[i3 * 2U]) = __PACK2((int16_t) (__DOTP2(CoSi3, R) >> 16U), (int16_t) (__DOTP2(__PACK2(-CoSi3[1], CoSi3[0]), R) >> 16U));

			}
		}
    	/*  Twiddle coefficients index modifier */
		twidCoefModifier <<= 2U;
	}
  	/* end of middle stage process */


  	/* data is in 10.6(q6) format for the 1024 point */
  	/* data is in 8.8(q8) format for the 256 point */
  	/* data is in 6.10(q10) format for the 64 point */
  	/* data is in 4.12(q12) format for the 16 point */

  	/*  Initializations for the last stage */
	n1 = n2;
	n2 >>= 2U;

  	/* start of last stage process */

  	/*  Butterfly implementation */
	for (i0 = 0U; i0 <= (fftLen - n1); i0 += n1)
	{
    	/*  index calculation for the input as, */
    	/*  pSrc16[i0 + 0], pSrc16[i0 + fftLen/4], pSrc16[i0 + fftLen/2], pSrc16[i0 + 3fftLen/4] */
		i1 = i0 + n2;
		i2 = i1 + n2;
		i3 = i2 + n2;

    	/*  Reading i0, i0+fftLen/2 inputs */
    	/* Read ya (real), xa(imag) input */
		T = *(v2s *) &pSrc16[i0 * 2U];

    	/* Read yc (real), xc(imag) input */
		S = *(v2s *) &pSrc16[i2 * 2U];

    	/* R0 = (ya + yc), R1 = (xa + xc) */
		R = __ADD2(T, S);

    	/* S0 = (ya - yc), S1 = (xa - xc) */
		S = __SUB2(T, S);

    	/*  Reading i0+fftLen/4 , i0+3fftLen/4 inputs */
    	/* Read yb (real), xb(imag) input */
		T = *(v2s *) &pSrc16[i1 * 2U];
    	/* Read yd (real), xd(imag) input */
		U = *(v2s *) &pSrc16[i3 * 2U];

    	/* T0 = (yb + yd), T1 = (xb + xd)) */
		T = __ADD2(T, U);

    	/*  writing the butterfly processed i0 sample */
    	/* xa' = xa + xb + xc + xd */
    	/* ya' = ya + yb + yc + yd */
		*((v2s *)&pSrc16[i0 * 2U]) = __ADD2(__SRA2(R, ((v2s){1,1})), __SRA2(T, ((v2s){1,1})));


    	/* R0 = (ya + yc) - (yb + yd), R1 = (xa + xc) - (xb + xd) */
		R = __SUB2(__SRA2(R, ((v2s){1,1})), __SRA2(T, ((v2s){1,1})));

    	/* Read yb (real), xb(imag) input */
		T = *(v2s *) &pSrc16[i1 * 2U];

    	/*  writing the butterfly processed i0 + fftLen/4 sample */
    	/* xc' = (xa-xb+xc-xd) */
    	/* yc' = (ya-yb+yc-yd) */
		*((v2s *)&pSrc16[i1 * 2U]) = R;

    	/* Read yd (real), xd(imag) input */
		U = *(v2s *) &pSrc16[i3 * 2U];

    	/* T0 = (yb - yd), T1 = (xb - xd)  */
		T = __SUB2(T, U);

		T = __SRA2(T, ((v2s){1,1}));
		S = __SRA2(S, ((v2s){1,1}));

    	/*  writing the butterfly processed i0 + fftLen/2 sample */
    	/* xb' = (xa+yb-xc-yd) */
    	/* yb' = (ya-xb-yc+xd) */
		*((v2s *)&pSrc16[i2 * 2U]) = __ADD2(S, __PACK2(T[1], -T[0]));

    	/*  writing the butterfly processed i0 + 3fftLen/4 sample */
    	/* xd' = (xa-yb-xc+yd) */
    	/* yd' = (ya+xb-yc-xd) */
		*((v2s *)&pSrc16[i3 * 2U]) = __ADD2(S, __PACK2(-T[1], T[0]));

	}

  	/* end of last stage process */

  	/* output is in 11.5(q5) format for the 1024 point */
  	/* output is in 9.7(q7) format for the 256 point   */
  	/* output is in 7.9(q9) format for the 64 point  */
  	/* output is in 5.11(q11) format for the 16 point  */
}


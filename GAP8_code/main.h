#ifndef __MAIN_H
#define __MAIN_H

#include "Gap.h"
#include "mpr_math.h"

/**
 * @brief: Data Length, i.e. number of complex 16-bit integers
 */
#define DATA_LENGTH 246

/**
 * @brief: Data Length in Bytes
 */
#define DATA_LENGTH_BYTES 984		// range * complex * data_size

/**
 * @brief: Length of FFT
 */
#define FFT_LENGTH 32

/**
 * @brief: Sequence Length
 */
#define SEQ_LENGTH 299

/**
 * @brief: Number of input features
 */
#define NUM_IN_FEAT 20

/**
 * @brief: Number of output features
 */
#define NUM_OUT_FEAT 16

/**
 * @brief: Number of output classes
 */
#define NUM_OUT_CLASSES 2


/**
 * @brief: Half of FFT length
 */
#define HALF_FFT_LENGTH 16

#define WORK_PACKET_OFFSET 31		// ceil(DATA_LENGTH/Core_count) = ceil(246/8) = ceil(30.75)

#define FFT_FIXPOINT_OUTPUT 11		// Output format Q(16-x).x of CFFT 

#define CNN_INPUT_SCALE_FACTOR	199		// 1/CNN_INPUT_SCALE in Q12.4

extern AT_HYPERFLASH_FS_EXT_ADDR_TYPE quant_model_L3_Flash;

#endif /* __MAIN_H */

/**
 * @defgroup   MAIN main
 *
 * @brief      This file implements main.
 *
 * @author     Michael Rogenmoser
 * @date       2020
 */


#include <stdio.h>
#include <rt/rt_api.h>
#include <stdint.h>
#include "main.h"
#include "mpr_math.h"
#include "mpr_const_structs.h"
#include "pmsis.h"
#include "quant_modelKernels.h"
#include "sample_data.h"

#define PERFORMANCE_PRINT
// #define GPIO_INTERRUPT
// #define SPI_COMMUNICATION
#define RUN_EMPTY
#define RUN_SAMPLE
// #define RUN_FFT


#define STACK_SIZE           2048
#define CLUSTER_PWON         1
#define CLUSTER_PWDOWN       0
#define CLUSTER_GAP          0
#define CLUSTER_FLAG_DEFAULT 0

#define TAB_SIZE 100
#define INTERVAL_US				200000

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define GPIO_INTERRUPT_PIN	19
#define GPIO_BUTTON_PIN		0
#define GPIO_ENABLE_PIN		5
#define GPIO_LED_PIN		3

AT_HYPERFLASH_FS_EXT_ADDR_TYPE quant_model_L3_Flash = 0;
//TODO if secnd model add flash here


/**
 * SPI Receive Buffer
 */
RT_L2_DATA uint8_t rx_buff[DATA_LENGTH_BYTES];

/**
 * SPI Event Scheduler
 */
L2_MEM rt_event_sched_t sched;

/**
 * SPI event flag
 * 1: data available
 * 0: waiting for data
 */
volatile int flag = 0;

L2_MEM volatile int fft_flag = 0;

RT_L2_DATA v2s input_data[DATA_LENGTH*FFT_LENGTH];
RT_L2_DATA int16_t cnn_data[DATA_LENGTH*FFT_LENGTH];

RT_L2_DATA int8_t zero_point = 0;
//first NUM_OUT_FEAT are for class 0, second NUM_OUT_FEAT are for class 1
RT_L2_DATA int8_t dense_kernel[NUM_OUT_FEAT*NUM_OUT_CLASSES] = {-64,   88,   76,  -13,  -42,  -40,  -50,  -39,   54,   16,  107,   79,  -74, -128,  -49,    8,   45,  -53,   85,   41,   75,  126,  -25,   46,  -55,    4,   20,   40,  -9,   55,   89,  -60};
RT_L2_DATA int8_t dense_bias[NUM_OUT_CLASSES] = {24 - 24};

L2_MEM unsigned char *NN_input;
L2_MEM unsigned char *NN_output;
// L2_MEM unsigned char *TCN_input;
// L2_MEM unsigned char *TCN_output;
// L2_MEM short int *DNS_output;

void run_separated_fft() {
	for (int i = rt_core_id()*WORK_PACKET_OFFSET; i < MIN(DATA_LENGTH, (rt_core_id()+1)*WORK_PACKET_OFFSET); i++) {

		// Calculate fourier transform of data
		// plp_cfft_q16(&plp_cfft_sR_q16_len32, (int16_t *) &input_data[i*FFT_LENGTH], 0, 1);

		// Calculate complex magnitude and apply fftshift
		plp_cmplx_mag_q16((int16_t *) &input_data[i*FFT_LENGTH], FFT_FIXPOINT_OUTPUT, &cnn_data[i*FFT_LENGTH+HALF_FFT_LENGTH], HALF_FFT_LENGTH);
		plp_cmplx_mag_q16((int16_t *) &input_data[i*FFT_LENGTH + HALF_FFT_LENGTH], FFT_FIXPOINT_OUTPUT, &cnn_data[i*FFT_LENGTH], HALF_FFT_LENGTH);
		for (int j = 0; j < FFT_LENGTH; j++) {
			NN_input[i*FFT_LENGTH + j] = __CLIP(((cnn_data[i*FFT_LENGTH + j]>>8)*CNN_INPUT_SCALE_FACTOR)>>6,7);
		}
	}

}

void cluster_task_CNN(){



quant_modelCNN(NN_input, NN_output);

#ifdef PERFORMANCE_PRINT
	printf("Model complete\n");

	unsigned int TotalCycles = 0, TotalOper = 0;
	printf("\n");
	for (int i = 0; i <(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
		printf("%45s: Cycles: %10d, Operations: %10d, Operations/Cycle: %f\n", AT_GraphNodeNames[i], AT_GraphPerf[i], AT_GraphOperInfosNames[i], ((float) AT_GraphOperInfosNames[i])/ AT_GraphPerf[i]);
		TotalCycles += AT_GraphPerf[i]; TotalOper += AT_GraphOperInfosNames[i];
	}
	printf("\n");
	printf("%45s: Cycles: %10d, Operations: %10d, Operations/Cycle: %f\n", "Total", TotalCycles, TotalOper, ((float) TotalOper)/ TotalCycles);
	printf("\n");

	
#endif
}

/**
 * @brief 	   Run Hyperflash constructors for NNs
 */
int construct_NN() {

	// printf("CNN Constructor\n");
	if(quant_modelCNN_Construct()) {
		printf("Constructor failed!\n");
		return 4;
	}

	return 0;

}

/**
 * @brief      Interrupt handler for SPI communication complete
 */
void irq_handle(void *arg){

	// for (int i = 0; i < 12; ++i)
	// {
	// 	printf("%02X %02X\n", rx_buff[2*i], rx_buff[2*i + 1]);
	// }
	flag = 1;
}

/**
 * @brief      Main function
 */
int main(void) {
	printf("\n---Start---\n\n");



/* -------------------------------------------------------------------------- */

	

	int data_index = 0;
		
	NN_input = (unsigned char *) AT_L2_ALLOC(0,SEQ_LENGTH*NUM_IN_FEAT); //TODO change to correct input size SEQ_LENGTH*NUM_IN_FEAT
	if(NN_input == NULL) {
		printf("Input buffer alloc error\n");
		return 2;
	}

	NN_output = (unsigned char *) AT_L2_ALLOC(0,NUM_OUT_FEAT*SEQ_LENGTH); //TODO change to correct output size NUM_OUT_FEAT*NUM_OUT_CLASSES
	if(NN_output == NULL) {
		printf("Output buffer alloc error\n");
		return 2;
	}

	/* Configure and open Cluster */
	struct pi_device cluster_dev;
	struct pi_cluster_conf cl_conf;
	cl_conf.id = 0;
	pi_open_from_conf(&cluster_dev, (void *) &cl_conf);
	if (pi_cluster_open(&cluster_dev)){
		printf("Cluster open failed\n");
		return 3;
	}

	// printf("Frequency: %i\n", rt_freq_get(RT_FREQ_DOMAIN_CL));

	if (construct_NN()){return 4;}


	printf("Initialization complete\n");

	// printf("Call cluster\n");
	struct pi_cluster_task task = {0};
	task.entry = cluster_task_CNN;
	task.arg =  NULL;
	task.stack_size = (unsigned int) STACK_SIZE;
	task.slave_stack_size = (unsigned int) 1024;



#ifdef RUN_SAMPLE
	printf("RUNNING SAMPLE\n");
	for (int i = 0; i< 1; i++){
	for (int j = 0; j < SEQ_LENGTH*NUM_IN_FEAT; j++) {
		NN_input[j] = 0;//test_cnn_data[i];
	}
#endif

	pi_cluster_send_task(&cluster_dev, &task);

#ifdef RUN_SAMPLE
	printf("%d: [", i);
	for (int k=0; k < NUM_OUT_FEAT*SEQ_LENGTH; k++) {
		//printf("%hhd, ", NN_output[k]);
	}
	printf("]\n");
	}
	
	//int offset = (SEQ_LENGTH-1) * NUM_IN_FEAT - 1 ;
	int offset = SEQ_LENGTH-1;
	printf("Offset:  %hhd\n", offset);
	int class_0 = 0;
	int class_1 = 0; 
	for (int i=0; i<NUM_OUT_FEAT; i++){
		//printf(" %hhd,",*(NN_output + offset + i*SEQ_LENGTH));
		//printf(" %hhd,",*(dense_kernel+NUM_OUT_FEAT+i));
		//int hi = *(NN_output + offset + i*SEQ_LENGTH);
		//printf(" %hhd,",hi);

		class_0 = class_0 + *(dense_kernel+i) * (*(NN_output + offset + i*SEQ_LENGTH) - zero_point);
		class_1 = class_1 + *(dense_kernel+NUM_OUT_FEAT+i) * (*(NN_output + offset + i*SEQ_LENGTH) - zero_point);
		
	}
	class_0 = class_0 + *(dense_bias);
	class_1 = class_1 + *(dense_bias+1);

	printf("\n");
	printf("Class 0:  %hhd\n", class_0);
	printf("Class 1:  %hhd\n", class_1);
	if (class_0 > class_1) {
	printf("Predict Class 0\n");
	}
	else {
	printf("Predict Class 1\n");}


#endif
 /* -------------------------------------------------------------------------- */

	quant_modelCNN_Destruct();


	pi_cluster_close(&cluster_dev);

	AT_L2_FREE(0, NN_input, SEQ_LENGTH*NUM_IN_FEAT);
	AT_L2_FREE(0, NN_output, NUM_OUT_FEAT*SEQ_LENGTH);

	printf("ended\n");

	//power off the cluster
	// rt_cluster_mount(CLUSTER_PWDOWN, CLUSTER_GAP, 0, NULL);

}

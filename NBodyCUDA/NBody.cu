#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include "random.h"
#include <omp.h>

#include "NBody.h"
#include "NBodyVisualiser.h"
#include "Nbody_d.cuh"

#define USER_NAME "aca16vb"

nbody_soa *b_soa, *d_b_soa;	            // structure with arrays of body parameters
nbody_soa *b_buff, *d_b_buff;			// body buffers
nbody *bodies, *d_bodies;				// array of bodies
float *activity, *d_activity;			// activity map arrays
float *d_a_empt, *d_a_buff;				// activity map buffers
FILE *f = NULL;							// pointer to the input file
MODE mode;								// enum mode to use
int I = 0;								// number of itterations
int N, D;								// number of bodies and dimensions for the activity map


int main(int argc, char *argv[]) {

	// Initialize random seed
	init_random();
	
	// Parse the command line arguments
	parse_args(argc, argv);

	// Allocate any heap memory
	activity = (float*)malloc((D * D) * sizeof(float));
	bodies = (nbody*)malloc(N * sizeof(nbody));
	b_soa = (nbody_soa*)malloc(sizeof(nbody_soa));
	b_buff = (nbody_soa*)malloc(sizeof(nbody_soa));

	float** b_soa_ptr[NUM_VALUES] = { &b_soa->x, &b_soa->y, &b_soa->vx, &b_soa->vy, &b_soa->m };
	float** b_buff_ptr[NUM_VALUES] = { &b_buff->x, &b_buff->y, &b_buff->vx, &b_buff->vy, &b_buff->m };
	if (mode == CUDA) {
		for (int i = 0; i < NUM_VALUES; i++) {
			cudaMalloc(b_soa_ptr[i], N * sizeof(float));
			cudaMalloc(b_buff_ptr[i], N * sizeof(float));
		}
		checkCUDAError("CUDA malloc main");
	}
	else {
		for (int i = 0; i < NUM_VALUES; i++) {
			b_soa->x = (float*)malloc(N * sizeof(float));
			b_soa->y = (float*)malloc(N * sizeof(float));
			b_soa->vx = (float*)malloc(N * sizeof(float));
			b_soa->vy = (float*)malloc(N * sizeof(float));
			b_soa->m = (float*)malloc(N * sizeof(float));
		}
	}

	// Depending on program arguments, either read initial data from file or generate random data.
	random_bodies(); // Always initialise b to random

	if (f != NULL)   // If file provided, fill in values from the file
		read_csv();

	// Convert b to Structure of Arrays (SAO):
	nbody_AOS2SOA();
	free(bodies);  // not needed anymore

	// Initialise activity map
	calc_activity();

	if (mode == CUDA) {
		nbodyCUDA();
	}
	else {
		nbodyCPU();
	}

	// Free any allocated memory
	for (int i = 0; i < NUM_VALUES; i++) {
		if (mode == CUDA) {
			cudaFree(*b_soa_ptr[i]);
			cudaFree(*b_buff_ptr[i]);
		} else {
			free(*b_soa_ptr[i]);
		}
	}
	free(b_soa);
	free(b_buff);
	free(activity);

	return 0;
}

///////////////////////////////////////////// GPU CODE ///////////////////////////////////////////////////////////

void nbodyCUDA(void) {

	cudaEvent_t start, stop;        // timers
	float elapsed, sec, milisec;    // elapsed time

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Allocate memroy on GPU
	// cudaMalloc((void**)&d_bodies, N * sizeof(nbody));
	cudaMalloc((void**)&d_b_buff, sizeof(nbody_soa));
	cudaMalloc((void**)&d_a_buff, (D * D) * sizeof(float));
	cudaMalloc((void**)&d_b_soa, sizeof(nbody_soa));
	cudaMalloc((void**)&d_activity, (D * D) * sizeof(float));
	cudaMalloc((void**)&d_a_empt, (D * D) * sizeof(float));
	checkCUDAError("CUDA malloc nbodyCUDA");

	// Copy data to device memory
	// cudaMemcpy(d_bodies, bodies, N * sizeof(nbody), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b_buff, b_buff, sizeof(nbody_soa), cudaMemcpyHostToDevice);
	cudaMemcpy(d_a_buff, activity, (D * D) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b_soa, b_soa, sizeof(nbody_soa), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_D, &D, sizeof(int));
	cudaMemcpyToSymbol(d_N, &N, sizeof(int));
	checkCUDAError("CUDA memcp nbodyCPU");

	if (I > 0) {
		cudaEventRecord(start, 0);
		for (int i = 0; i < I; i++) {
			stepCUDA();
		}
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop); // in ms
		sec = floor(elapsed / 1000.0f);
		milisec = (elapsed - sec);
		printf("Execution time %d seconds %d milliseconds \n", (int)sec, (int)milisec);
		// printf("Execution time per itteration is %.1f milliseconds \n", (elapsed / (float)I));  // Used for running experiments
	}
	else {
		initViewer(N, D, mode, &stepCUDA);
		// setNBodyPositions(d_bodies);
		setNBodyPositions2f(b_soa->x, b_soa->y);
		setActivityMapData(d_activity);
		startVisualisationLoop();
	}

	// Free any CUDA allocated memory
	cudaFree(d_b_soa);
	cudaFree(d_b_buff);
	cudaFree(d_a_empt);
	cudaFree(d_a_buff);
	cudaFree(d_activity);

	// Destroy events and reset the device
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaDeviceReset();

}

void stepCUDA() {

	// Launch Kernel
	step_kernel_shared <<< blocksPerGrid(N), threadsPerBlock(), sharedMemSize >>> (d_b_soa, d_b_buff, d_a_buff);
	checkCUDAError("CUDA kernel");
	cudaMemset(d_a_empt, 0, (D * D) * sizeof(float));
	cudaDeviceSynchronize();

	// Swap calculated activity with currently visualized:
	swap_arrays(&d_a_buff, &d_activity);
	// Swap buffer activity with empty activity:
	swap_arrays(&d_a_buff, &d_a_empt);
	setActivityMapData(d_activity);

	// Swap attributes of buffered bodies and vizualization:
	swap_bodies(&d_b_buff, &d_b_soa);
	setNBodyPositions2f(b_soa->x, b_soa->y);

}


///////////////////////////////////////////// CPU CODE ///////////////////////////////////////////////////////////

void nbodyCPU(void) {

	double start, end;				// timers
	double elapsed, sec, milisec;	// elapsed time 

	// Depending on program arguments, either configure and start the visualiser or perform a fixed number of simulation steps (then output the timing results).
	if (I > 0) {
		start = omp_get_wtime();
		for (int i = 0; i < I; i++) {
			stepCPU();
		}
		end = omp_get_wtime();
		elapsed = (end - start);
		sec = floor(elapsed);
		milisec = 1000 * (elapsed - sec);
		printf("Execution time %d seconds %d milliseconds \n", (int)sec, (int)milisec);
		printf("Execution time per itteration is %.1f milliseconds \n", 1000 * (elapsed / (float)I));  // Used for running experiments
	}
	else {
		initViewer(N, D, mode, &stepCPU);
		setNBodyPositions2f(b_soa->x, b_soa->y);
		setActivityMapData(activity);
		startVisualisationLoop();
	}

}

// Perform the main simulation of the NBody system
void stepCPU(void) {
	float2 pos;
	float2 force;
	float2 diff;
	float denom;
	int j;

#pragma omp parallel for default(none) private(j, pos, force, diff) schedule(dynamic) if(mode == OPENMP) 
	for (int i = 0; i < N; i++) {
		force = { 0.0f, 0.0f };
		pos.x = b_soa->x[i];
		pos.y = b_soa->y[i];
		for (j = 0; j < N; j++) {
			if (i != j) {

				// x and y vector difference in positions
				diff.x = b_soa->x[j] - pos.x;
				diff.y = b_soa->y[j] - pos.y;

				// bottom part of the fraction withing the summation
				denom = (diff.x * diff.x) + (diff.y * diff.y) + SOFTENING2;
				denom = sqrtf(denom * denom * denom);

				// calculate the force from body j
				force.x += (b_soa->m[j] * diff.x) / denom;
				force.y += (b_soa->m[j] * diff.y) / denom;
			}
		}

		// update position
		b_soa->x[i] += dt * b_soa->vx[i];
		b_soa->y[i] += dt * b_soa->vy[i];

		// update velocity
		b_soa->vx[i] += dt_x_G * force.x;
		b_soa->vy[i] += dt_x_G * force.y;
	}
	calc_activity();
}

void calc_activity(void) {

	int idx;
	const float val = D / (float)N;

	// Reset the act map:
	for (int r = 0; r < D * D; r++)
		activity[r] = 0.0f;

	if (mode != CUDA) {
#pragma omp parallel for default(none) private(idx) shared(activity, b_soa) if (mode == OPENMP)
		for (int n = 0; n < N; n++) {
			if (between_1(b_soa->x[n], b_soa->y[n])) {
				idx = (int)floor(D * b_soa->x[n]) + (int)(D * floor(D * b_soa->y[n]));
#pragma omp atomic
				activity[idx] += val;
			}
		}
	}
}

///////////////////////////////////////////// ARGUMENTS AND INITIALISATION ///////////////////////////////////////////////////////////

void parse_args(int argc, char* argv[]) {

	char* str_mode;	 // string for mode to use

	if (argc < MIN_ARGS) {
		fprintf(stderr, "Error: Not enough arguments passed. %d passed, should be at least %d \n\n", argc - 1, MIN_ARGS - 1);
		print_help();
		exit(1);
	}

	// Processes the command line arguments
	N = atoi(argv[1]);
	D = atoi(argv[2]);
	str_mode = argv[3];

	if (N <= 0) {
		fprintf(stderr, "Error: Incorect argument passed for N. Should be a positive integer \n\n");
		print_help();
		exit(1);
	}

	if (D <= 0) {
		fprintf(stderr, "Error: Incorrect argument passed for D. Should be a postitive integer \n\n");
		print_help();
		exit(1);
	}

	if (strcmp(str_mode, "CPU") == 0)
		mode = CPU;
	else if (strcmp(str_mode, "OPENMP") == 0)
		mode = OPENMP;
	else if (strcmp(str_mode, "CUDA") == 0)
		mode = CUDA;
	else {
		fprintf(stderr, "Error: Incorrect argument passed for M. %s passed, should be either CPU or OPENMP \n\n", str_mode);
		print_help();
		exit(1);
	}

	// Read in the optional arguments: 
	if (argc > MIN_ARGS) {
		for (int i = MIN_ARGS; i < argc; i += 2) {
			switch (argv[i][1]) {
			case 'i':
				I = atoi(argv[i + 1]);
				if (I <= 0) {
					fprintf(stderr, "Error: Incorrect argument passed for -i. Should be a positive integer \n\n");
					print_help();
					exit(1);
				}
				break;
			case 'f':
				f = fopen(argv[i + 1], "r");
				if (f == NULL) {
					fprintf(stderr, "Warning: Could not open file. Using randomly generated parameters \n");
				}
				break;
			default:
				fprintf(stderr, "Error: Inccorrect flag. Could not recognize %s flag \n\n", argv[i]);
				print_help();
				exit(1);
			}
		}
	}
}

void read_csv() {

	char line_buffer[LINE_SIZE];
	int n = 0;

	while (fgets(line_buffer, LINE_SIZE, f) != NULL) {
		if (line_buffer[0] != '#') {
			read_line(line_buffer, &bodies[n]);
			//print_body(&b[n]);
			n++;
		}
	}

	fclose(f); // Done with file

	if (n != N) {
		fprintf(stderr, "Error: Miss-match between b count. %d in the file, while %d was provided as command argument \n", n, N);
		printf("Change N to correspond to the number of b in the file \n");
		exit(1);
	}
}


void read_line(char* line, nbody *b) {

	char value_buffer[VALUE_SIZE];
	float value;

	int l = 0, v = 0, i = 0;
	float *b_ptr[NUM_VALUES] = { &b->x, &b->y, &b->vx, &b->vy, &b->m };

	while (i < NUM_VALUES) {
		if (line[l] == ',' || line[l] == '\n' || line[l] == '\0') {
			if (v > 1) {
				value_buffer[v] = '\0';
				value = strtof(value_buffer, NULL);
				memcpy(b_ptr[i], &value, sizeof(float));
			}
			i++;
			v = 0;
		}
		else {
			if (v > VALUE_SIZE) {
				fprintf(stderr, "Error: One of the provided values in the file is too long. Max lenght is %i", VALUE_SIZE);
				exit(1);
			}
			value_buffer[v] = line[l];
			v++;
		}	
		if (i < NUM_VALUES)
			l++;
	}
}

void random_bodies() {
	nbody *b;
	for (int n = 0; n < N; n++) {
		b = &bodies[n];
		b->x = random_unit_float();
		b->y = random_unit_float();
		b->vx = 0;
		b->vy = 0;
		b->m = 1.0f / (float)N;
	}
}

///////////////////////////////////////////// UTILITIES ///////////////////////////////////////////////////////////

void nbody_AOS2SOA() {

	// Temporary arrays for body attributes:
	float *t_x, *t_y, *t_vx, *t_vy, *t_m;
	t_x = (float*)malloc(N * sizeof(float));
	t_y = (float*)malloc(N * sizeof(float));
	t_vx = (float*)malloc(N * sizeof(float));
	t_vy = (float*)malloc(N * sizeof(float));
	t_m = (float*)malloc(N * sizeof(float));

	// Populate temporary arrays with values from bodies AOS
	for (int i = 0; i < N; i++) {
		t_x[i] = bodies[i].x;
		t_y[i] = bodies[i].y;
		t_vx[i] = bodies[i].vx;
		t_vy[i] = bodies[i].vy;
		t_m[i] = bodies[i].m;
	}

	// Coppy temporary vaues to relevant SOA structures:
	float *b_soa_ptr[NUM_VALUES] = { b_soa->x, b_soa->y, b_soa->vx, b_soa->vy, b_soa->m };
	float *b_buff_ptr[NUM_VALUES] = { b_buff->x, b_buff->y, b_buff->vx, b_buff->vy, b_buff->m };
	float *b_temp_ptr[NUM_VALUES] = { t_x, t_y, t_vx, t_vy, t_m };
	for (int i = 0; i < NUM_VALUES; i++) {
		if (mode == CUDA) {
			cudaMemcpy(b_soa_ptr[i], b_temp_ptr[i], sizeof(float) * N, cudaMemcpyHostToDevice);
			cudaMemcpy(b_buff_ptr[i], b_temp_ptr[i], sizeof(float) * N, cudaMemcpyHostToDevice);
			checkCUDAError("CUDA memcp nbody_AOS2SOA");
		} else {
			memcpy(b_soa_ptr[i], b_temp_ptr[i], sizeof(float) * N);
		}
		free(b_temp_ptr[i]);
	}
}

void swap_arrays(float** arr1, float** arr2) {
	float *act_temp = *arr1;
	*arr1 = *arr2;
	*arr2 = act_temp;
}

void swap_bodies(nbody_soa** body1, nbody_soa** body2) {
	nbody_soa *body_temp = *body1;
	*body1 = *body2;
	*body2 = body_temp;
}

void print_body(nbody* b) {
	printf("Body: \n");
	printf("Position - x: %f y: %f \n", b->x, b->y);
	printf("Velocity - vx: %f vy: %f \n", b->vx, b->vy);
	printf("Mass: %f \n\n", b->m);
}

void print_activity() {
	for (int i = 0; i < D; i++) {
		printf("\n");
		for (int j = 0; j < D; j++) {
			printf("%f ", activity[(i * D) + j]);
		}
	}
	printf("\n");
}

void print_help() {
	printf("nbody_%s N D M [-i I] [-i input_file] \n", USER_NAME);

	printf("where:\n");
	printf("\tN                Is the number of b to simulate.\n");
	printf("\tD                Is the integer dimension of the act grid. The Grid has D*D locations.\n");
	printf("\tM                Is the operation mode, either  'CPU' or 'OPENMP'\n");
	printf("\t[-i I]           Optionally specifies the number of simulation iterations 'I' to perform. Specifying no value will use visualisation mode. \n");
	printf("\t[-f input_file]  Optionally specifies an input file with an initial N b of data. If not specified random data will be created.\n");
}
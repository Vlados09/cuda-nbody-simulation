#ifndef KERNEL_H //ensures header is only included once
#define KERNEL_H

#include <math.h>
#include "NBody.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

// Kernel dimension definitions:
#define THREADS_PER_BLOCK 256
#define blocksPerGrid(x) (dim3 (ceilf((float)x / (float)THREADS_PER_BLOCK), 1, 1))
#define threadsPerBlock() (dim3 (THREADS_PER_BLOCK, 1, 1))
#define sharedMemSize (THREADS_PER_BLOCK * sizeof(float4))

// Wrap around from 0 if x over m: 
#define MOD(x, m) ((x < m) ? (x) : (x - m))


// Number of itterations for the fast inverse square root function
#define SQRT_PRECISION 1

// Fast inverse square root aproximation 
// Taken from: http://www.lomont.org/papers/2003/InvSqrt.pdf
__device__ float fast_rsqrt_d(float x) {
	float xhalf = 0.5f * x;
	int i = *(int*)&x;						// get bits for floating value
	i = 0x5f3759df - (i >> 1);              // gives initial guess y0
	x = *(float*)&i;                        // convert bits back to float
	for (int j = 0; j < SQRT_PRECISION; j++)
		x *= (1.5f - (xhalf * x * x));      // Newton step, repeating increases accuracy
	return x;
}

__constant__ int d_N;
__constant__ int d_D;

__device__ float2 calc_force_d(float2 b_i, float2 force) {

	extern __shared__ float4 sharedBodies[];

	float4 b_j;
	float2 diff;
	float denom;

	for (int j = 0; j < blockDim.x; j++) {

		b_j = sharedBodies[j];

		// x and y vector difference in positions
		diff.x = b_j.x - b_i.x;
		diff.y = b_j.y - b_i.y;

		// Inverse square distance calculation
		denom = (diff.x * diff.x) + (diff.y * diff.y) + SOFTENING2;
		denom = fast_rsqrt_d(denom * denom * denom);

		// Update the force
		force.x += b_j.z * diff.x * denom;
		force.y += b_j.z * diff.y * denom;
	}

	return force;
}

__device__ void calc_activity_d(float2 b_i, float* d_activity) {

	int idx;
	float val = (float)d_D / (float)d_N;

	// If body is within the range:
	if (between_1(b_i.x, b_i.y)) {
		// Get its index inside activity array
		idx = (int)floorf(d_D * b_i.x) + (int)(d_D * floorf(d_D * b_i.y));
		atomicAdd(&d_activity[idx], val);
	}

}

__global__ void step_kernel_shared(nbody_soa const* __restrict__ d_bodies_read, nbody_soa* d_bodies_write, float* d_a_buff) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ float4 sharedBodies[];

	float2 pos;
	float2 force;
	bool within = (i < d_N);

	if (within) {
		pos = { d_bodies_read->x[i], d_bodies_read->y[i] };
		force = { 0.0f, 0.0f };
	}

	int step = 0;
	// Iterate over bodies with step of block size:
	for (int s = 0; s < d_N; s += THREADS_PER_BLOCK, step++) {
		// Get shared index: 
		int s_idx = MOD(blockIdx.x + step, gridDim.x) * blockDim.x + threadIdx.x;
		if (s_idx < d_N) {
			// Only need to have shared access to poisition and mass:
			float4 b_s = { d_bodies_read->x[s_idx], d_bodies_read->y[s_idx], d_bodies_read->m[s_idx] };
			sharedBodies[threadIdx.x] = b_s;
			__syncthreads(); // Ensure all bodies are loaded into shared memory 
			if (within) {
				force = calc_force_d(pos, force);
			}
			__syncthreads(); // Ensure all threads completed force calculation
		}
	}

	if (within) {

		// update velocity
		d_bodies_write->vx[i] += dt_x_G * force.x;
		d_bodies_write->vy[i] += dt_x_G * force.y;
		
		// update position
		d_bodies_write->x[i] += dt * d_bodies_read->vx[i];
		d_bodies_write->y[i] += dt * d_bodies_read->vy[i];

		calc_activity_d(pos, d_a_buff);
	}
}

/////////////////////////////////////// PREVIOUS VERSIONS /////////////////////////////////////////////////

__global__ void step_kernel_read_only(nbody_soa const* __restrict__ d_bodies_read, nbody_soa* d_bodies_write, float* d_a_buff) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < d_N) {

		float2 pos = { d_bodies_read->x[i], d_bodies_read->y[i] };
		float2 force = { 0.0f, 0.0f };;
		float2 diff;
		float denom;
		float m_j;

		for (int j = 0; j < d_N; j++) {
			if (i != j) {
				// x and y vector difference in positions
				diff.x = d_bodies_read->x[j] - pos.x;
				diff.y = d_bodies_read->y[j] - pos.y;

				// bottom part of the fraction within the summation
				denom = (diff.x * diff.x) + (diff.y * diff.y) + SOFTENING2;
				denom = sqrtf(denom * denom * denom);

				// calculate the force from body j
				m_j = d_bodies_read->m[j];
				force.x += m_j * diff.x * denom;
				force.y += m_j * diff.y * denom;
			}
		}

		__syncthreads();

		// update position
		d_bodies_write->x[i] += dt * d_bodies_read->vx[i];
		d_bodies_write->y[i] += dt * d_bodies_read->vy[i];

		// update velocity
		d_bodies_write->vx[i] += dt_x_G * force.x;
		d_bodies_write->vy[i] += dt_x_G * force.y;

		calc_activity_d(pos, d_a_buff);
	}
}

__global__ void step_kernel_soa(nbody_soa *d_bodies_soa, float *d_a_buff) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < d_N) {

		float2 pos = { d_bodies_soa->x[i], d_bodies_soa->y[i] };
		float2 force = { 0.0f, 0.0f };;
		float2 diff;
		float denom;
		
		for (int j = 0; j < d_N; j++) {
			if (i != j) {

				// x and y vector difference in positions
				diff.x = d_bodies_soa->x[j] - pos.x;
				diff.y = d_bodies_soa->y[j] - pos.y;

				// bottom part of the fraction within the summation
				denom = (diff.x * diff.x) + (diff.y * diff.y) + SOFTENING2;
				denom = sqrtf(denom * denom * denom);

				// calculate the force from body j
				force.x += d_bodies_soa->m[j] * diff.x * denom;
				force.y += d_bodies_soa->m[j] * diff.y * denom;
			}
		}

		__syncthreads();

		// update position
		d_bodies_soa->x[i] += dt * d_bodies_soa->vx[i];
		d_bodies_soa->y[i] += dt * d_bodies_soa->vy[i];

		// update velocity
		d_bodies_soa->vx[i] += dt_x_G * force.x;
		d_bodies_soa->vy[i] += dt_x_G * force.y;

		calc_activity_d(pos, d_a_buff);
	}

}

__global__ void step_kernel_baseline(nbody* d_bodies, float* d_a_buff) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < d_N) {

		nbody* b_i, * b_j;
		float a_x, a_y;
		float sum_x, sum_y;
		float diff_x, diff_y;
		float denom;
		int idx;

		float val = (float)d_D / (float)d_N;

		b_i = &d_bodies[i];
		sum_x = 0, sum_y = 0;
		for (int j = 0; j < d_N; j++) {
			if (i != j) {
				b_j = &d_bodies[j];

				// x and y vector difference in positions
				diff_x = b_j->x - b_i->x;
				diff_y = b_j->y - b_i->y;

				// Bottom part of the fraction within the summation
				denom = (diff_x * diff_x) + (diff_y * diff_y) + SOFTENING2;
				denom = sqrtf(denom * denom * denom);

				// Calculate the force from body j
				sum_x += (b_j->m * diff_x) / denom;
				sum_y += (b_j->m * diff_y) / denom;
			}
		}

		// Scale force by constant to get acceleration
		a_x = G * sum_x;
		a_y = G * sum_y;

		// Update position
		b_i->x += dt * b_i->vx;
		b_i->y += dt * b_i->vy;

		// Update velocity
		b_i->vx += dt * a_x;
		b_i->vy += dt * a_y;

		// Update activity map:
		if (between_1(b_i->x, b_i->y)) {
			idx = (int)floorf(d_D * b_i->x) + (int)(d_D * floorf(d_D * b_i->y));
			atomicAdd(&d_a_buff[idx], val);
		}
	}

}

#endif //KERNEL_H
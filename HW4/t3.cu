#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math_constants.h>

extern "C" 
{

__device__ inline float rexp(float lambda, curandState *state)
// Exponential distribution sampler based on inverse CDF method 
{
    if (lambda < 0)
        return(999);
    float u = curand_uniform(state);
    if (u < 1e-10)
        u = 1e-10;
    else if ( u > 1 - 1e-10)
        u = 1 - 1e-10;
    return(-log(1 - u) / lambda);
}

__device__ inline float rtruncnorm_std(float lo, float hi, curandState *state) 
// This function output one truncated normal disbution 
// from TN(0, 1, lo, hi)
{
    float prop=0;
	int sign_lo = 1;
	// Convert the cases to one suitable for lo > 0 and hi > 0
	if (lo < 0 && hi < 0 || isinf(lo) && hi <=0) {
		sign_lo = -1;
		float tmp = lo;
		lo = -hi;
		hi = -tmp;
	}
	// alpha^* in Robert 1995
	float thresh = lo + 2*sqrt(M_E) / (lo+sqrt(lo*lo+4)) * exp((lo*lo-lo*sqrt(lo*lo+4))/4);
	
    if (lo * hi <= 0 && hi - lo > 2 || isinf(hi) && lo <= 0) 
    // Normal proposal rejection sampling
        for (prop=curand_normal(state); prop <= lo || prop > hi; prop=curand_normal(state));
		
    else if (lo * hi > 0 && hi > thresh)
	// Shifted exponential proposal rejection sampling
	{
		float lambda = (lo + sqrt(lo * lo + 4)) / 2;
		for (prop=lo + rexp(lambda, state); !(prop <= hi && curand_uniform(state) < exp(-(prop-lambda)*(prop-lambda) / 2)); prop=lo + rexp(lambda, state));
	}
	
    else 
    // Uniform proposal rejection sampling
	{
		// The maximal density location of a truncated standard normal  
        float x_max;
        if (lo * hi <= 0)
            x_max = 0;
        else 
            x_max = lo;
       	// Calculate the density ratio	
        float Mg = 1 / sqrt(2 * M_PI) * exp(-x_max * x_max / 2) ;
        for (prop=lo + (hi - lo) * curand_uniform(state); curand_uniform(state) > 1 / sqrt(2 * M_PI) * exp(-prop * prop / 2) / Mg; prop=lo + (hi - lo) * curand_uniform(state));
    }
	
	// Remember to convert the positive sample back to negative 
	// if neccesary
    prop *= sign_lo;
    return(prop);
}

 __global__ void 
rtruncnorm_kernel(float *vals, int n, 
                  float *mu, float *sigma, 
                  float *lo, float *hi,
                  int mu_len, int sigma_len,
                  int lo_len, int hi_len,
                  curandState *state, int n_seed)
// n_seed is to limit the number of seeds (different curandState's) 
// used, in order to save memory and speed up.
{
    // Usual block/thread indexing...
    int myblock = blockIdx.x + blockIdx.y * gridDim.x;
    int blocksize = blockDim.x * blockDim.y * blockDim.z;
    int subthread = threadIdx.z*(blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	// Seed number is set according to the intial index of the thread
    int idx_init = myblock * blocksize + subthread;
    
    for (int idx = idx_init; idx < n; idx += n_seed) 
	// idx indexes the number of sample
	{
        // Recycle if lenght of vectors is 1
        float mu_0, sigma_0, lo_0, hi_0;
        if (mu_len == 1)
            mu_0 = mu[0];
        else
            mu_0 = mu[idx];
			
        if (sigma_len == 1)
            sigma_0 = sigma[0];
        else
            sigma_0 = sigma[idx];
			
        if (lo_len == 1)
            lo_0 = lo[0];
        else 
            lo_0 = lo[idx];
			
        if (hi_len == 1)
            hi_0 = hi[0];
        else
            hi_0 = hi[idx];

        float lo_std = (lo_0 - mu_0) / sigma_0;
        float hi_std = (hi_0 - mu_0) / sigma_0;

        // Sample: note the index of the state here
        vals[idx] = rtruncnorm_std(lo_std, hi_std, &state[idx_init]) * sigma_0 + mu_0;
    }
    return;
}


__global__ void setup_kernel(curandState *state, int n_seed)
// Initialize the random states
{
    int myblock = blockIdx.x + blockIdx.y * gridDim.x;
    int blocksize = blockDim.x * blockDim.y * blockDim.z;
    int subthread = threadIdx.z*(blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
    int idx = myblock * blocksize + subthread;
    if (idx < n_seed)
        curand_init(9131 + idx*19, idx, 0, &state[idx]);
    return;
}


} // End extern "C"

// auxilary functions for running apps

#ifndef __MISC_H__
#define __MISC_H__
#include <stdio.h>

static void HandleError(cudaError_t err, const char* file, int line)
{
	if(err != cudaSuccess)
	{
		printf("%s in %s at line %d \n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
#define HANDLE_NULL(a) {if(a == NULL){ \
							printf("Host memory failed in %s at line %d \n", \
									__FILE, __LINE__); \
									exit(EXIT_FAILURE);}}

__device__ unsigned char value(float n1, float n2, int hue)
{
	if(hue > 360)
		hue -= 360;
	else if(hue < 0)
		hue += 360;

	if(hue < 60)
		return (unsigned char)(255 * (n1 + (n2 - n1) * hue / 60));
	if(hue < 180)
		return (unsigned char)(255 * n2);
	if(hue < 240)
		return (unsigned char)(255 * (n1 + (n2 - n1) * (240 - hue) / 60));
	return (unsigned char)(255 * n1);

}

__global__ void float_to_color(unsigned char* optr, const float* outSrc)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	
	float l = outSrc[offset];
	float s = 1;
	int h = (180 + (int)(360.0f * outSrc[offset])) % 360;
	float m1, m2;

	if( l <= 0.5f)
		m2 = l * (s + 1);
	else
		m2 = l + s - l * s;
	m1 = l * 2 - m2;

	optr[offset*4 + 0] = value(m1, m2, h + 120);
	optr[offset*4 + 1] = value(m1, m2, h);
	optr[offset*4 + 2] = value(m1, m2, h - 120);
	optr[offset*4 + 3] = 255;

}


#endif

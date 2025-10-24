#pragma once

#include <stdio.h>


// Host-side assertion (CPU)
#define ASSERTC(expression) \
do { \
if (!(expression)) { \
fprintf(stderr, "Host assertion failed: (%s), function %s, file %s, line %d.\n", \
#expression, __func__, __FILE__, __LINE__); \
fflush(stderr); \
*(volatile int*)0 = 0; \
} \
} while(0)


#define CHK(call) \
do { \
cudaError_t err = call; \
if (err != cudaSuccess) { \
fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
cudaGetErrorString(err)); \
ASSERTC(0); \
} \
} while(0)
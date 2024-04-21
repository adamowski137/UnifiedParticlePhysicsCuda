#pragma once
#include <cuda_runtime.h>


// world constants
#define MAXDIMENSION 50.0f
#define MINDIMENSION -50.0f

// particle constants
#define PARTICLERADIUS 2.f
#define MAXPARTICLES 8192

// grid constants
#define CUBESIZE (PARTICLERADIUS * 5)
#define CUBESPERDIMENSION (int)((MAXDIMENSION - MINDIMENSION) / CUBESIZE)
#define TOTALCUBES (int) (CUBESPERDIMENSION * CUBESPERDIMENSION * CUBESPERDIMENSION)

//namespace CUDAConstants
//{
//	__constant__ float const_invmass[MAXPARTICLES];
//}
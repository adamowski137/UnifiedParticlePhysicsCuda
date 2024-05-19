#pragma once

// world constants
#define MAXDIMENSION 50.0f
#define MINDIMENSION -50.0f

// particle constants
#define PARTICLERADIUS 1.f

// grid constants
#define CUBESIZE (PARTICLERADIUS * 5)
#define CUBESPERDIMENSION (int)((MAXDIMENSION - MINDIMENSION) / CUBESIZE)
#define TOTALCUBES (int) (CUBESPERDIMENSION * CUBESPERDIMENSION * CUBESPERDIMENSION)

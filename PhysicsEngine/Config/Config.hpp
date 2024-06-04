#pragma once
#include <string>

struct EngineConfig 
{
public:
	float K_DISTANCE_CONSTRAINT_COLLISION;
	float K_DISTANCE_CONSTRAINT_CLOTH_BENDING;
	float K_DISTANCE_CONSTRAINT_CLOTH_STRETCHING;
	float K_SURFACE_CONSTRAINT;

	static EngineConfig readConfig(std::string path);
};

class GlobalEngineConfig 
{
public:
	static EngineConfig config;
};



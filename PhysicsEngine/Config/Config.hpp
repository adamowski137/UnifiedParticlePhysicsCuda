#pragma once

class EngineConfig 
{
public:
	static float K_DISTANCE_CONSTRAINT_COLLISION;
	static float K_DISTANCE_CONSTRAINT_CLOTH_BENDING;
	static float K_DISTANCE_CONSTRAINT_CLOTH_STRETCHING;
	static float K_SURFACE_CONSTRAINT;

	static void readConfig();
};
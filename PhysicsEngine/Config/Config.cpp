#include "Config.hpp"
#include "fstream"

float EngineConfig::K_DISTANCE_CONSTRAINT_COLLISION;
float EngineConfig::K_DISTANCE_CONSTRAINT_CLOTH_BENDING;
float EngineConfig::K_DISTANCE_CONSTRAINT_CLOTH_STRETCHING;
float EngineConfig::K_SURFACE_CONSTRAINT;

void EngineConfig::readConfig()
{
	std::ifstream f("../../../../PhysicsEngine/Config/engine_config.txt");
	f >> K_DISTANCE_CONSTRAINT_COLLISION;
	f >> K_DISTANCE_CONSTRAINT_CLOTH_STRETCHING;
	f >> K_DISTANCE_CONSTRAINT_CLOTH_BENDING;
	f >> K_SURFACE_CONSTRAINT;
	f.close();
}

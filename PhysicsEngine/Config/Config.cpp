#include "Config.hpp"
#include "fstream"

EngineConfig GlobalEngineConfig::config;

EngineConfig EngineConfig::readConfig(std::string path)
{
	EngineConfig config;
	std::ifstream f(path);
	f >> config.K_DISTANCE_CONSTRAINT_COLLISION;
	f >> config.K_DISTANCE_CONSTRAINT_CLOTH_STRETCHING;
	f >> config.K_DISTANCE_CONSTRAINT_CLOTH_BENDING;
	f >> config.K_SURFACE_CONSTRAINT;
	f.close();

	return config;
}

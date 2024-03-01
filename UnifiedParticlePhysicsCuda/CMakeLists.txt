cmake_minimum_required(VERSION 3.8)

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/UnifiedParticlePhysicsCuda/*.cpp")
#file(GLOB gpu_source_files)

add_executable (UnifiedParticlePhysicsCuda "${CMAKE_SOURCE_DIR}/UnifiedParticlePhysicsCuda/UnifiedParticlePhysicsCuda.cpp")
enable_language("CUDA")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET UnifiedParticlePhysicsCuda PROPERTY CXX_STANDARD 20)
  set_property(TARGET UnifiedParticlePhysicsCuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/GUI gui)
add_subdirectory(${CMAKE_SOURCE_DIR}/PhysicsEngine PhysicsEngine)

target_link_libraries(UnifiedParticlePhysicsCuda PUBLIC gui)
target_link_libraries(UnifiedParticlePhysicsCuda PUBLIC PhysicsEngine)

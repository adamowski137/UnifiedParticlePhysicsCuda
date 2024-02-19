# CMakeList.txt : CMake project for UnifiedParticlePhysicsCuda, include source and define
# project specific logic here.
#

# Add source to this project's executable.
file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/UnifiedParticlePhysicsCuda/*.cpp")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/UnifiedParticlePhysicsCuda/*.cu")


add_executable (UnifiedParticlePhysicsCuda ${cpu_source_files} ${gpu_source_files})
enable_language("CUDA")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET UnifiedParticlePhysicsCuda PROPERTY CXX_STANDARD 20)
  set_property(TARGET UnifiedParticlePhysicsCuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

find_package(CUDAToolkit)
target_link_libraries(UnifiedParticlePhysicsCuda CUDA::cudart)
# TODO: Add tests and install targets if needed.

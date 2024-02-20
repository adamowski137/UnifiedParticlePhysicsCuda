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

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_library(
	logic
	${CMAKE_SOURCE_DIR}/Logic/Particle.cuh
	${CMAKE_SOURCE_DIR}/Logic/Particle.cu
)


find_package(OpenGL REQUIRED)

add_subdirectory(${CMAKE_SOURCE_DIR}/External/glfw-3.3.9 glfw)
add_subdirectory(${CMAKE_SOURCE_DIR}/External/glad glad)
target_link_libraries(logic CUDA::cudart glad)

add_library(
	window
	${CMAKE_SOURCE_DIR}/Window/Window.hpp
	${CMAKE_SOURCE_DIR}/Window/Window.cpp	
)

set(SOURCES UnifiedParticlePhysics/UnifiedParticlePhysics.cpp)

target_include_directories(UnifiedParticlePhysics PUBLIC src)
target_include_directories(window PUBLIC src)

target_link_libraries(window PUBLIC glfw glad OpenGL::GL)

target_link_libraries(UnifiedParticlePhysicsCuda PUBLIC window)
target_link_libraries(UnifiedParticlePhysicsCuda PUBLIC CUDA::cudart)
# TODO: Add tests and install targets if needed.

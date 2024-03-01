cmake_minimum_required(VERSION 3.8)

project(PhysicsEngine)

add_library(PhysicsEngine STATIC
	${CMAKE_CURRENT_SOURCE_DIR}/Constrain/Constrain.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Constrain/DistanceConstrain/DistanceConstrain.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Constrain/FloorConstrain/FloorConstrain.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Math/ConstrainSolver.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Math/LinearSolver.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Particle/Particle.cu
)


if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET PhysicsEngine PROPERTY CXX_STANDARD 20)
  set_property(TARGET PhysicsEngine PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  endif()

set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")

find_package(CUDAToolkit)

target_link_libraries(PhysicsEngine PUBLIC CUDA::cudart glad)

cmake_minimum_required(VERSION 3.8)

project(PhysicsEngine)

add_library(PhysicsEngine STATIC
	${CMAKE_CURRENT_SOURCE_DIR}/Constrain/Constrain.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Collision/CollisionGrid.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Constrain/SurfaceConstraint/SurfaceConstraint.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Collision/SurfaceCollisionFinder.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Constrain/DistanceConstrain/DistanceConstrain.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Math/ConstrainSolver.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Math/LinearSolver.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Particle/Particle.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Particle/ParticleData.cu
	${CMAKE_CURRENT_SOURCE_DIR}/List/List.cu
)


if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET PhysicsEngine PROPERTY CXX_STANDARD 20)
  set_property(TARGET PhysicsEngine PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET PhysicsEngine PROPERTY CMAKE_POSITION_INDEPENDENT_CODE ON)
  endif()

set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "--expt-extended-lambda -g -G --rdc=true")

find_package(CUDAToolkit)
find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

target_compile_options(PhysicsEngine PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
					   --expt-extended-lambda
                       >)

target_link_libraries(PhysicsEngine PUBLIC CUDA::cudart glad Thrust)

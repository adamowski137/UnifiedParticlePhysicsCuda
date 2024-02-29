# CMakeList.txt : CMake project for UnifiedParticlePhysicsCuda, include source and define
# project specific logic here.
#

# Add source to this project's executable.
file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/UnifiedParticlePhysicsCuda/*.cpp"
)
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/UnifiedParticlePhysicsCuda/*.cu"
	"${CMAKE_SOURCE_DIR}/UnifiedParticlePhysicsCuda/Math/*.cu"
	"${CMAKE_SOURCE_DIR}/UnifiedParticlePhysicsCuda/Particle/*.cu"
	"${CMAKE_SOURCE_DIR}/UnifiedParticlePhysicsCuda/Constrain/FloorConstrain/*.cu"
	"${CMAKE_SOURCE_DIR}/UnifiedParticlePhysicsCuda/Constrain/*.cu"
	"${CMAKE_SOURCE_DIR}/UnifiedParticlePhysicsCuda/Constrain/DistanceConstrain/*.cu"

)


add_executable (UnifiedParticlePhysicsCuda ${cpu_source_files} ${gpu_source_files} )
enable_language("CUDA")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET UnifiedParticlePhysicsCuda PROPERTY CXX_STANDARD 20)
  set_property(TARGET UnifiedParticlePhysicsCuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  endif()

  set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")

find_package(CUDAToolkit)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)



find_package(OpenGL REQUIRED)

add_subdirectory(${CMAKE_SOURCE_DIR}/GUI gui)

target_link_libraries(UnifiedParticlePhysicsCuda PUBLIC CUDA::cudart)
target_link_libraries(UnifiedParticlePhysicsCuda PUBLIC gui)
# TODO: Add tests and install targets if needed.
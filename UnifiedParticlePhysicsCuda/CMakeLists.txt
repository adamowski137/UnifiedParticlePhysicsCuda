﻿cmake_minimum_required(VERSION 3.8)

add_executable (UnifiedParticlePhysicsCuda 
		${CMAKE_CURRENT_SOURCE_DIR}/UnifiedParticlePhysicsCuda.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/Scene/Scene.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/Scene/TestScene/TestScene.cu
		${CMAKE_CURRENT_SOURCE_DIR}/Scene/Scene_Covering/Scene_Covering.cu
		${CMAKE_CURRENT_SOURCE_DIR}/Scene/Scene_Trampoline/Scene_Trampoline.cu
		${CMAKE_CURRENT_SOURCE_DIR}/Scene/Scene_External/Scene_External.cu
		${CMAKE_CURRENT_SOURCE_DIR}/Scene/Scene_Friction/Scene_Friction.cu
		${CMAKE_CURRENT_SOURCE_DIR}/Scene/Cloth_Scene/Cloth_Scene.cu
		${CMAKE_CURRENT_SOURCE_DIR}/Scene/RigidBody_Scene/Scene_RigidBody.cu
		${CMAKE_CURRENT_SOURCE_DIR}/ResourceManager/ResourceManager.cpp
)
enable_language("CUDA")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET UnifiedParticlePhysicsCuda PROPERTY CXX_STANDARD 20)
  set_property(TARGET UnifiedParticlePhysicsCuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/GUI gui)
add_subdirectory(${CMAKE_SOURCE_DIR}/PhysicsEngine PhysicsEngine)

target_link_libraries(UnifiedParticlePhysicsCuda PUBLIC gui PhysicsEngine)

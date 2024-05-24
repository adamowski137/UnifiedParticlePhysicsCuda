cmake_minimum_required (VERSION 3.8)

project(gui)

file(GLOB imgui_source_files "${CMAKE_CURRENT_SOURCE_DIR}/imgui/*.cpp" 
"${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends/*.cpp")

add_library(gui STATIC
	${CMAKE_CURRENT_SOURCE_DIR}/Camera/Camera.cpp	
	${CMAKE_CURRENT_SOURCE_DIR}/Error/ErrorHandling.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Renderer/MeshGenerator.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Renderer/Renderer.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Renderer/ParticleRenderer.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Renderer/ClothRenderer.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Renderer/RenderInfo.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Shader/Shader.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Window/Window.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Input/KeyInput.cpp
	${imgui_source_files}
)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

find_package(OpenGL REQUIRED)

add_subdirectory(${CMAKE_SOURCE_DIR}/External/glfw-3.3.9 glfw)
add_subdirectory(${CMAKE_SOURCE_DIR}/External/glad glad)
add_subdirectory(${CMAKE_SOURCE_DIR}/External/glm glm)

target_include_directories(gui PUBLIC ${CMAKE_SOURCE_DIR}/External/imgui)
target_link_directories(gui PUBLIC ${CMAKE_SOURCE_DIR}/External/imgui)

target_link_libraries(gui PUBLIC glfw glad OpenGL::GL glm)
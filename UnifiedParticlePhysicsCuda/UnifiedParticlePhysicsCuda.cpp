// UnifiedParticlePhysicsCuda.cpp : Defines the entry point for the application.
//

#include "UnifiedParticlePhysicsCuda.h"
#include "Window/Window.hpp"
#include "App/App.hpp"

using namespace std;

int main()
{
	/*int A[4] = { 1, 1, 1, 1 };
	int B[4] = { 1, 2, 3, 4 };
	addWithCuda(A, A, B, 4);
	for (int i = 0; i < 4; i++)
		std::cout << A[i] << "\n";
	cout << "Hello CMake." << endl;*/

	// init glfw, glad
	Window::getInstance();

	App app(1024, 768);
	while (!Window::getInstance().isClosed())
	{
		app.update();

		Window::getInstance().clear(255, 255, 255, 1);
		app.draw();
		Window::getInstance().finishRendering();
	}
	return 0;
}

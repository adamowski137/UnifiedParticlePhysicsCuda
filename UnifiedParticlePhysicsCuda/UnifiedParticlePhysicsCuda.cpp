﻿// UnifiedParticlePhysicsCuda.cpp : Defines the entry point for the application.
//

#include "UnifiedParticlePhysicsCuda.h"
#include "Window/Window.hpp"
using namespace std;

int main()
{
	/*int A[4] = { 1, 1, 1, 1 };
	int B[4] = { 1, 2, 3, 4 };
	addWithCuda(A, A, B, 4);
	for (int i = 0; i < 4; i++)
		std::cout << A[i] << "\n";
	cout << "Hello CMake." << endl;*/

	Window::getInstance().runWindow();
	return 0;
}

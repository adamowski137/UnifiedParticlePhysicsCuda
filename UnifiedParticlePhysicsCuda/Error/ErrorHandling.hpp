#pragma once

void glClearError();
bool glCheckError();


#define Call(x) x; \
				glCheckError()
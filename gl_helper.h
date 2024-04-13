// base syste: ubuntu 22.04.1


#ifndef	__GL_HELPER_H__
#define __GL_HELPER_H__

#include <GL/glut.h>
#include <GL/glext.h>
#include <GL/glx.h>

#define GET_PROC_ADDRESS( str ) glXGetProcAddress( (const GLubyte*)str )


#endif

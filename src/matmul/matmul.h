/* Matmul CL-GL benchmark */

#include <stdio.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

void initialize(int argc, char* argv[]);
void init_window(int argc, char **argv);
void resize_function(int Width, int Height);
void render_function(void);
void idle_function();
void timer_function(int value);
void cleanup_function(void);
void create_vbo(void);
void destroy_vbo(void);
void create_shaders(void);
void destroy_shaders(void);

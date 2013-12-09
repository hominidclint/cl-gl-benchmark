AC_INIT([clglbenchmark], [0.1])
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE([-Wall -Werror])
AM_PROG_AS
AC_PROG_CC
AC_PROG_CXX
AC_PROG_INSTALL

# Silent rules if supported
ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

# Libtool
LT_INIT
AC_PROG_LIBTOOL

###############
# Debug/Profile
###############

debug=false
CFLAGS="-msse2 -O3 -std=gnu99 -DNDEBUG"
CXXFLAGS="-std=gnu++0x -O3 -DNDEBUG"
AC_ARG_ENABLE(debug,
[  --enable-debug    Turn on debugging],
[
	case "${enableval}" in
	yes)
		debug=true
		
		# Add flag '-Wno-error=deprecated-declarations' only if it is supported.
		# This flag prevents warnings caused by newer GTK versions that would cause with -Werror
		CXXFLAGS="-msse2 -std=gnu++0x -Wall -Werror -g -O0"
		saved_flags="-msse2 -Wall -Werror -DMHANDLE -g -O0 -std=gnu99"
		CFLAGS="-Wno-error=deprecated-declarations"
		AC_MSG_CHECKING([whether CC supports -Wno-error=deprecated-declarations])
		AC_COMPILE_IFELSE([AC_LANG_PROGRAM([])],
				[ AC_MSG_RESULT([yes]) ]
				[ saved_flags="$saved_flags -Wno-error=deprecated-declarations" ],
				[ AC_MSG_RESULT([no]) ]
			)
		CFLAGS="$saved_flags"
		;;
	no)
		;;
	*)
		AC_MSG_ERROR(bad value ${enableval} for --enable-debug)
		;;
	esac
],
[])

profile=false
AC_ARG_ENABLE(profile,
[  --enable-profile    Turn on profiling],
[
	case "${enableval}" in
	yes)
		CFLAGS+=" -pg"
		profile=true
		;;
	no)
		;;
	*)
		AC_MSG_ERROR(bad value ${enableval} for --enable-profile)
		;;
	esac
],
[])

AC_ARG_ENABLE(no-strict-aliasing,
[  --enable-no-strict-aliasing    Allow for no strict aliasing],
[
	case "${enableval}" in
	yes)
		CFLAGS+=" -fno-strict-aliasing"
		;;
	no)
		;;
	*)
		AC_MSG_ERROR(bad value ${enableval} for --enable-no-strict-aliasing)
		;;
	esac
],
[])
AC_SUBST([CFLAGS])


##################
# Test AMD OpenCL
##################
have_amd_opencl=no

if test "x$AMDAPPSDKROOT" != "x"
then
	# Use temporary CFLAGS, LDFLAGS
	saved_cflags="$CFLAGS"
	saved_ldflags="$LDFLAGS"
	saved_libs="$LIBS"
	CFLAGS="-I$AMDAPPSDKROOT/include"
	LDFLAGS="-L$AMDAPPSDKROOT/lib/x86 -L$AMDAPPSDKROOT/lib/x86_64"
	LIBS="-lOpenCL"

	# Try program
	AC_LINK_IFELSE(
	[
		AC_LANG_PROGRAM(
			[[#include <CL/cl.h>]],
			[[#include <CL/cl_gl.h>]],
			[[
				cl_int ret;
				cl_platform_id platform;
				cl_uint num_platforms;
				ret = clGetPlatformIDs(1, &platform, &num_platforms);
			        return CL_SUCCESS;
			]]
		)
	],
	[
		have_amd_opencl=yes
	])

	# Restore original CFLAGS, LDFLAGS, LIBS
	CFLAGS="$saved_cflags"
	LDFLAGS="$saved_ldflags"
	LIBS="$saved_libs"
fi

# Warning if not
if test "x$have_amd_opencl" = "xno"
then
	AC_MSG_WARN([
	
	The AMD OpenCL SDK is not presently installed in your system, as
	detected by the absence of the AMDAPPSDKROOT environment variable. The
	headers and libraries provided by this package are needed to build the
	Valar benchmark. Otherwise the AMD version won't be build.
	])
fi

#############
# Test OpenGL
#############

# Assume that we have OpenGL
have_opengl=yes

# Check headers
AC_CHECK_HEADER([GL/glut.h], [], [have_opengl=no])
AC_CHECK_HEADER([GL/gl.h], [], [have_opengl=no])
AC_CHECK_HEADER([GL/glu.h], [], [have_opengl=no])
AC_CHECK_HEADER([GL/glew.h], [], [have_opengl=no])

# Linker flags for OpenGL
OPENGL_LDFLAGS="-lGL -lGLU -lGLEW -lglut"
AC_SUBST([OPENGL_LDFLAGS])

# Export temporary LDFLAGS
saved_libs="$LIBS"
LIBS="$OPENGL_LDFLAGS"

# Try link
AC_MSG_CHECKING([for $OPENGL_LDFLAGS])
AC_LINK_IFELSE(
	[
		AC_LANG_PROGRAM(
			[[
				#include <GL/glut.h>
				#include <GL/gl.h>
				#include <GL/glu.h>
			]],
			[[
				glutInit(NULL, NULL);
				glBegin(0);
				gluOrtho2D(0.0, 0.0, 0.0, 0.0);
			]]
		)
	],
	[
		AC_MSG_RESULT(yes)
	],
	[
		AC_MSG_RESULT(no)
		have_opengl=no
	]
)

# Restore original LIBS
LIBS="$saved_libs"

# Message if not present
if test "x$have_opengl" = "xno"
then
	AC_MSG_WARN([

Your system lacks at least one of the graphics libraries required to
compile this OpenGL sample packege. These libraries are OpenGL, GLU,
GLUT, and GLEW. Please make sure that your linker supports flags

	$OPENGL_LDFLAGS

In Ubuntu, the following command is suggested to install the missing
libraries:

	sudo apt-get install freeglut3-dev libglew-dev]
)

fi

# Define macro for Automake and gcc
AM_CONDITIONAL(BUILD_BENCHMARK, test x$have_amd_opencl = xyes -a x$have_opengl = xyes)
if(test x$have_amd_opencl = xyes -a x$have_opengl = xyes)
then
	AC_DEFINE(BUILD_BENCHMARK, [1])
	CL_GL_LDFLAGS="-L$AMDAPPSDKROOT/lib/x86 -L$AMDAPPSDKROOT/lib/x86_64 -lGL -lGLU -lGLEW -lglut -lOpenCL"
	AC_SUBST([CL_GL_LDFLAGS])
	CL_GL_CPPFLAGS="-I$AMDAPPSDKROOT/include"
	AC_SUBST([CL_GL_CPPFLAGS])
fi


###########
# Makefiles
###########

AC_CONFIG_FILES([
	Makefile
	util/Makefile
	src/Makefile
	src/naive/Makefile
	src/naive/FFT/Makefile
	src/naive/GaussianNoise/Makefile
	src/naive/Julia/Makefile
	src/naive/MatMul/Makefile
	src/naive/NBody/Makefile
	src/optimized/Makefile
	src/optimized/FFT/Makefile
	src/optimized/GaussianNoise/Makefile
	src/optimized/Julia/Makefile
	src/optimized/MatMul/Makefile
	src/optimized/NBody/Makefile
])

AC_OUTPUT()

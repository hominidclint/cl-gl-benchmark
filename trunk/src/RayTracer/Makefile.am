if BUILD_BENCHMARK

bin_PROGRAMS = $(top_builddir)/bin/RayTracer

__top_builddir__bin_RayTracer_SOURCES = \
	RayTracer.cpp \
	RayTracer.hpp
	
AM_LDFLAGS = @CL_GL_LDFLAGS@
AM_CPPFLAGS = @CL_GL_CPPFLAGS@ -I$(top_builddir)/util
LDADD = $(top_builddir)/util/libsdk.a

endif

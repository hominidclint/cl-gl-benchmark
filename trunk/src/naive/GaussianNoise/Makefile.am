if BUILD_BENCHMARK

bin_PROGRAMS = $(top_builddir)/bin/naive/GaussianNoise

__top_builddir__bin_naive_GaussianNoise_SOURCES = \
	GaussianNoise.cpp \
	GaussianNoise.hpp
	
AM_LDFLAGS = @CL_GL_LDFLAGS@
AM_CPPFLAGS = @CL_GL_CPPFLAGS@ -I$(top_builddir)/util
LDADD = $(top_builddir)/util/libsdk.a

endif

if BUILD_BENCHMARK

bin_PROGRAMS = $(top_builddir)/bin/GaussianNoise

__top_builddir__bin_GaussianNoise_SOURCES = \
	GaussianNoise.cpp \
	GaussianNoise.hpp
	
AM_LDFLAGS = @CL_GL_LDFLAGS@
AM_CPPFLAGS = -I$(top_builddir)/util/include
LDADD = $(top_builddir)/util/libsdk.a

endif

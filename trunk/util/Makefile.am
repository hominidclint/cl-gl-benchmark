if BUILD_BENCHMARK

lib_LIBRARIES = libsdk.a

libsdk_a_SOURCES = \
			SDKApplication.cpp \
			SDKBitMap.cpp \
			SDKCommandArgs.cpp \
			SDKCommon.cpp \
			SDKFile.cpp \
			SDKThread.cpp \
			include/SDKApplication.hpp \
			include/SDKBitMap.hpp \
			include/SDKCommandArgs.hpp \
			include/SDKCommon.hpp \
			include/SDKFile.hpp \
			include/SDKThread.hpp 

AM_LDFLAGS = @CL_GL_LDFLAGS@
AM_CPPFLAGS = -I$(top_builddir)/util/include

endif
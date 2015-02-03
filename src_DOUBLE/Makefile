### Compilation options.

# C++ compiler. Tested with g++ and Intel icpc.
#CXX=/opt/local/bin/g++-mp-4.7
#CXX=icpc
CXX=/usr/bin/g++


# Compiler options. Note that -DEIGEN_NO_DEBUG is essential for good performance!
#CFLAGS=-g
CFLAGS=-O3 -DDOUBLE -DEIGEN_NO_DEBUG  -DNDEBUG -fpermissive


# Architecture. Set to x86_64 or i686 to override.
ARCH:=$(shell uname -m)
# Operating system. Set to override (the only option that makes any difference is Darwin).
OS:=$(shell uname -s)

# To build static binaries, uncomment the line below:
#STATIC=1

### Required libraries. You must install these prior to building.

# Set this to the root directory of Boost (should have a subdirectory named boost):
BOOST=/usr/usc/boost/1.51.0
#BOOST=/usr
#BOOST=/home/nlg-02/qingdou/tools/boost
# Where to find Boost header files
BOOST_INC=$(BOOST)/include

# Set this to the root directory of Eigen (should have a subdirectory named Eigen):
#EIGEN=../src/3rdparty
EIGEN=/home/nlg-05/qingdou/GITHUB/DECIPHERMENT_WITH_EMBEDDINGS/src/3rdparty

TBB=/home/nlg-05/qingdou/GITHUB/DECIPHERMENT_WITH_EMBEDDINGS/src/3rdparty/tbb

TBB_INC=$(TBB)/include

TBB_CFLAGS=-I$(TBB_INC)

TBB_LDFLAGS=-L$(TBB)/lib/intel64/gcc4.1

TBB_LDLIBS=-ltbb

### Optional libraries.

# To disable multithreading, comment out the line below:
OMP=1

# To use the MKL library, uncomment the line below and set it to the MKL root:
#MKL=/usr/usc/intel/12.1.1/mkl
# Set to 1 if you want to use the Single Dynamic Library; comment out otherwise.
# This is required for building the Python extensions, but doesn't work with building a static binary.
MKL_SINGLE=1

# For Python bindings, set the following and run 'make python/nplm.so'.
PYTHON_VERSION=2.7
#PYTHON_ROOT=/opt/local/Library/Frameworks/Python.framework/Versions/$(PYTHON_VERSION)
PYTHON_ROOT=/home/nlg-01/chiangd/pkg64/python
CYTHON=$(PYTHON_ROOT)/bin/cython

##### End of configurable options #####

# used for profiling
#USE_CHRONO=1

TCLAP=/home/nlg-05/qingdou/GITHUB/DECIPHERMENT_WITH_EMBEDDINGS/src/3rdparty/tclap/include

# Currently, this is needed only if USE_CHRONO is defined:
# Where to find Boost libraries
BOOST_LIB=$(BOOST)/lib
# On some systems, a suffix is appended for the multithreaded version.
#BOOST_LIB_SUFFIX=
BOOST_LIB_SUFFIX=-mt

BOOST_CFLAGS=-I$(BOOST_INC)
BOOST_LDFLAGS=
BOOST_LDLIBS=
#BOOST_LDLIBS=-lboost_system$(BOOST_LIB_SUFFIX) -lboost_thread$(BOOST_LIB_SUFFIX)
ifdef USE_CHRONO
  BOOST_CFLAGS+=-DUSE_CHRONO
  BOOST_LDLIBS+=-lboost_system$(BOOST_LIB_SUFFIX) -lboost_chrono$(BOOST_LIB_SUFFIX)
endif
ifdef BOOST_LDLIBS
  BOOST_LDFLAGS+=-L$(BOOST_LIB) -Wl,-rpath -Wl,$(BOOST_LIB)
endif

ifdef OMP
  ifneq (,$(findstring g++,$(CXX)))
    OMP_CFLAGS=-fopenmp
    OMP_LDFLAGS=-fopenmp
  endif
  ifneq (,$(findstring icpc,$(CXX)))
    OMP_CFLAGS=-openmp
    OMP_LDFLAGS=-openmp
  endif
endif

ifdef MKL
  ifdef MKL_SINGLE
    ifeq ($(ARCH),x86_64)
      MKL_LDFLAGS=-L$(MKL)/lib/intel64 -Wl,-rpath -Wl,$(MKL)/lib/intel64
    endif
    ifeq ($(ARCH),i686)
      MKL_LDFLAGS=-L$(MKL)/lib/ia32 -Wl,-rpath -Wl,$(MKL)/lib/ia32
    endif
    MKL_CFLAGS=-I$(MKL)/include -DEIGEN_USE_MKL_ALL -DMKL_SINGLE
    MKL_LDLIBS=-lmkl_rt

  else

    MKL_CFLAGS=-I$(MKL)/include -DEIGEN_USE_MKL_ALL
    MKL_LDLIBS=-Wl,--start-group
    ifeq ($(ARCH),x86_64)
      MKL_LDFLAGS=-L$(MKL)/lib/intel64 -Wl,-rpath -Wl,$(MKL)/lib/intel64
      MKL_LDLIBS+=-lmkl_intel_lp64
    endif
    ifeq ($(ARCH),i686)
      MKL_LDFLAGS=-L$(MKL)/lib/ia32 -Wl,-rpath -Wl,$(MKL)/lib/ia32
      MKL_LDLIBS+=-lmkl_intel
    endif

    ifneq (,$(findstring g++,$(CXX)))
      MKL_LDLIBS+=-lmkl_gnu_thread
    endif
    ifneq (,$(findstring icpc,$(CXX)))
      MKL_LDLIBS+=-lmkl_intel_thread
    endif

    MKL_LDLIBS+=-lmkl_core -Wl,--end-group
endif
endif


ifdef STATIC
  LDFLAGS+=-static
endif

ALL_CFLAGS=$(OMP_CFLAGS) $(MKL_CFLAGS) $(BOOST_CFLAGS) $(TBB_CFLAGS) -I$(TCLAP) -I$(EIGEN) $(CFLAGS) 
ALL_LDFLAGS=$(OMP_LDFLAGS) $(MKL_LDFLAGS) $(BOOST_LDFLAGS) $(LDFLAGS) $(TBB_LDFLAGS)
ALL_LDLIBS=$(MKL_LDLIBS) $(BOOST_LDLIBS) $(TBB_LDLIBS)

PYTHON_CFLAGS+=-I$(PYTHON_ROOT)/include/python$(PYTHON_VERSION)
ifeq ($(OS),Darwin)
  # avoid having to link in libpython
  PYTHON_LDFLAGS+=-undefined dynamic_lookup
endif

# Some other programs

AR=ar
RANLIB=ranlib

# Rules

#BINS=fast_align 
BINS=slice_with_embeddings_para
#LIBS=libnplm.a libnplm.so
OBJS=util.o model.o

all: $(BINS) $(LIBS)

clean:
	rm -f *.o shared/*.o python/*.o $(BINS) python/nplm.{cpp,so} python/nptm.{cpp,so}

install: all
	mkdir -p ../bin
	cp $(BINS) ../bin
	#mkdir -p ../lib
	#cp $(LIBS) ../lib

%.o: %.cpp
	$(CXX) -c $(ALL_CFLAGS) $< -o $@

shared/%.o: %.cpp
	$(CXX) -c -fPIC $(ALL_CFLAGS) $< -o $@

#fast_align: fast_align.o $(OBJS)
#	$(CXX) $(ALL_LDFLAGS) $^ $(ALL_LDLIBS) -o $@
	
		
slice_with_embeddings_para: slice_with_embeddings_para.o $(OBJS)
	$(CXX) $(ALL_LDFLAGS) $^ $(ALL_LDLIBS) -o $@
	
	#src/fast_align.cc src/ttables.h src/da.h src/corpus.h

#trainNeuralNetwork: trainNeuralNetwork.o $(OBJS)
#	$(CXX) $(ALL_LDFLAGS) $^ $(ALL_LDLIBS) -o $@

#libnplm.a: neuralLM.o $(OBJS)
#	rm -f $@
#	$(AR) rv $@ $^
#	$(RANLIB) $@

#libnplm.so: $(addprefix shared/,neuralLM.o $(OBJS))
#	$(CXX) -shared $(ALL_LDFLAGS) $^ $(ALL_LDLIBS) -o $@

#fast_align: src/fast_align.cc src/ttables.h src/da.h src/corpus.h
#	/opt/local/bin/g++-mp-4.5 -Werror -Wall -O3 -I. src/fast_align.cc -o $@

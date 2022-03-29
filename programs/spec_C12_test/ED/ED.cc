#ifdef BLAS
#include "util/LapackManager.h"
#pragma message("LapackManager")
#endif

#define USE_OLD_COMPRESSION
#define USE_HDF5_STORAGE
#define DMRG_DONT_USE_OPENMP
#define LINEARSOLVER_DIMK 100
#define HELPERS_IO_TABLE

#include <iostream>
#include <fstream>
#include <complex>

#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

#include "Logger.h"
Logger lout;
#include "ArgParser.h"

#include "StringStuff.h"
#include "Stopwatch.h"

#include "GrandSpinfulFermions.h"
#include "LanczosWrappers.h"
#include "Photo.h"

#include <iostream>
#include <fstream>
#include <complex>

#include "LanczosSolver.h"
#include "IntervalIterator.h"
#include "ParamHandler.h"

#include <boost/math/quadrature/ooura_fourier_integrals.hpp>
#include "InterpolGSL.h"

#include "models/ParamCollection.h"
#include "EigenFiles.h"
#include "LanczosPropagator.h"
#include "ComplexInterpolGSL.h"

typedef ED::GrandSpinfulFermions MODEL;
typedef VectorXd RealVector;
typedef VectorXcd ComplexVector;
#define USING_ED

#include "../spec_C12_test.cc"

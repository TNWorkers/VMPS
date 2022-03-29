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

#include "models/HubbardU1spin.h"
#include "solvers/DmrgSolver.h"
#include "DmrgLinearAlgebra.h"
#include "solvers/TDVPPropagator.h"

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

typedef VMPS::HubbardU1spin MODEL;
typedef MODEL::StateXd RealVector;
typedef MODEL::StateXcd ComplexVector;
#define USING_U1

enum DECONSTRUCTION     {CREATE=false, ANNIHILATE=true};

#include "../spec_C12_test.cc"

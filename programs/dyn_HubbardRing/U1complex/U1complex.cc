#if defined(BLAS) or defined(BLIS) or defined(MKL)
#include "util/LapackManager.h"
#pragma message("LapackManager")
#endif

//#define USE_OLD_COMPRESSION
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

#include "solvers/DmrgSolver.h"
#include "DmrgLinearAlgebra.h"
#include "models/ParamCollection.h"
#include "models/PeierlsHubbardU1.h"
#include "solvers/TDVPPropagator.h"
#include "IntervalIterator.h"

typedef VMPS::PeierlsHubbardU1 MODEL;
typedef Mpo<MODEL::Symmetry,MODEL::Scalar_> OPERATOR;
#define USING_U1_COMPLEX

#include "../dyn_HubbardRing.cc"

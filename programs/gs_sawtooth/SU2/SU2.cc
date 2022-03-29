#ifdef BLAS
#include "util/LapackManager.h"
#pragma message("LapackManager")
#endif

//#define USE_WIG_SU2_COEFFS

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

#include "solvers/DmrgSolver.h"
#include "models/ParamCollection.h"

//#include "models/HeisenbergU1.h"
//typedef VMPS::HeisenbergU1 MODEL;
//#define USING_U1

//#include "models/Heisenberg.h"
//typedef VMPS::Heisenberg MODEL;
//#define USING_U0

#include "models/HeisenbergSU2.h"
typedef VMPS::HeisenbergSU2 MODEL;
#define USING_SU2

#include "DmrgLinearAlgebra.h"

#include "../gs_sawtooth.cc"

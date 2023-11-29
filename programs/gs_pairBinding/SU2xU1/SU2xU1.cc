#if defined(BLAS) or defined(BLIS) or defined(MKL)
#include "util/LapackManager.h"
#pragma message("LapackManager")
#endif

//#define USE_OLD_COMPRESSION
#define USE_HDF5_STORAGE
#define DMRG_DONT_USE_OPENMP
#define LINEARSOLVER_DIMK 100
#define HELPERS_IO_TABLE
//#define VUMPS_SOLVER_DONT_USE_OPENMP

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

#include "models/ParamCollection.h"
#include "solvers/DmrgSolver.h"
#include "IntervalIterator.h"
#include "EigenFiles.h"
#include "SaveData.h"
#include "../TOOLS/HDF5Interface.h"

#include "models/HubbardSU2xU1.h"
typedef VMPS::HubbardSU2xU1 MODEL;
typedef Mpo<MODEL::Symmetry,MODEL::Scalar_> OPERATOR;
#define USING_SU2xU1

#include "../gs_pairBinding.cc"

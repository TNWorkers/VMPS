#ifdef BLAS
#include "util/LapackManager.h"
#pragma message("LapackManager")
#endif

#define DEBUG_VERBOSITY 0

#define USE_HDF5_STORAGE

// with Eigen:
#define DMRG_DONT_USE_OPENMP
//#define MPSQCOMPRESSOR_DONT_USE_OPENMP

// with own parallelization:
//#define EIGEN_DONT_PARALLELIZE

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_DEFAULT_INDEX_TYPE int

#include "Logger.h"
Logger lout;
using namespace std;

#if defined (USING_U0)
typedef VMPS::HeisenbergU0 MODEL;
#elif defined (USING_U1)
#include "models/HeisenbergU1.h"
#include "models/HeisenbergU1XXZ.h"
typedef VMPS::HeisenbergU1 MODEL;
#include "models/Heisenberg.h"
#include "models/HeisenbergXYZ.h"
#elif defined (USING_SU2)
#include "models/HeisenbergSU2.h"
typedef VMPS::HeisenbergSU2 MODEL;
#endif

#include "../test_2D.cc"

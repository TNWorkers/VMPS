#if defined(BLAS) or defined(BLIS) or defined(MKL)
#include "util/LapackManager.h"
#pragma message("LapackManager")
#endif

#define LANCZOS_MAX_ITERATIONS 20
#define USE_OLD_COMPRESSION
#define USE_HDF5_STORAGE
#define DMRG_DONT_USE_OPENMP
#define LINEARSOLVER_DIMK 100
#define HELPERS_IO_TABLE

#include <iostream>
#include <fstream>
#include <complex>
#include <filesystem>
#include <boost/asio/ip/host_name.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include "Logger.h"
Logger lout;
#include "ArgParser.h"

#include "StringStuff.h"
#include "Stopwatch.h"

#include "solvers/DmrgSolver.h"
#include "models/ParamCollection.h"
//#include "SloanCompressor.h"

//#include "models/HeisenbergU1.h"
//typedef VMPS::HeisenbergU1 MODEL;
//#define USING_U1

//#include "models/Heisenberg.h"
//typedef VMPS::Heisenberg MODEL;
//#define USING_U0

#include "models/HeisenbergSU2.h"
typedef VMPS::HeisenbergSU2 MODEL;
typedef Mpo<MODEL::Symmetry,MODEL::Scalar_> OPERATOR;
#define USING_SU2

#include "DmrgLinearAlgebra.h"
#include "SaveData.h"
#include "EigenFiles.h"

#include "../gs_HeisenbergKagomeBAB.cc"

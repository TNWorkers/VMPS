#if defined(BLAS) or defined(BLIS) or defined(MKL)
#include "util/LapackManager.h"
#pragma message("LapackManager")
#endif

#define USE_HDF5_STORAGE
#define DMRG_DONT_USE_OPENMP
#define GREENPROPAGATOR_USE_HDF5
#define TIME_PROP_USE_TERMPLOT
#define USE_OLD_COMPRESSION

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

#include <filesystem>
#include <boost/asio/ip/host_name.hpp>

#include "StringStuff.h"
#include "Stopwatch.h"

#include "solvers/DmrgSolver.h"
#include "models/ParamCollection.h"

#include "models/HubbardSU2xSU2.h"
typedef VMPS::HubbardSU2xSU2 MODEL;
#define USING_SU2xSU2

#include "solvers/GreenPropagator.h"
#include "DmrgLinearAlgebra.h"
#include "solvers/SpectralManager.h"

#include <boost/math/quadrature/ooura_fourier_integrals.hpp>
#include "InterpolGSL.h"
#include "IntervalIterator.h"

#include "../thermodyn_AltHubbard.cc"

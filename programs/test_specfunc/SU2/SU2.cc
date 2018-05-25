#define DONT_USE_LAPACK_SVD
#define DONT_USE_LAPACK_QR
#define DMRG_DONT_USE_OPENMP
#define EIGEN_DONT_PARALLELIZE

using namespace std;

#include "Logger.h"
Logger lout;

#include "tensors/SiteOperatorQ.h"
#include "models/HubbardSU2xU1.h"

typedef VMPS::HubbardSU2xU1 MODEL;

#include "../test_specfunc.cc"

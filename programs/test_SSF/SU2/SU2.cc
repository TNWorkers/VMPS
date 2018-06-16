#define DONT_USE_LAPACK_SVD
#define DONT_USE_LAPACK_QR
#define DMRG_DONT_USE_OPENMP
#define EIGEN_DONT_PARALLELIZE
#define PRINT_SU2_FACTORS

using namespace std;

#include "Logger.h"
Logger lout;

#include "tensors/SiteOperatorQ.h"
#include "models/HeisenbergSU2.h"

typedef VMPS::HeisenbergSU2 MODEL;
#define USING_SU2

#include "../test_SSF.cc"

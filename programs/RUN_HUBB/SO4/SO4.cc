#define DONT_USE_LAPACK_SVD
#define DONT_USE_LAPACK_QR
#define DMRG_DONT_USE_OPENMP
#define EIGEN_DONT_PARALLELIZE

using namespace std;

#include "Logger.h"
Logger lout;

#ifdef _OPENMP
#include <omp.h>
#endif

#include "models/HubbardSU2xSU2.h"
#include "models/HubbardSU2xSU2BondOperator.h"

typedef VMPS::HubbardSU2xSU2 MODEL;
#define USING_SO4

#include "../RUN_HUBB.cc"

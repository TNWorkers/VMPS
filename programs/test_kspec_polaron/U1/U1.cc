#define DONT_USE_LAPACK_SVD
#define DONT_USE_LAPACK_QR
#define DMRG_DONT_USE_OPENMP
#define EIGEN_DONT_PARALLELIZE

using namespace std;

#include "Logger.h"
Logger lout;

#include "models/KondoU1xU1.h"

typedef VMPS::KondoU1xU1 MODEL;

#include "../kspec_polaron.cc"

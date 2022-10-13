#define DONT_USE_LAPACK_SVD
#define DONT_USE_LAPACK_QR
#define DMRG_DONT_USE_OPENMP
#define EIGEN_DONT_PARALLELIZE

using namespace std;

#include "Logger.h"
Logger lout;

#include "models/HubbardSU2.h"
//#include "models/HubbardSU2xSU2BondOperator.h"

typedef VMPS::HubbardSU2 MODEL;
#define USING_SU2

#include "../RUN_HUBB.cc"

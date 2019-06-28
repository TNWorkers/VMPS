#include "util/LapackManager.h"
#pragma message("LapackManager")

#define DMRG_DONT_USE_OPENMP
#define EIGEN_DONT_PARALLELIZE

//#define USE_FAST_WIG_SU2_COEFFS
//#define OWN_HASH_CGC

using namespace std;

#include "Logger.h"
Logger lout;

#include "tensors/SiteOperatorQ.h"
#include "models/KondoSU2xU1.h"

typedef VMPS::KondoSU2xU1 MODEL;
#define USING_SU2

#include "../kspec_polaron.cc"

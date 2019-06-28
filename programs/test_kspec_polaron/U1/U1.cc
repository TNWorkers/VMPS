#include "util/LapackManager.h"
#pragma message("LapackManager")
#define DMRG_DONT_USE_OPENMP
#define EIGEN_DONT_PARALLELIZE

using namespace std;

#include "Logger.h"
Logger lout;

#include "models/KondoU1xU1.h"

typedef VMPS::KondoU1xU1 MODEL;
#define USING_U1

#include "../kspec_polaron.cc"

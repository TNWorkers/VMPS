#define DONT_USE_LAPACK_SVD
#define DONT_USE_LAPACK_QR
//#define USE_HDF5_STORAGE
//#define EIGEN_USE_THREADS

// with Eigen:
#define DMRG_DONT_USE_OPENMP
//#define MPSQCOMPRESSOR_DONT_USE_OPENMP

// with own parallelization:
//#define EIGEN_DONT_PARALLELIZE

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_DEFAULT_INDEX_TYPE int

#include <iostream>
#include <fstream>
#include <complex>
#include <variant>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

#include "Logger.h"
Logger lout;
#include "ArgParser.h"

//#include "solvers/MpsCompressor.h"
#include "solvers/TDVPPropagator.h"

//#include "models/HeisenbergU1.h"
//typedef VMPS::HeisenbergU1 MODEL;
//#include "models/Heisenberg.h"
//typedef VMPS::Heisenberg MODEL;
#include "models/HeisenbergSU2.h"
typedef VMPS::HeisenbergSU2 MODEL;
#include "solvers/DmrgSolver.h"

int main (int argc, char* argv[])
{
	MODEL H(4,{{"J",-1.}});
	cout << H.info() << endl;
	Eigenstate<MODEL::StateXd> g;
	MODEL::Solver DMRG(DMRG::VERBOSITY::ON_EXIT);
	DMRG.edgeState(H, g, {1ul}, LANCZOS::EDGE::GROUND, DMRG::CONVTEST::VAR_2SITE);
}

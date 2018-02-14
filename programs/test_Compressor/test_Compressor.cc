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

#include "solvers/DmrgSolver.h"
#include "solvers/TDVPPropagator.h"
#include "solvers/MpsCompressor.h"

#include "models/HeisenbergSU2.h"
#include "models/HeisenbergU1XXZ.h"
#include "models/HeisenbergXYZ.h"
#include "models/KondoU1xU1.h"

template<typename Scalar>
string to_string_prec (Scalar x, int n=14)
{
	ostringstream ss;
	ss << setprecision(n) << x;
	return ss.str();
}

bool CALC_DYNAMICS;
int M, S;
size_t D;
size_t L, Lx, Ly;
double J, Jprime;
double alpha;
double t_U0, t_U1, t_SU2;
int Dinit, Dlimit, Imin, Imax;
double tol_eigval, tol_state;
double dt;
DMRG::VERBOSITY::OPTION VERB;

int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	Lx = args.get<size_t>("Lx",10); L=Lx;
	Ly = args.get<size_t>("Ly",1);
	J = args.get<double>("J",-1.);
	Jprime = args.get<double>("Jprime",0.);
	M = args.get<int>("M",0);
	D = args.get<size_t>("D",2);
	S = abs(M)+1;
	alpha = args.get<double>("alpha",1.);
	VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",2));
	dt = 0.2;
	
	Dinit  = args.get<int>("Dmin",2);
	Dlimit = args.get<int>("Dmax",100);
	Imin   = args.get<int>("Imin",2);
	Imax   = args.get<int>("Imax",50);
	tol_eigval = args.get<double>("tol_eigval",1e-6);
	tol_state  = args.get<double>("tol_state",1e-5);
	
	CALC_DYNAMICS = args.get<bool>("CALC_DYN",0);
	
	lout << args.info() << endl;
	lout.set(make_string("Lx=",Lx,"_Ly=",Ly,"_M=",M,"_D=",D,"_J=",J,".log"),"log");
	
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	//--------U(1)---------
	lout << endl << "--------U(1)---------" << endl << endl;
	
	Stopwatch<> Watch_U1;
	VMPS::HeisenbergSU2 H_U1(Lx,{{"J",J},{"D",D},{"Ly",Ly}});
	lout << H_U1.info() << endl;
	Eigenstate<VMPS::HeisenbergSU2::StateXd> g_U1;
	
	VMPS::HeisenbergSU2::Solver DMRG_U1(DMRG::VERBOSITY::SILENT);
	DMRG_U1.edgeState(H_U1, g_U1, {1}, LANCZOS::EDGE::GROUND, LANCZOS::CONVTEST::NORM_TEST, tol_eigval,tol_state, Dinit,Dlimit, Imax,Imin, alpha);
	cout << g_U1.state.info() << endl;
	
	t_U1 = Watch_U1.time();
	
	VMPS::HeisenbergSU2::StateXd Psi = g_U1.state; Psi.setRandom();
	Psi.canonize();
	cout << "norm=" << Psi.squaredNorm() << endl;
//	VMPS::HeisenbergSU2::StateXd OxPsi;
//	
//	Psi.N_sv = 1e-15;
//	OxV(H_U1.Scomp(SZ,L/2), Psi, OxPsi, DMRG::BROOM::SVD);
	
	// compressor
	
	VMPS::HeisenbergSU2::StateXd HxPsi;
	HxV(H_U1, Psi, HxPsi, VERB);
	
	cout << setprecision(14) << avg(Psi, H_U1, Psi) << "\t" << dot(Psi,HxPsi) << endl << endl;
	
	VMPS::HeisenbergSU2::StateXd HxHxPsi;
	HxV(H_U1, HxPsi, HxHxPsi, VERB);
	
	cout << setprecision(14) << avg(Psi, H_U1, H_U1, Psi) << "\t" << dot(Psi,HxHxPsi) << endl << endl;
	
	
	
//	
//	VMPS::HeisenbergSU2 Heis(Lx,{{"J",J},{"Ly",Ly}});
//	cout << Heis.info() << endl;
//	VMPS::HeisenbergSU2::StateXd Phi(Heis,10,{1});
//	Phi.setRandom();
//	VMPS::HeisenbergSU2::StateXd Phi_init = Phi;
////	Phi.canonize();
//	size_t loc = static_cast<size_t>(Lx/2);
//	cout << "loc=" << loc << endl;
//	vector<Biped<VMPS::HeisenbergSU2::Symmetry,Matrix<double,Dynamic,Dynamic> > > Apair;
//	contract_AA(Phi.A_at(loc), Phi.locBasis(loc), Phi.A_at(loc+1), Phi.locBasis(loc+1), Apair);
//	cout << "contract_AA done!" << endl;
//	
//	cout << Phi.test_ortho() << endl;
//	Phi.sweepStep2(DMRG::DIRECTION::RIGHT, loc, Apair);
//	cout << Phi.test_ortho() << endl;
//	cout << "dot before=" << Phi.dot(Phi) << ", after=" << Phi_init.dot(Phi) << endl;
//	
}

#define DONT_USE_LAPACK_SVD
#define DONT_USE_LAPACK_QR
#define EIGEN_USE_THREADS

//#define USE_HDF5_STORAGE
//#define EIGEN_USE_THREADS
#undef _OPENMP
// with Eigen:
#define DMRG_DONT_USE_OPENMP
//#define MPSQCOMPRESSOR_DONT_USE_OPENMP

// with own parallelization:
#define EIGEN_DONT_PARALLELIZE

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_DEFAULT_INDEX_TYPE int

#include <iostream>
#include <fstream>
#include <complex>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

#include "Logger.h"
Logger lout;

#include <Eigen/Eigen>

#include "ArgParser.h"
// #include "LanczosWrappers.h"
#include "StringStuff.h"
#include "PolychromaticConsole.h"
#include "Stopwatch.h"
#include "numeric_limits.h"

// #include "SiteOperator.h"
#include "DmrgTypedefs.h"
#include "DmrgSolverQ.h"

#include "models/HeisenbergU1.h"
// #include "models/Heisenberg.h"
#include "models/HeisenbergSU2.h"

// #include "MpsQCompressor.h"

// #include "DmrgPivotStuff0.h"
// #include "DmrgPivotStuff2Q.h"
// #include "TDVPPropagator.h"


template<typename Scalar>
string to_string_prec (Scalar x, int n=14)
{
	ostringstream ss;
	ss << setprecision(n) << x;
	return ss.str();
}

// VMPS::HeisenbergU1::StateXcd Neel (const VMPS::HeisenbergU1 &H)
// {
// 	vector<qarray<1> > Neel_config(H.length());
// 	for (int l=0; l<H.length(); l+=2)
// 	{
// 		Neel_config[l]   = qarray<1>{+1};
// 		Neel_config[l+1] = qarray<1>{-1};
// 	}
	
// 	VMPS::HeisenbergU1::StateXcd Psi; 
// 	Psi.setProductState(H,Neel_config);
	
// 	return Psi;
// }

size_t L, Ly, Norb;
int M, S;
size_t D;
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
	L = args.get<int>("L",10);
	Ly = args.get<size_t>("Ly",1);
	Norb = args.get<size_t>("Norb",1);
	J = args.get<double>("J",-1.);
	Jprime = args.get<double>("Jprime",0.);
	M = args.get<int>("M",0);
	D = args.get<size_t>("D",2);
	S = abs(M)+1;
	alpha = args.get<double>("alpha",1.);
	VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",2));
	
	Dinit  = args.get<int>("Dmin",2);
	Dlimit = args.get<int>("Dmax",100);
	Imin   = args.get<int>("Imin",2);
	Imax   = args.get<int>("Imax",50);
	tol_eigval = args.get<double>("tol_eigval",1e-6);
	tol_state  = args.get<double>("tol_state",1e-5);

	MatrixXd Jpara(Ly,Ly); Jpara.setZero();
	Jpara.setConstant(-1.);
	
	lout << args.info() << endl;
	lout.set(make_string("L=",L,"_Ly=",Ly,"_M=",M,"_D=",D,"_J=",J,".log"),"log");
	
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif

	//---------ED----------
	spins::BaseSU2<> B(Norb, D);
	MatrixXd Jmat (Norb,Norb); Jmat.setZero();
	for(size_t i=0; i<Norb-2; i++)
	{
		if(i%2==0) {Jmat(i,i+1) = J; Jmat(i,i+2) = J; Jmat(i,i+3) = J;}
		if(i%2!=0) {Jmat(i,i+1) = J; Jmat(i,i+2) = J;}
	}
	Jmat(Norb-2,Norb-1) = J;

	lout << endl << "----------ED----------" << endl << endl;
	lout << "dimH=" << B.basis().inner_dim(qarray<1>{S}) << ", first 10 eigenenergies:" << endl;
	auto sol = B.HeisenbergHamiltonian(Jmat).diagonalize();
	for( size_t nu=0; nu<sol.data().size(); nu++)
	{
		if(sol.data().in[nu] == qarray<1>{S}) {lout << sol.data().block[nu].diagonal().head(10) << endl;}
	}
	
	//--------U(1)---------
	lout << endl << "--------U(1)---------" << endl << endl;
	
	VMPS::HeisenbergU1 H_U1(L,{{"D",D},{"Jpara",Jpara},{"Jperp",J}},Ly);
	lout << H_U1.info() << endl;
	Eigenstate<VMPS::HeisenbergU1::StateXd> g_U1;
	
	VMPS::HeisenbergU1::Solver DMRG_U1(VERB);
	DMRG_U1.edgeState(H_U1, g_U1, {M}, LANCZOS::EDGE::GROUND, LANCZOS::CONVTEST::NORM_TEST, tol_eigval,tol_state, Dinit,Dlimit, Imax,Imin, alpha);
		
	//--------SU(2)---------
	lout << endl << "--------SU(2)---------" << endl << endl;
	
	VMPS::HeisenbergSU2 H_SU2(L,{{"D",D},{"Jpara",Jpara},{"Jperp",J}},Ly);
	lout << H_SU2.info() << endl;
	Eigenstate<VMPS::HeisenbergSU2::StateXd> g_SU2;
	
	VMPS::HeisenbergSU2::Solver DMRG_SU2(VERB);
	DMRG_SU2.edgeState(H_SU2, g_SU2, {S}, LANCZOS::EDGE::GROUND, LANCZOS::CONVTEST::NORM_TEST, tol_eigval,tol_state, Dinit,Dlimit, Imax,Imin, alpha);
	
}

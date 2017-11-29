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

#include "SiteOperator.h"
#include "DmrgTypedefs.h"

#include "DmrgSolverQ.h"
#include "MpHeisenbergModel.h"
#include "MpGrandHeisenbergModel.h"
#include "MpsQCompressor.h"

#include "DmrgPivotStuff0.h"
#include "DmrgPivotStuff2Q.h"
#include "TDVPPropagator.h"

#include "models/MpHeisenbergSU2.h"

template<typename Scalar>
string to_string_prec (Scalar x, int n=14)
{
	ostringstream ss;
	ss << setprecision(n) << x;
	return ss.str();
}

int L, Ly, M, D, S;
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
	J = args.get<double>("J",-1.);
	Jprime = args.get<double>("Jprime",0.);
	M = args.get<int>("M",0);
	D = args.get<int>("D",2);
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
	
	lout << args.info() << endl;
	lout.set(make_string("L=",L,"_Ly=",Ly,"_M=",M,"_D=",D,"_J=",J,".log"),"log");
	
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	//--------U(0)---------
	lout << endl << "--------U(0)---------" << endl << endl;

	Stopwatch<> Watch_U0;
	VMPS::GrandHeisenbergModel H_U0(L,J,J,0,0,Ly,true,D); // Bz=0, Bx=0
	lout << H_U0.info() << endl;
	Eigenstate<VMPS::GrandHeisenbergModel::StateXd> g_U0;
	
	VMPS::GrandHeisenbergModel::Solver DMRG_U0(VERB);
	DMRG_U0.edgeState(H_U0, g_U0, {}, LANCZOS::EDGE::GROUND, LANCZOS::CONVTEST::NORM_TEST, tol_eigval,tol_state, Dinit,Dlimit, Imax,Imin, alpha);
	
	t_U0 = Watch_U0.time();

	VMPS::GrandHeisenbergModel::StateXd Hxg_U0;
	HxV(H_U0,g_U0.state,Hxg_U0,VERB);
	double E_U0_compressor = g_U0.state.dot(Hxg_U0);
	
	VMPS::GrandHeisenbergModel::StateXd Oxg_U0;
	Oxg_U0.eps_svd = 1e-15;
	OxV(H_U0,g_U0.state,Oxg_U0,DMRG::BROOM::SVD);
	double E_U0_zipper = g_U0.state.dot(Oxg_U0);
	
	//--------U(1)---------
	lout << endl << "--------U(1)---------" << endl << endl;
	
	Stopwatch<> Watch_U1;
	VMPS::HeisenbergModel H_U1(L,J,J,0,D,Ly,true); // Bz=0
	lout << H_U1.info() << endl;
	Eigenstate<VMPS::HeisenbergModel::StateXd> g_U1;
	
	VMPS::HeisenbergModel::Solver DMRG_U1(VERB);
	DMRG_U1.edgeState(H_U1, g_U1, {M}, LANCZOS::EDGE::GROUND, LANCZOS::CONVTEST::NORM_TEST, tol_eigval,tol_state, Dinit,Dlimit, Imax,Imin, alpha);
	
	t_U1 = Watch_U1.time();
	
	VMPS::HeisenbergModel::StateXd Hxg_U1;
	HxV(H_U1,g_U1.state,Hxg_U1,VERB);
	double E_U1_compressor = g_U1.state.dot(Hxg_U1);
	
	VMPS::HeisenbergModel::StateXd Oxg_U1;
	Oxg_U1.eps_svd = 1e-15;
	OxV(H_U1,g_U1.state,Oxg_U1,DMRG::BROOM::SVD);
	double E_U1_zipper = g_U1.state.dot(Oxg_U1);
	
	cout << avg(g_U1.state, H_U1.SzSz(0,1), g_U1.state) << endl;
	VMPS::HeisenbergModel H_U1t(L,2*J,0,0,D,Ly,true); // Bz=0
	VMPS::HeisenbergModel::StateXcd Psi = g_U1.state.cast<complex<double> >();
	TDVPPropagator<VMPS::HeisenbergModel,Sym::U1<double>,double,complex<double>,VMPS::HeisenbergModel::StateXcd> TDVP(H_U1t,Psi);
	for (int i=0; i<1; ++i)
	{
		TDVP.t_step(H_U1t,Psi, -1.i*dt, 1,1e-8);
		lout << avg(Psi, H_U1.SzSz(0,1), Psi) << endl;
	}
	
	//--------SU(2)---------
	lout << endl << "--------SU(2)---------" << endl << endl;
	
	Stopwatch<> Watch_SU2;
	VMPS::models::HeisenbergSU2 H_SU2(L,J,D,Jprime,Ly);
	lout << H_SU2.info() << endl;
	Eigenstate<VMPS::models::HeisenbergSU2::StateXd> g_SU2;
	
	VMPS::models::HeisenbergSU2::Solver DMRG_SU2(VERB);
	DMRG_SU2.edgeState(H_SU2, g_SU2, {S}, LANCZOS::EDGE::GROUND, LANCZOS::CONVTEST::NORM_TEST, tol_eigval,tol_state, Dinit,Dlimit, Imax,Imin, alpha);
	
	t_SU2 = Watch_SU2.time();
	
//	cout << avg(g_SU2.state, H_SU2.SSdag(0,1), g_SU2.state) << endl;
	
	//--------output---------
	
	TextTable T( '-', '|', '+' );
	
	double V = L*Ly;
	T.add(""); T.add("U(0)"); T.add("U(1)"); T.add("SU(2)"); T.endOfRow();
	
	T.add("E/L"); T.add(to_string_prec(g_U0.energy/V)); T.add(to_string_prec(g_U1.energy/V)); T.add(to_string_prec(g_SU2.energy/V)); T.endOfRow();
	T.add("E/L diff"); T.add(to_string_prec(abs(g_U0.energy-g_SU2.energy)/V)); T.add(to_string_prec(abs(g_U1.energy-g_SU2.energy)/V)); T.add("0");
	T.endOfRow();
	T.add("E/L Compressor"); T.add(to_string_prec(E_U0_compressor/V)); T.add(to_string_prec(E_U1_compressor/V)); T.add("-"); T.endOfRow();
	T.add("E/L Zipper"); T.add(to_string_prec(E_U0_zipper/V)); T.add(to_string_prec(E_U1_zipper/V)); T.add("-"); T.endOfRow();

	T.add("t/s"); T.add(to_string_prec(t_U0,2)); T.add(to_string_prec(t_U1,2)); T.add(to_string_prec(t_SU2,2)); T.endOfRow();
	T.add("t gain"); T.add(to_string_prec(t_U0/t_SU2,2)); T.add(to_string_prec(t_U1/t_SU2,2)); T.add("1"); T.endOfRow();
	
	T.add("Dmax"); T.add(to_string(g_U0.state.calc_Dmax())); T.add(to_string(g_U1.state.calc_Dmax())); T.add(to_string(g_SU2.state.calc_Dmax()));
	T.endOfRow();
	T.add("Mmax"); T.add(to_string(g_U0.state.calc_Dmax())); T.add(to_string(g_U1.state.calc_Mmax())); T.add(to_string(g_SU2.state.calc_Mmax()));
	T.endOfRow();
	
	lout << endl << T;
}

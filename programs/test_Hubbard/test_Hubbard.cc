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

#include <Eigen/Eigen>

#include "ArgParser.h"
// #include "LanczosWrappers.h"
#include "StringStuff.h"
#include "PolychromaticConsole.h"
#include "Stopwatch.h"
#include "numeric_limits.h"

#include "HubbardModel.h"
#include "LanczosWrappers.h"
#include "LanczosSolver.h"
#include "Photo.h"

#include "SiteOperator.h"
#include "DmrgTypedefs.h"

#include "DmrgSolverQ.h"
#include "models/HeisenbergU1.h"
#include "models/Heisenberg.h"
#include "MpsQCompressor.h"

#include "DmrgPivotStuff0.h"
#include "DmrgPivotStuff2Q.h"
#include "TDVPPropagator.h"

#include "models/Hubbard.h"
#include "models/HubbardU1xU1.h"
#include "models/HubbardSU2xU1.h"

template<typename Scalar>
string to_string_prec (Scalar x, int n=14)
{
	ostringstream ss;
	ss << setprecision(n) << x;
	return ss.str();
}

bool CALC_DYNAMICS;
size_t L, Lx, Ly;
double t, tPrime, U, mu, Bz;
int Nupdn, N;
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
	std::array<size_t,2> Lxy = {Lx,Ly};
	t = args.get<double>("t",1.);
	tPrime = args.get<double>("tPrime",0.);
	U = args.get<double>("U",8.);
	mu = args.get<double>("mu",0.5*U);
	Nupdn = args.get<int>("Nupdn",L/2);
	N = 2*Nupdn;
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
	lout.set(make_string("Lx=",Lx,"_Ly=",Ly,"_t=",t,"_t'=",tPrime,"_U=",U,".log"),"log");
	
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	//--------ED-----------
	lout << endl << "--------ED---------" << endl << endl;
	
	InteractionParams params;
	params.set_U(U);
	params.set_hoppings({-t,-tPrime});
	HubbardModel H_ED(L,Nupdn,Nupdn,params, BC_DANGLING);
	lout << H_ED.info() << endl;
	Eigenstate<VectorXd> g_ED;
	LanczosSolver<HubbardModel,VectorXd,double> Lutz;
	Lutz.edgeState(H_ED,g_ED,LANCZOS::EDGE::GROUND);
	
	HubbardModel H_EDm(L,Nupdn-1,Nupdn,params, BC_DANGLING);
	Eigenstate<VectorXd> g_EDm;
	Lutz.edgeState(H_EDm,g_EDm,LANCZOS::EDGE::GROUND);
	
	Photo Ph(H_EDm,H_ED,UP,L-1);
	cout << g_EDm.energy << ", <c>=" << avg(g_EDm.state, Ph.Operator(), g_ED.state) << endl;
	
	lout << "Emin/L=" << to_string_prec(g_ED.energy/Lx) << endl;
	
//	lout << endl << H_ED.eigenvalues() << endl;
	
	MatrixXd densityMatrix_ED(L,L); densityMatrix_ED.setZero();
	for (size_t i=0; i<L; ++i) 
	for (size_t j=0; j<L; ++j)
	{
		densityMatrix_ED(i,j) = avg(g_ED.state, H_ED.hopping_element(j,i,UP), g_ED.state)+
		                        avg(g_ED.state, H_ED.hopping_element(j,i,DN), g_ED.state);
	}
	lout << "<cdagc>=" << endl << densityMatrix_ED << endl;
	
	ArrayXd d_ED(L); d_ED=0.;
	for (size_t i=0; i<L; ++i) 
	{
		d_ED(i) = avg(g_ED.state, H_ED.d(i), g_ED.state);
	}
	lout << "<d>=" << endl << d_ED << endl;
	
	//--------U(0)---------
	lout << endl << "--------U(0)---------" << endl << endl;
	
	Stopwatch<> Watch_U0;
	VMPS::Hubbard H_U0(Lxy,{{"t",t},{"tPrime",tPrime},{"U",U},{"mu",mu}});
	lout << H_U0.info() << endl;
	Eigenstate<VMPS::Hubbard::StateXd> g_U0;
	
	VMPS::Hubbard::Solver DMRG_U0(VERB);
	DMRG_U0.edgeState(H_U0, g_U0, {}, LANCZOS::EDGE::GROUND, LANCZOS::CONVTEST::NORM_TEST, 10.*tol_eigval,10.*tol_state, Dinit,3*Dlimit, Imax,Imin, 0.1);
	
	lout << endl;
	double Ntot = 0.;
	for (size_t lx=0; lx<Lx; ++lx)
	for (size_t ly=0; ly<Ly; ++ly)
	{
		double n_l = avg(g_U0.state, H_U0.n(UPDN,lx,ly), g_U0.state);
		cout << "lx=" << lx << ", ly=" << ly << "\tn=" << n_l << endl;
		Ntot += n_l;
	}
	
	double Emin_U0 = g_U0.energy+mu*Ntot;
	double emin_U0 = Emin_U0/(Lx*Ly);
	lout << "correction for mu: E=" << to_string_prec(Emin_U0) << ", E/L=" << to_string_prec(emin_U0) << endl;
	
	t_U0 = Watch_U0.time();
//	
//	// observables
//	
//	Eigen::MatrixXd SpinCorr_U0(L,L); SpinCorr_U0.setZero();
//	for(size_t i=0; i<L; i++) for(size_t j=0; j<L; j++) { SpinCorr_U0(i,j) = 3*avg(g_U0.state, H_U0.SzSz(i,j), g_U0.state); }
//	
//	// compressor
//	
//	VMPS::Hubbard::StateXd Hxg_U0;
//	HxV(H_U0,g_U0.state,Hxg_U0,VERB);
//	double E_U0_compressor = g_U0.state.dot(Hxg_U0);
//	
//	// zipper
//	
//	VMPS::Hubbard::StateXd Oxg_U0;
//	Oxg_U0.eps_svd = 1e-15;
//	OxV(H_U0,g_U0.state,Oxg_U0,DMRG::BROOM::SVD);
//	double E_U0_zipper = g_U0.state.dot(Oxg_U0);
	
	//--------U(1)---------
	lout << endl << "--------U(1)---------" << endl << endl;
	
	Stopwatch<> Watch_U1;
	
	VMPS::HubbardU1xU1 H_U1(Lxy,{{"t",t},{"tPrime",tPrime},{"U",U},{"CALC_SQUARE",false}});
	lout << H_U1.info() << endl;
	Eigenstate<VMPS::HubbardU1xU1::StateXd> g_U1;
	
	VMPS::HubbardU1xU1::Solver DMRG_U1(VERB);
	DMRG_U1.edgeState(H_U1, g_U1, {Nupdn,Nupdn}, LANCZOS::EDGE::GROUND, LANCZOS::CONVTEST::SQ_TEST, tol_eigval,tol_state, Dinit,Dlimit, Imax,Imin, alpha);
	
	t_U1 = Watch_U1.time();
	
	Eigenstate<VMPS::HubbardU1xU1::StateXd> g_U1m;
	DMRG_U1.edgeState(H_U1, g_U1m, {Nupdn-1,Nupdn}, LANCZOS::EDGE::GROUND, LANCZOS::CONVTEST::SQ_TEST, tol_eigval,tol_state, Dinit,Dlimit, Imax,Imin, alpha);
	cout << g_U1m.energy << ", <c>=" << avg(g_U1m.state, H_U1.c(UP,L-1), g_U1.state) << endl;
	cout << H_U1.c(UP,L-1) << endl;
	
	// observables
	
	MatrixXd densityMatrix_U1(L,L); densityMatrix_U1.setZero();
	for (size_t i=0; i<L; ++i) 
	for (size_t j=0; j<L; ++j)
	{
		densityMatrix_U1(i,j) = avg(g_U1.state, H_U1.cdagc(UP,i,j), g_U1.state)+
		                        avg(g_U1.state, H_U1.cdagc(DN,i,j), g_U1.state);
	}
	lout << "<cdagc>=" << endl << densityMatrix_U1 << endl;
	
	MatrixXd densityMatrix_U1B(L,L); densityMatrix_U1B.setZero();
	for (size_t i=0; i<L; ++i) 
	for (size_t j=0; j<L; ++j)
	{
		densityMatrix_U1B(i,j) = avg(g_U1.state, H_U1.cdag(UP,i), H_U1.c(UP,j), g_U1.state)+
		                         avg(g_U1.state, H_U1.cdag(DN,i), H_U1.c(DN,j), g_U1.state);
	}
	lout << endl << densityMatrix_U1B << endl;
	
	lout << endl;
	complex<double> P=0.;
	int L_2 = static_cast<int>(Lx)/2;
	for (int i=0; i<Lx; ++i)
	for (int j=0; j<Lx; ++j)
	for (int n=-L_2; n<L_2; ++n)
	{
		double k = 2.*M_PI*n/Lx;
		P += k * exp(-1.i*k*static_cast<double>(i-j)) * densityMatrix_U1(i,j);
	}
	P /= (Lx*Lx);
	cout << "P=" << P << endl;
	
	ArrayXd d_U1(L); d_U1=0.;
	for (size_t i=0; i<L; ++i) 
	{
		d_U1(i) = avg(g_U1.state, H_U1.d(i), g_U1.state);
	}
	lout << "<d>=" << endl << d_U1 << endl;
	
//	// compressor
//	
//	VMPS::HubbardU1xU1::StateXd Hxg_U1;
//	HxV(H_U1,g_U1.state,Hxg_U1,VERB);
//	double E_U1_compressor = g_U1.state.dot(Hxg_U1);
//	
//	// zipper
//	
//	VMPS::HubbardU1xU1::StateXd Oxg_U1;
//	Oxg_U1.eps_svd = 1e-15;
//	OxV(H_U1,g_U1.state,Oxg_U1,DMRG::BROOM::SVD);
//	double E_U1_zipper = g_U1.state.dot(Oxg_U1);
	
	// --------SU(2)---------
	lout << endl << "--------SU(2)---------" << endl << endl;
	
	Stopwatch<> Watch_SU2;
	
	VMPS::HubbardSU2xU1 H_SU2(Lxy,{{"t",t},{"tPrime",tPrime},{"U",U},{"CALC_SQUARE",false}});
	lout << H_SU2.info() << endl;
	Eigenstate<VMPS::HubbardSU2xU1::StateXd> g_SU2;
	
	VMPS::HubbardSU2xU1::Solver DMRG_SU2(VERB);
	DMRG_SU2.edgeState(H_SU2, g_SU2, {1,N}, LANCZOS::EDGE::GROUND, LANCZOS::CONVTEST::SQ_TEST, tol_eigval,tol_state, Dinit,Dlimit, Imax,Imin, alpha);
	
	t_SU2 = Watch_SU2.time();
	
	MatrixXd densityMatrix_SU2(L,L); densityMatrix_SU2.setZero();
	for (size_t i=0; i<L; ++i) 
	for (size_t j=0; j<L; ++j)
	{
		densityMatrix_SU2(i,j) = avg(g_SU2.state, H_SU2.cdagc(i,j), g_SU2.state);
	}
	lout << densityMatrix_SU2 << endl;
	
	MatrixXd densityMatrix_SU2B(L,L); densityMatrix_SU2B.setZero();
	for (size_t i=0; i<L; ++i) 
	for (size_t j=0; j<L; ++j)
	{
		densityMatrix_SU2B(i,j) = sqrt(2.)*avg(g_SU2.state, H_SU2.cdag(i), H_SU2.c(j), g_SU2.state);
	}
	lout << endl << densityMatrix_SU2B << endl;
	
	ArrayXd d_SU2(L); d_SU2=0.;
	for (size_t i=0; i<L; ++i) 
	{
		d_SU2(i) = avg(g_SU2.state, H_SU2.d(i), g_SU2.state);
	}
	lout << "<d>=" << endl << d_SU2 << endl;
	
	//--------output---------
	TextTable T( '-', '|', '+' );
	
	double V = L*Ly; double Vsq = V*V;
	T.add(""); T.add("U(0)"); T.add("U(1)xU(1)"); T.add("SU(2)xU(1)"); T.endOfRow();
	
	T.add("E/L");
	T.add(to_string_prec(emin_U0));
	T.add(to_string_prec(g_U1.energy/V));
	T.add(to_string_prec(g_SU2.energy/V));
	T.endOfRow();
	
	T.add("E/L diff"); 
	T.add(to_string_prec(abs(Emin_U0-g_ED.energy)/V)); 
	T.add(to_string_prec(abs(g_U1.energy-g_ED.energy)/V)); 
	T.add(to_string_prec(abs(g_SU2.energy-g_ED.energy)/V));
	T.endOfRow();
	
//	T.add("E/L Compressor"); T.add(to_string_prec(E_U0_compressor/V)); T.add(to_string_prec(E_U1_compressor/V)); T.add("-"); T.endOfRow();
//	T.add("E/L Zipper"); T.add(to_string_prec(E_U0_zipper/V)); T.add(to_string_prec(E_U1_zipper/V)); T.add("-"); T.endOfRow();
	
	T.add("t/s"); T.add(to_string_prec(t_U0,2)); T.add(to_string_prec(t_U1,2)); T.add(to_string_prec(t_SU2,2)); T.endOfRow();
	T.add("t gain"); T.add(to_string_prec(t_U0/t_SU2,2)); T.add(to_string_prec(t_U1/t_SU2,2)); T.add("1"); T.endOfRow();
	
	T.add("observables diff");
	T.add("-");
	T.add(to_string_prec((densityMatrix_U1-densityMatrix_ED).norm()));
	T.add(to_string_prec((densityMatrix_SU2-densityMatrix_ED).norm()));
	T.endOfRow();
	
	T.add("Dmax"); T.add(to_string(g_U0.state.calc_Dmax())); T.add(to_string(g_U1.state.calc_Dmax())); T.add(to_string(g_SU2.state.calc_Dmax()));
	T.endOfRow();
	T.add("Mmax"); T.add(to_string(g_U0.state.calc_Dmax())); T.add(to_string(g_U1.state.calc_Mmax())); T.add(to_string(g_SU2.state.calc_Mmax()));
	T.endOfRow();
	
	lout << endl << T;
}

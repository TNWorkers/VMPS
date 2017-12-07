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
#include "models/HeisenbergU1.h"
#include "models/Heisenberg.h"
#include "MpsQCompressor.h"

#include "DmrgPivotStuff0.h"
#include "DmrgPivotStuff2Q.h"
#include "TDVPPropagator.h"

#include "models/HeisenbergSU2.h"

#include "models/HubbardU1xU1.h"

template<typename Scalar>
string to_string_prec (Scalar x, int n=14)
{
	ostringstream ss;
	ss << setprecision(n) << x;
	return ss.str();
}

VMPS::HeisenbergU1::StateXcd Neel (const VMPS::HeisenbergU1 &H)
{
	vector<qarray<1> > Neel_config(H.length());
	for (int l=0; l<H.length(); l+=2)
	{
		Neel_config[l]   = qarray<1>{+1};
		Neel_config[l+1] = qarray<1>{-1};
	}
	
	VMPS::HeisenbergU1::StateXcd Psi; 
	Psi.setProductState(H,Neel_config);
	
	return Psi;
}

bool CALC_DYNAMICS;
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
	
	double U = args.get<double>("U",10.);
	double t = args.get<double>("t",1.);
	int Nupdn = args.get<double>("Nupdn",L/2);
	
	Dinit  = args.get<int>("Dmin",2);
	Dlimit = args.get<int>("Dmax",100);
	Imin   = args.get<int>("Imin",2);
	Imax   = args.get<int>("Imax",50);
	tol_eigval = args.get<double>("tol_eigval",1e-6);
	tol_state  = args.get<double>("tol_state",1e-5);

	CALC_DYNAMICS = args.get<bool>("CALC_DYN",0);

	lout << args.info() << endl;
	lout.set(make_string("L=",L,"_Ly=",Ly,"_M=",M,"_D=",D,"_J=",J,".log"),"log");
	
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	//--------U(0)---------
//	lout << endl << "--------U(0)---------" << endl << endl;
//	
//	Stopwatch<> Watch_U0;
//	VMPS::GrandHeisenbergU1 H_U0(L,J,J,0,0,Ly,true,D); // Bz=0, Bx=0
//	lout << H_U0.info() << endl;
//	Eigenstate<VMPS::GrandHeisenbergU1::StateXd> g_U0;
//	
//	VMPS::GrandHeisenbergU1::Solver DMRG_U0(VERB);
//	DMRG_U0.edgeState(H_U0, g_U0, {}, LANCZOS::EDGE::GROUND, LANCZOS::CONVTEST::NORM_TEST, tol_eigval,tol_state, Dinit,3*Dlimit, Imax,Imin, alpha);
//	
//	t_U0 = Watch_U0.time();
//	
//	// observables
//	
//	Eigen::MatrixXd SpinCorr_U0(L,L); SpinCorr_U0.setZero();
//	for(size_t i=0; i<L; i++) for(size_t j=0; j<L; j++) { SpinCorr_U0(i,j) = 3*avg(g_U0.state, H_U0.SzSz(i,j), g_U0.state); }
//	
//	// compressor	
//	
//	VMPS::GrandHeisenbergU1::StateXd Hxg_U0;
//	HxV(H_U0,g_U0.state,Hxg_U0,VERB);
//	double E_U0_compressor = g_U0.state.dot(Hxg_U0);
//	
//	// zipper
//	
//	VMPS::GrandHeisenbergU1::StateXd Oxg_U0;
//	Oxg_U0.eps_svd = 1e-15;
//	OxV(H_U0,g_U0.state,Oxg_U0,DMRG::BROOM::SVD);
//	double E_U0_zipper = g_U0.state.dot(Oxg_U0);
//	
	//--------U(1)---------
	lout << endl << "--------U(1)---------" << endl << endl;
	
	VMPS::HubbardU1xU1 Hub_U1(L,{{"U",U},{"t",t}},Ly);
	lout << Hub_U1.info() << endl;
	Eigenstate<VMPS::HubbardU1xU1::StateXd> g_U1;
	
	VMPS::HubbardU1xU1::Solver DMRG_U1(VERB);
	DMRG_U1.edgeState(Hub_U1, g_U1, {Nupdn,Nupdn}, LANCZOS::EDGE::GROUND, LANCZOS::CONVTEST::NORM_TEST, tol_eigval,tol_state, Dinit,Dlimit, Imax,Imin, alpha);
	
	
	VMPS::HubbardU1xU1 H1(L,{{"U",U,0},{"U",2*U,1},{"t",t}},Ly);
	lout << H1.info() << endl;
	VMPS::HubbardU1xU1 H2(L,{{"U",U,0},{"t",t,0},{"t",2*t,1}},Ly);
	lout << H2.info() << endl;
	
//	Stopwatch<> Watch_U1;
////	VMPS::HeisenbergU1 H_U1(L,J,J,0,D,Ly,true); // Bz=0
//	VMPS::HeisenbergU1 H_U1(L,{{"J",-1.,0},{"J",-0.5,1}},Ly);
//	lout << H_U1.info() << endl;
//	Eigenstate<VMPS::HeisenbergU1::StateXd> g_U1;
//	
//	VMPS::HeisenbergU1::Solver DMRG_U1(VERB);
//	DMRG_U1.edgeState(H_U1, g_U1, {M}, LANCZOS::EDGE::GROUND, LANCZOS::CONVTEST::NORM_TEST, tol_eigval,tol_state, Dinit,Dlimit, Imax,Imin, alpha);
//	
//	t_U1 = Watch_U1.time();

//	// observables
//	Eigen::MatrixXd SpinCorr_U1(L,L); SpinCorr_U1.setZero();
//	for(size_t i=0; i<L; i++) for (size_t j=0; j<L; j++) { SpinCorr_U1(i,j) = 3*avg(g_U1.state, H_U1.SzSz(i,j), g_U1.state); }

//	// compressor
//	
//	VMPS::HeisenbergU1::StateXd Hxg_U1;
//	HxV(H_U1,g_U1.state,Hxg_U1,VERB);
//	double E_U1_compressor = g_U1.state.dot(Hxg_U1);
//	
//	// zipper
//	
//	VMPS::HeisenbergU1::StateXd Oxg_U1;
//	Oxg_U1.eps_svd = 1e-15;
//	OxV(H_U1,g_U1.state,Oxg_U1,DMRG::BROOM::SVD);
//	double E_U1_zipper = g_U1.state.dot(Oxg_U1);
	
	// dynamics (of Néel state)
//	if(CALC_DYNAMICS)
//	{
//		int Ldyn = 12;
//		vector<double> Jz_list = {0, -1, -2, -4};
//		
//		for (const auto& Jz:Jz_list)
//		{
//			VMPS::HeisenbergU1 H_U1t(Ldyn,J,Jz,0,D,1,true);
//			lout << H_U1t.info() << endl;
//			VMPS::HeisenbergU1::StateXcd Psi = Neel(H_U1t);
//			TDVPPropagator<VMPS::HeisenbergU1,Sym::U1<double>,double,complex<double>,VMPS::HeisenbergU1::StateXcd> TDVP(H_U1t,Psi);
//		
//			double t = 0;
//			ofstream Filer(make_string("Mstag_Jxy=",J,"_Jz=",Jz,".dat"));
//			for (int i=0; i<=static_cast<int>(6./dt); ++i)
//			{
//				double res = 0;
//				for (int l=0; l<Ldyn; ++l)
//				{
//					res += pow(-1.,l) * isReal(avg(Psi, H_U1t.Sz(l), Psi));
//				}
//				res /= Ldyn;
//				if(VERB != DMRG::VERBOSITY::SILENT) {lout << t << "\t" << res << endl;}
//				Filer << t << "\t" << res << endl;
//		
//				TDVP.t_step(H_U1t,Psi, -1.i*dt, 1,1e-8);
//				if(VERB != DMRG::VERBOSITY::SILENT) {lout << TDVP.info() << endl << Psi.info() << endl;}
//				t += dt;
//			}
//			Filer.close();
//		}
//	}
	
	//--------SU(2)---------
//	lout << endl << "--------SU(2)---------" << endl << endl;
//	
//	Stopwatch<> Watch_SU2;
//	VMPS::models::HeisenbergSU2 H_SU2(L,J,D,Jprime,Ly);
//	lout << H_SU2.info() << endl;
//	Eigenstate<VMPS::models::HeisenbergSU2::StateXd> g_SU2;
//	
//	VMPS::models::HeisenbergSU2::Solver DMRG_SU2(VERB);
//	DMRG_SU2.edgeState(H_SU2, g_SU2, {S}, LANCZOS::EDGE::GROUND, LANCZOS::CONVTEST::NORM_TEST, tol_eigval,tol_state, Dinit,Dlimit, Imax,Imin, alpha);
//	
//	t_SU2 = Watch_SU2.time();

//	Eigen::MatrixXd SpinCorr_SU2(L,L); SpinCorr_SU2.setZero();
//	for(size_t i=0; i<L; i++) for(size_t j=0; j<L; j++) { SpinCorr_SU2(i,j) = avg(g_SU2.state, H_SU2.SSdag(i,j), g_SU2.state); }
//	//--------output---------
//	
//	TextTable T( '-', '|', '+' );
//	
//	double V = L*Ly; double Vsq = V*V;
//	T.add(""); T.add("U(0)"); T.add("U(1)"); T.add("SU(2)"); T.endOfRow();
//	
//	T.add("E/L"); T.add(to_string_prec(g_U0.energy/V)); T.add(to_string_prec(g_U1.energy/V)); T.add(to_string_prec(g_SU2.energy/V)); T.endOfRow();
//	T.add("E/L diff"); T.add(to_string_prec(abs(g_U0.energy-g_SU2.energy)/V)); T.add(to_string_prec(abs(g_U1.energy-g_SU2.energy)/V)); T.add("0");
//	T.endOfRow();
//	T.add("E/L Compressor"); T.add(to_string_prec(E_U0_compressor/V)); T.add(to_string_prec(E_U1_compressor/V)); T.add("-"); T.endOfRow();
//	T.add("E/L Zipper"); T.add(to_string_prec(E_U0_zipper/V)); T.add(to_string_prec(E_U1_zipper/V)); T.add("-"); T.endOfRow();

//	T.add("t/s"); T.add(to_string_prec(t_U0,2)); T.add(to_string_prec(t_U1,2)); T.add(to_string_prec(t_SU2,2)); T.endOfRow();
//	T.add("t gain"); T.add(to_string_prec(t_U0/t_SU2,2)); T.add(to_string_prec(t_U1/t_SU2,2)); T.add("1"); T.endOfRow();

//	T.add("observables"); T.add(to_string_prec(SpinCorr_U0.sum()));
//	T.add(to_string_prec(SpinCorr_U1.sum())); T.add(to_string_prec(SpinCorr_SU2.sum())); T.endOfRow();

//	T.add("observables diff"); T.add(to_string_prec((SpinCorr_U0-SpinCorr_SU2).lpNorm<1>()/Vsq));
//	T.add(to_string_prec((SpinCorr_U1-SpinCorr_SU2).lpNorm<1>()/Vsq)); T.add("0"); T.endOfRow();

//	T.add("Dmax"); T.add(to_string(g_U0.state.calc_Dmax())); T.add(to_string(g_U1.state.calc_Dmax())); T.add(to_string(g_SU2.state.calc_Dmax()));
//	T.endOfRow();
//	T.add("Mmax"); T.add(to_string(g_U0.state.calc_Dmax())); T.add(to_string(g_U1.state.calc_Mmax())); T.add(to_string(g_SU2.state.calc_Mmax()));
//	T.endOfRow();
//	
//	lout << endl << T;
//	
//	VMPS::HeisenbergU1 Htest1(L,{{"Jxy",-1.},{"Jz",-4.}});
//	cout << Htest1.info() << endl;
//	
//	VMPS::HeisenbergU1 Htest2(L,{{"Jxy",-1.}});
//	cout << Htest2.info() << endl;
//	
//	VMPS::HeisenbergU1 Htest3(L,{{"Jz",-1.}});
//	cout << Htest3.info() << endl;
//	
//	VMPS::HeisenbergU1 Htest4(L,{{"J",-1.},{"Bz",10.},{"K",2.}});
//	cout << Htest4.info() << endl;
//	
//	MatrixXd Jpara(2,2); Jpara.setRandom();
//	VMPS::HeisenbergU1 Htest5(L,{{"Jpara",Jpara}},2,2);
//	cout << Htest5.info() << endl;
//	
//	MatrixXd Jxypara(2,2); Jxypara.setRandom();
//	VMPS::HeisenbergU1 Htest6(L,{{"Jxypara",Jxypara}},2,2);
//	cout << Htest6.info() << endl;
//	
//	MatrixXd Jzpara(2,2); Jzpara.setRandom();
//	VMPS::HeisenbergU1 Htest7(L,{{"Jzpara",Jzpara}},2,2);
//	cout << Htest7.info() << endl;
//	
//	VMPS::HeisenbergU1 Htest8(L,{{"Jxypara",Jxypara},{"Jzpara",Jzpara}},2,2);
//	cout << Htest8.info() << endl;
//	
//	VMPS::HeisenbergU1 Htest9(L,{{"Jpara",Jpara},{"Jperp",-1.5}},2,2);
//	cout << Htest9.info() << endl;
//	
//	VMPS::HeisenbergU1 Htest10(L,{{"Jpara",Jpara},{"Jxyperp",-3.},{"Jzperp",-4.}},2,2);
//	cout << Htest10.info() << endl;
//	
//	VMPS::HeisenbergU1 Htest11(L,{{"K",5.}});
//	cout << Htest11.info() << endl;
}

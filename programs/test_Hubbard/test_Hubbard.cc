#ifdef BLAS
#include "util/LapackManager.h"
#pragma message("LapackManager")
#endif
//#define LANCZOS_MAX_ITERATIONS 100
//#define USE_HDF5_STORAGE

#define DEBUG_VERBOSITY 0
// with Eigen:
#define DMRG_DONT_USE_OPENMP
//#define MPSQCOMPRESSOR_DONT_USE_OPENMP

// with own parallelization:
//#define EIGEN_DONT_PARALLELIZE

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_DEFAULT_INDEX_TYPE int

//Also calculate SU2xSU2, implies no tPrime
#define SU2XSU2

#include <iostream>
#include <fstream>
#include <complex>
#include <variant>
#include <fenv.h> 
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

#include "Logger.h"
Logger lout;
#include "ArgParser.h"

// ED stuff
#include "HubbardModel.h"
#include "LanczosWrappers.h"
#include "LanczosSolver.h"
#include "Photo.h"
#include "Auger.h"

#include "solvers/DmrgSolver.h"

#include "models/Hubbard.h"
#include "models/HubbardU1xU1.h"
#include "models/HubbardU1.h"
#include "models/HubbardSU2xU1.h"
#include "models/HubbardSU2.h"
#ifdef SU2XSU2
#include "models/HubbardSU2xSU2.h"
#endif

template<typename Scalar>
string to_string_prec (Scalar x, bool COLOR=false, int n=14)
{
	ostringstream ss;
	COLOR=false;
	if (x < 1e-5 and COLOR)
	{
		ss << termcolor::colorize << termcolor::green << setprecision(n) << x << termcolor::reset;
	}
	else if (x >= 1e-5 and COLOR)
	{
		ss << termcolor::colorize << termcolor::red << setprecision(n) << x << termcolor::reset;
	}
	else
	{
		ss << setprecision(n) << x;
	}
	return ss.str();
}

complex<double> Ptot (const MatrixXd &densityMatrix, int L)
{
	complex<double> P=0.;
	int L_2 = static_cast<int>(L)/2;
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	for (int n=-L_2; n<L_2; ++n)
	{
		double k = 2.*M_PI*n/L;
		P += k * exp(-1.i*k*static_cast<double>(i-j)) * densityMatrix(i,j);
	}
	P /= (L*L);
	return P;
}

constexpr double phase(int i)
{
	if (i % 2) {return -1.;}
	return 1.;
}

size_t L, Ly, Ly2;
int Vol, Vsq;
double t, tPrime, tRung, U, mu, Bz, V;
int M, N, S, Nup, Ndn;
double alpha;
double t_U0, t_U1, t_SU2, t_SU2xSU2;
int Dinit, Dlimit, Imin, Imax, Qinit;
double tol_eigval, tol_state;
int i0;
DMRG::VERBOSITY::OPTION VERB;
double overlap_ED = 0.;
double overlap_U1_zipper = 0.;
double emin_U0 = 0.;
double Emin_U0 = 0.;
double Emin_SU2xSU2 = 0.;
double emin_SU2xSU2 = 0.;
bool ED, U0, U1, SU2, SO4, CORR, PRINT;

Eigenstate<VectorXd> g_ED;
Eigenstate<VMPS::Hubbard::StateXd> g_U0;
Eigenstate<VMPS::HubbardU1xU1::StateXd> g_U1;
Eigenstate<VMPS::HubbardSU2xU1::StateXd> g_SU2;
Eigenstate<VMPS::HubbardSU2xSU2::StateXd> g_SU2xSU2;

ArrayXd cup_U1, cdn_U1;
ArrayXd c_SU2;
ArrayXd cdag_U1;
ArrayXd cdag_SU2;

ArrayXd c_U1e;
ArrayXd c_SU2e;
ArrayXd cdag_U1e;
ArrayXd cdag_SU2e;

MatrixXd densityMatrix_ED;
VectorXd d_ED, h_ED;

MatrixXd densityMatrix_U1A, densityMatrix_U1B;
VectorXd d_U1;

MatrixXd densityMatrix_SU2A, densityMatrix_SU2B, spin_SU2, isospin_SU2;
VectorXd d_SU2;

MatrixXd densityMatrix_SU2xSU2A, densityMatrix_SU2xSU2B, spin_SU2xSU2, isospin_SU2xSU2;

VectorXd nh_SU2xSU2, ns_SU2xSU2;

int main (int argc, char* argv[])
{
//	feenableexcept(FE_INVALID | FE_OVERFLOW);
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",4);
	Ly = args.get<size_t>("Ly",1);
	Ly2 = args.get<size_t>("Ly2",Ly);
	t = args.get<double>("t",1.);
	tPrime = args.get<double>("tPrime",0.);
	tRung = args.get<double>("tRung",0.);
	U = args.get<double>("U",8.);
	V = args.get<double>("V",0.);
	double Vz = args.get<double>("Vz",V);
	double Vxy = args.get<double>("Vxy",V);
	mu = args.get<double>("mu",0.5*U);
	N = args.get<int>("N",L*Ly);
	M = args.get<int>("M",0);;
	S = abs(M)+1;
	// for ED:
	Nup = (N+M)/2;
	Ndn = (N-M)/2;

	size_t maxPower = args.get<size_t>("maxPower",2);
	
	ED = args.get<bool>("ED",false);
	U0 = args.get<bool>("U0",false);
	U1 = args.get<bool>("U1",true);
	SU2 = args.get<bool>("SU2",true);
	SO4 = args.get<bool>("SO4",true);
	CORR = args.get<bool>("CORR",false);
	if (CORR) {ED = true;}
	PRINT = args.get<bool>("PRINT",false);
	if (CORR == false) {PRINT = false;}
	
	DMRG::CONTROL::GLOB GlobParam;
	DMRG::CONTROL::DYN  DynParam;
	size_t min_Nsv = args.get<size_t>("min_Nsv",0ul);
	DynParam.min_Nsv = [min_Nsv] (size_t i) {return min_Nsv;};
	
	alpha = args.get<double>("alpha",100.);
	
	VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",2));
	
	i0 = args.get<int>("i0",L/2);
	
	GlobParam.Minit  = args.get<int>("Minit",2);
	GlobParam.Mlimit = args.get<int>("Mlimit",100);
	GlobParam.Qinit = args.get<int>("Qinit",10);
	GlobParam.min_halfsweeps = args.get<int>("Imin",6);
	GlobParam.max_halfsweeps = args.get<int>("Imax",20);
	GlobParam.tol_eigval = args.get<double>("tol_eigval",1e-6);
	GlobParam.tol_state = args.get<double>("tol_state",1e-5);
	
	lout << args.info() << endl;
	lout.set(make_string("L=",L,"_Ly=",Ly,"_t=",t,"_t'=",tPrime,"_U=",U,".log"),"log");
	
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	if (CORR)
	{
		// resize all to prevent crashes at the end:
		d_ED.resize(L); d_ED.setZero();
		h_ED.resize(L); h_ED.setZero();
		d_U1.resize(L); d_U1.setZero();
		d_SU2.resize(L); d_SU2.setZero();
		nh_SU2xSU2.resize(L); nh_SU2xSU2.setZero();
		ns_SU2xSU2.resize(L); ns_SU2xSU2.setZero();
		
		densityMatrix_ED.resize(L,L); densityMatrix_ED.setZero();
		densityMatrix_U1A.resize(L,L); densityMatrix_U1A.setZero();
		densityMatrix_U1B.resize(L,L); densityMatrix_U1B.setZero();
		densityMatrix_SU2A.resize(L,L); densityMatrix_SU2A.setZero();
		densityMatrix_SU2B.resize(L,L); densityMatrix_SU2B.setZero();
		spin_SU2.resize(L,L); spin_SU2.setZero();
		isospin_SU2.resize(L,L); isospin_SU2.setZero();
		densityMatrix_SU2xSU2A.resize(L,L); densityMatrix_SU2xSU2A.setZero();
		densityMatrix_SU2xSU2B.resize(L,L); densityMatrix_SU2xSU2B.setZero();
		spin_SU2xSU2.resize(L,L); spin_SU2xSU2.setZero();
		isospin_SU2xSU2.resize(L,L); isospin_SU2xSU2.setZero();
	}
	
	//--------ED-----------
	if (ED)
	{
		lout << endl << termcolor::red << "--------ED---------" << termcolor::reset << endl << endl;
		
		HubbardModel Test(2,1,1,12.,BC_DANGLING);
		Test.print_basis();
		cout << Test.Hmatrix() << endl;
		cout << Test.eigenvalues().transpose() << endl;
		cout << endl;
		cout << Test.eigenvectors() << endl;
		
		if (L <= 12)
		{
			InteractionParams params;
			params.set_U(U);
			(tPrime!=0) ? params.set_hoppings({-t,-tPrime}):params.set_hoppings({-t});
			
		//	MatrixXd BondMatrix(L*Ly,L*Ly); BondMatrix.setZero();
		//	BondMatrix(0,1) = -t;
		//	BondMatrix(1,0) = -t;
		//	
		//	BondMatrix(0,2) = -t;
		//	BondMatrix(2,0) = -t;
		//	
		//	BondMatrix(2,3) = -t;
		//	BondMatrix(3,2) = -t;
		//	
		//	BondMatrix(1,3) = -t;
		//	BondMatrix(3,1) = -t;
		
		//	HubbardModel H_ED(L*Ly,Nup,Ndn,U,BondMatrix.sparseView(), BC_DANGLING);
			HubbardModel H_ED(L*Ly,Nup,Ndn,params, BC_DANGLING);
			lout << H_ED.info() << endl;
			LanczosSolver<HubbardModel,VectorXd,double> Lutz;
			Lutz.edgeState(H_ED,g_ED,LANCZOS::EDGE::GROUND, 1.e-12, 1.e-12);
			cout << Lutz.info() << endl;
		//	HubbardModel H_EDm(L*Ly,Nup-1,Ndn,U,BondMatrix.sparseView(), BC_DANGLING);
			HubbardModel H_EDm(L*Ly,Nup-1,Ndn,params, BC_DANGLING);
			Eigenstate<VectorXd> g_EDm;
			Lutz.edgeState(H_EDm,g_EDm,LANCZOS::EDGE::GROUND, 1.e-12, 1.e-12);
			
		//	HubbardModel H_EDmm(L*Ly,Nup-1,Ndn-1,U,BondMatrix.sparseView(), BC_DANGLING);
			HubbardModel H_EDmm(L*Ly,Nup-1,Ndn-1,params, BC_DANGLING);
			Eigenstate<VectorXd> g_EDmm;
			Lutz.edgeState(H_EDmm,g_EDmm,LANCZOS::EDGE::GROUND);
			cout << "ED: E=" << g_ED.energy << endl;
			for (int l=0; l<L; ++l)
			{
				Photo Ph(H_EDm,H_ED,UP,l);
			}
			
			Auger A(H_EDmm, H_ED, i0);
			VectorXd OxV_ED = A.Operator() * g_ED.state;
			double overlap_ED = g_EDmm.state.dot(OxV_ED);
			
			if (CORR)
			{
				densityMatrix_ED.resize(L,L); densityMatrix_ED.setZero();
				for (size_t i=0; i<L; ++i) 
				for (size_t j=0; j<L; ++j)
				{
					densityMatrix_ED(i,j) = avg(g_ED.state, H_ED.hopping_element(j,i,UP), g_ED.state)+
						                    avg(g_ED.state, H_ED.hopping_element(j,i,DN), g_ED.state);
				}
	//			lout << "<cdagc>=" << endl << densityMatrix_ED << endl;
				
				
				for (size_t i=0; i<L; ++i) 
				{
					d_ED(i) = avg(g_ED.state, H_ED.d(i), g_ED.state);
					h_ED(i) = 1.-avg(g_ED.state, H_ED.n(i), g_ED.state)+d_ED(i);
				}
			}
		}
	}
	
	//--------U(0)---------
	if (U0)
	{
		lout << endl << termcolor::red << "--------U(0)---------" << termcolor::reset << endl << endl;
		
		Stopwatch<> Watch_U0;
		//{"tPara",tParaA,0},{"tPara",tParaB,1}
		VMPS::Hubbard H_U0(L,{{"t",t},{"tPrime",tPrime},{"U",U},{"mu",mu},{"tRung",tRung},{"Ly",Ly,0},{"Ly",Ly2,1},{"maxPower",maxPower}});
//		VMPS::Hubbard H_U0(L,{{"t",t},{"tPrime",tPrime},{"U",U},{"mu",mu},{"Ly",Ly}});
		Vol = H_U0.volume();
		Vsq = V*V;
		lout << H_U0.info() << endl;
		
		VMPS::Hubbard::Solver DMRG_U0(VERB);
		DMRG_U0.userSetGlobParam();
		DMRG_U0.userSetDynParam();
		DMRG_U0.GlobParam = GlobParam;
		DMRG_U0.DynParam = DynParam;
		DMRG_U0.edgeState(H_U0, g_U0, {}, LANCZOS::EDGE::GROUND);
		
		t_U0 = Watch_U0.time();
		
		lout << endl;
		double Ntot = 0.;
		for (size_t lx=0; lx<L; ++lx)
		for (size_t ly=0; ly<Ly; ++ly)
		{
			double n_l = avg(g_U0.state, H_U0.n(lx,ly), g_U0.state);
			cout << "lx=" << lx << ", ly=" << ly << "\tn=" << n_l << endl;
			Ntot += n_l;
		}
		
		Emin_U0 = g_U0.energy+mu*Ntot;
		emin_U0 = Emin_U0/Vol;
		lout << "correction for mu: E=" << to_string_prec(Emin_U0) << ", E/V=" << to_string_prec(emin_U0) << endl;
	}
	
	//--------U(1)---------
	if (U1)
	{
		lout << endl << termcolor::red << "--------U(1)---------" << termcolor::reset << endl << endl;
		
		Stopwatch<> Watch_U1;
		
		//,{"tPara",tParaA,0},{"tPara",tParaB,1}
		VMPS::HubbardU1xU1 H_U1(L,{{"t",t},{"tPrime",tPrime},{"U",U},{"tRung",tRung},{"Ly",Ly,0},{"Ly",Ly2,1},{"maxPower",maxPower}});
//		VMPS::Hubbard H_U1(L,{{"t",t},{"tPrime",tPrime},{"U",U},{"Ly",Ly}});
		Vol = H_U1.volume();
		Vsq = V*V;
		lout << H_U1.info() << endl;
		
		VMPS::HubbardU1xU1::Solver DMRG_U1(VERB);
		DMRG_U1.userSetGlobParam();
		DMRG_U1.userSetDynParam();
		DMRG_U1.GlobParam = GlobParam;
		DMRG_U1.DynParam = DynParam;
		DMRG_U1.edgeState(H_U1, g_U1, {M,N}, LANCZOS::EDGE::GROUND);
		g_U1.state.graph("U1");
		if (g_U1.state.A_at(0)[0].block[0](0,0) < 0) {g_U1.state *= -1.;}
					
		ArrayXd check(maxPower);
		for (size_t i=1; i<=maxPower;i++)
		{
			check(i-1) = avg(g_U1.state,H_U1,g_U1.state,i) - std::pow(g_U1.energy,i);
		}
		cout << "check=" << check.transpose() << endl;		
		t_U1 = Watch_U1.time();
		
		if (CORR)
		{
			Eigenstate<VMPS::HubbardU1xU1::StateXd> g_U1Mm;
			DMRG_U1.set_verbosity(DMRG::VERBOSITY::ON_EXIT);
			DMRG_U1.edgeState(H_U1, g_U1Mm, {M-1,N-1}, LANCZOS::EDGE::GROUND);
			lout << "g_U1Mm.energy=" << g_U1Mm.energy << endl;
			if (g_U1Mm.state.A_at(0)[0].block[0](0,0) < 0) {g_U1Mm.state *= -1.;}
			
			cup_U1.resize(L);
			for (int l=0; l<L; ++l)
			{
				cup_U1(l) = avg(g_U1Mm.state, H_U1.c<UP>(l), g_U1.state);
				cout << "l=" << l << ", <cup>=" << cup_U1(l) << endl;
			}

			Eigenstate<VMPS::HubbardU1xU1::StateXd> g_U1Mp;
			DMRG_U1.set_verbosity(DMRG::VERBOSITY::ON_EXIT);
			DMRG_U1.edgeState(H_U1, g_U1Mp, {M+1,N-1}, LANCZOS::EDGE::GROUND);
			lout << "g_U1Mp.energy=" << g_U1Mp.energy << endl;
			if (g_U1Mm.state.A_at(0)[0].block[0](0,0) < 0) {g_U1Mm.state *= -1.;}

			cdn_U1.resize(L);
			for (int l=0; l<L; ++l)
			{
				cdn_U1(l) = avg(g_U1Mp.state, H_U1.c<DN>(l), g_U1.state);
				cout << "l=" << l << ", <cdn>=" << cdn_U1(l) << endl;
			}
			
			// cdag_U1.resize(L);
			// for (int l=0; l<L; ++l)
			// {
			// 	cdag_U1(l) = avg(g_U1.state, H_U1.cdag<UP>(l), g_U1m.state);
			// 	cout << "l=" << l << ", <cdag>=" << cdag_U1(l) << endl;
			// }

			// Eigenstate<VMPS::HubbardU1xU1::StateXd> g_U1me;
			// DMRG_U1.set_verbosity(DMRG::VERBOSITY::SILENT);
			// DMRG_U1.edgeState(H_U1, g_U1me, {M-1,N+1}, LANCZOS::EDGE::GROUND);
			// lout << "g_U1me.energy=" << g_U1me.energy << endl;
			// if (g_U1me.state.A_at(0)[0].block[0](0,0) < 0) {g_U1me.state *= -1.;}
			
			// c_U1e.resize(L);
			// for (int l=0; l<L; ++l)
			// {
			// 	c_U1e(l) = avg(g_U1.state, H_U1.c<DN>(l), g_U1me.state);
			// 	cout << "l=" << l << ", <ce>=" << c_U1e(l) << endl;
			// }
		
			// cdag_U1e.resize(L);
			// for (int l=0; l<L; ++l)
			// {
			// 	cdag_U1e(l) = avg(g_U1me.state, H_U1.cdag<DN>(l), g_U1.state);
			// 	cout << "l=" << l << ", <cdage>=" << cdag_U1e(l) << endl;
			// }
			
			for (size_t i=0; i<L; ++i) 
			for (size_t j=0; j<L; ++j)
			{
				densityMatrix_U1A(i,j) = avg(g_U1.state, H_U1.cdagc<UP,UP>(i,j), g_U1.state)+
				                         avg(g_U1.state, H_U1.cdagc<DN,DN>(i,j), g_U1.state);
			}
			
			for (size_t i=0; i<L; ++i) 
			for (size_t j=0; j<L; ++j)
			{
				densityMatrix_U1B(i,j) = avg(g_U1.state, H_U1.cdag<UP>(i), H_U1.c<UP>(j), g_U1.state)+
				                         avg(g_U1.state, H_U1.cdag<DN>(i), H_U1.c<DN>(j), g_U1.state);
			}
			
			cout << endl << densityMatrix_U1A << endl << endl;
			cout << endl << densityMatrix_U1B << endl << endl;
			
			for (size_t i=0; i<L; ++i) 
			{
				d_U1(i) = avg(g_U1.state, H_U1.d(i), g_U1.state);
			}
		}
		
		//////////
//		VMPS::HubbardU1xU1::StateXd cPhi;
//		auto C1 = H_U1.c<UP>(L/2);
//		OxV_exact(C1, g_U1.state, cPhi, 2., DMRG::VERBOSITY::SILENT);
//		
//		VMPS::HubbardU1xU1::StateXd cdagPhi;
//		auto C2 = H_U1.c<UP>(L/2);
//		OxV_exact(C2, g_U1.state, cdagPhi, 2., DMRG::VERBOSITY::SILENT);
//		
//		VMPS::HubbardU1xU1::StateXd aPhi;
//		auto A1 = H_U1.a<UP>(L/2);
//		OxV_exact(A1, g_U1.state, aPhi, 2., DMRG::VERBOSITY::SILENT);
//		
//		VMPS::HubbardU1xU1::StateXd adagPhi;
//		auto A2 = H_U1.a<UP>(L/2);
//		OxV_exact(A2, g_U1.state, adagPhi, 2., DMRG::VERBOSITY::SILENT);
//		
//		auto String = H_U1.JWstring(L/2,L/2);
//		cout << "cmp=" << avg(adagPhi, String, aPhi) << ", " << dot(cdagPhi, cPhi) << endl;
//		cout << "string left=" << avg(adagPhi, String, H_U1, aPhi) << endl;
//		cout << "string right=" << avg(adagPhi, H_U1, String, aPhi) << endl;
//		cout << "no string=" << avg(adagPhi, H_U1, aPhi) << endl;
//		cout << "with c,cdag=" << avg(cdagPhi, H_U1, cPhi) << endl;
		//////////
	}
	
//	// --------SU(2)---------
	if (SU2)
	{
		lout << endl << termcolor::red << "--------SU(2)---------" << termcolor::reset << endl << endl;
		
		Stopwatch<> Watch_SU2;
		
		VMPS::HubbardSU2xU1 H_SU2(L,{{"t",t},{"tPrime",tPrime},{"U",U},{"Ly",Ly,0},{"Ly",Ly2,1},{"tRung",tRung},{"Vz",Vz},{"Vxy",Vxy},{"maxPower",maxPower}}, BC::OPEN);
		Vol = H_SU2.volume();
		Vsq = Vol*Vol;
		lout << H_SU2.info() << endl;
		
		VMPS::HubbardSU2xU1::Solver DMRG_SU2(VERB);
		DMRG_SU2.userSetGlobParam();
		DMRG_SU2.userSetDynParam();
		DMRG_SU2.GlobParam = GlobParam;
		DMRG_SU2.DynParam = DynParam;
		DMRG_SU2.edgeState(H_SU2, g_SU2, {S,N}, LANCZOS::EDGE::GROUND);
		g_SU2.state.graph("SU2");
		if (g_SU2.state.A_at(0)[0].block[0](0,0) < 0) {g_SU2.state *= -1.;}
			
		ArrayXd check(maxPower);
		for (size_t i=1; i<=maxPower;i++)
		{
			check(i-1) = avg(g_SU2.state,H_SU2,g_SU2.state,i) - std::pow(g_SU2.energy,i);
		}
		cout << "check=" << check.transpose() << endl;
		
		t_SU2 = Watch_SU2.time();
		
		if (CORR)
		{
			Eigenstate<VMPS::HubbardSU2xU1::StateXd> g_SU2m;
			DMRG_SU2.set_verbosity(DMRG::VERBOSITY::ON_EXIT);
			DMRG_SU2.edgeState(H_SU2, g_SU2m, {2,N-1}, LANCZOS::EDGE::GROUND);
			if (g_SU2m.state.A_at(0)[0].block[0](0,0) < 0) {g_SU2m.state *= -1.;}
			lout << "g_SU2m.energy=" << g_SU2m.energy << endl;
			
			c_SU2.resize(L);
			for (int l=0; l<L; ++l)
			{
				c_SU2(l) = avg(g_SU2m.state, H_SU2.c(l,0,1.), g_SU2.state);
				cout << "l=" << l << ", <c>=" << c_SU2(l) << "\t" << c_SU2(l)/cup_U1(l) << "\t" << c_SU2(l)/cdn_U1(l) << endl;
			}
			
			// cdag_SU2.resize(L);
			// for (int l=0; l<L; ++l)
			// {
			// 	cdag_SU2(l) = avg(g_SU2.state, H_SU2.cdag(l,0,std::sqrt(0.5)), g_SU2m.state);
			// 	cout << "l=" << l << ", <cdag>=" << cdag_SU2(l) << "\t" << cdag_SU2(l)/cdag_U1(l) << endl;
			// }

			// Eigenstate<VMPS::HubbardSU2xU1::StateXd> g_SU2me;
			// DMRG_SU2.set_verbosity(DMRG::VERBOSITY::SILENT);
			// DMRG_SU2.edgeState(H_SU2, g_SU2me, {abs(Nup-1-Ndn)+1,N+1}, LANCZOS::EDGE::GROUND);
			// lout << "g_SU2me.energy=" << g_SU2me.energy << endl;
			// if (g_SU2me.state.A_at(0)[0].block[0](0,0) < 0) {g_SU2me.state *= -1.;}
			
			// c_SU2e.resize(L);
			// for (int l=0; l<L; ++l)
			// {
			// 	c_SU2e(l) = avg(g_SU2.state, H_SU2.c(l,0,1.), g_SU2me.state);
			// 	cout << "l=" << l << ", <ce>=" << c_SU2e(l) << "\t" << c_SU2e(l)/c_U1e(l) << endl;
			// }
			
			// cdag_SU2e.resize(L);
			// for (int l=0; l<L; ++l)
			// {
			// 	cdag_SU2e(l) = avg(g_SU2me.state, H_SU2.cdag(l,0,std::sqrt(0.5)), g_SU2.state);
			// 	cout << "l=" << l << ", <cdage>=" << cdag_SU2e(l) << "\t" << cdag_SU2e(l)/cdag_U1e(l) << endl;
			// }
			
			for (size_t i=0; i<L; ++i) 
			for (size_t j=0; j<L; ++j)
			{
				densityMatrix_SU2A(i,j) = avg(g_SU2.state, H_SU2.cdagc(i,j), g_SU2.state);
			}
			
			for (size_t i=0; i<L; ++i) 
			for (size_t j=0; j<L; ++j)
			{
				cout << "beginning cdagc" << endl;
				auto cdagc = MpoTerms<VMPS::HubbardSU2xU1::Symmetry,double>::prod(H_SU2.cdag(i,0,-sqrt(2.)), H_SU2.c(j), {1,0});
				
				// Doesn't compile:
//				densityMatrix_SU2B(i,j) = avg(g_SU2.state, cdagc, g_SU2.state);
				// densityMatrix_SU2B(i,j) = avg(g_SU2.state, H_SU2.cdag(i,0,-sqrt(2.)), H_SU2.c(j), g_SU2.state);
			}
			assert(false);
			for (size_t i=0; i<L; ++i) 
			for (size_t j=0; j<L; ++j)
			{
				spin_SU2(i,j) = avg(g_SU2.state, H_SU2.SdagS(i,j), g_SU2.state);
			}
			for (size_t i=0; i<L; ++i) 
			for (size_t j=0; j<L; ++j)
			{
				isospin_SU2(i,j) = avg(g_SU2.state, H_SU2.TzTz(i,j), g_SU2.state) + 
				                   0.5*(avg(g_SU2.state, H_SU2.TpTm(i,j), g_SU2.state) + 
				                        avg(g_SU2.state, H_SU2.TmTp(i,j), g_SU2.state));
			}
			
			cout << endl << densityMatrix_SU2A << endl << endl;
			cout << endl << densityMatrix_SU2B << endl << endl;
			
			cout << "(densityMatrix_SU2A-densityMatrix_SU2B).norm()=" << (densityMatrix_SU2A-densityMatrix_SU2B).norm() << endl;
			
			for (size_t i=0; i<L; ++i) 
			{
				d_SU2(i) = avg(g_SU2.state, H_SU2.d(i), g_SU2.state);
			}
		}
	}
	
//	// --------SU(2)xSU(2)---------
	if (SO4)
	{
		lout << endl << termcolor::red << "--------SU(2)xSU(2)---------" << termcolor::reset << endl << endl;
		
		Stopwatch<> Watch_SU2xSU2;
		
		vector<Param> paramsSU2xSU2;
		paramsSU2xSU2.push_back({"t",t,0});
		paramsSU2xSU2.push_back({"t",t,1});
		paramsSU2xSU2.push_back({"U",U,0});
		paramsSU2xSU2.push_back({"U",U,1});
		paramsSU2xSU2.push_back({"V",V,0});
		paramsSU2xSU2.push_back({"V",V,1});
		paramsSU2xSU2.push_back({"Ly",Ly,0});
		paramsSU2xSU2.push_back({"Ly",Ly,1});
		paramsSU2xSU2.push_back({"maxPower",maxPower});
		VMPS::HubbardSU2xSU2 H_SU2xSU2(L, paramsSU2xSU2, BC::OPEN);
		Vol = H_SU2xSU2.volume();
		Vsq = Vol*Vol;
		lout << H_SU2xSU2.info() << endl;
		
		VMPS::HubbardSU2xSU2::Solver DMRG_SU2xSU2(VERB);
		DMRG_SU2xSU2.userSetGlobParam();
		DMRG_SU2xSU2.userSetDynParam();
		DMRG_SU2xSU2.GlobParam = GlobParam;
		DMRG_SU2xSU2.DynParam = DynParam;
		DMRG_SU2xSU2.edgeState(H_SU2xSU2, g_SU2xSU2, {S,Vol-N+1}, LANCZOS::EDGE::GROUND); 
		//Todo: check Pseudospin quantum number... (1 <==> half filling)
		g_SU2xSU2.state.graph("SU2xSU2");
		ArrayXd check(maxPower);
		ArrayXd check2(maxPower);
		check2(3-1) = avg(g_SU2xSU2.state,H_SU2xSU2,H_SU2xSU2,g_SU2xSU2.state,qarray<2>{1,1},2,1) - std::pow(g_SU2xSU2.energy,3);
		check2(2-1) = avg(g_SU2xSU2.state,H_SU2xSU2,H_SU2xSU2,g_SU2xSU2.state,qarray<2>{1,1},1,1) - std::pow(g_SU2xSU2.energy,2);
		for (size_t i=1; i<=maxPower;i++)
		{
			check(i-1) = avg(g_SU2xSU2.state,H_SU2xSU2,g_SU2xSU2.state,i) - std::pow(g_SU2xSU2.energy,i);
		}
		cout << "check=" << check.transpose() << endl;
		cout << "check2=" << check2.transpose() << endl;
				
		cout << "vol=" << Vol << ", N=" << N << endl;
		Emin_SU2xSU2 = g_SU2xSU2.energy-0.5*U*(Vol-N);
		emin_SU2xSU2 = Emin_SU2xSU2/Vol;
		t_SU2xSU2 = Watch_SU2xSU2.time();
		
		if (CORR)
		{
//			Eigenstate<VMPS::HubbardSU2xSU2::StateXd> g_SU2xSU2m;
//			DMRG_SU2xSU2.set_verbosity(DMRG::VERBOSITY::SILENT);
//			DMRG_SU2xSU2.edgeState(H_SU2xSU2, g_SU2xSU2m, {abs(Nup-1-Ndn)+1,V-(N)+2}, LANCZOS::EDGE::GROUND, DMRG::CONVTEST::VAR_2SITE,
//			                    tol_eigval,tol_state, Dinit,Dlimit, Imax,Imin, alpha);
//			lout << "g_SU2xSU2m.energy=" << g_SU2xSU2m.energy-0.5*U*(V-Nup+1-Ndn) << endl;
//			
//			ArrayXd c_SU2xSU2(L);
//			for (int l=0; l<L; ++l)
//			{
//				c_SU2xSU2(l) = avg(g_SU2xSU2m.state, H_SU2xSU2.c(l), g_SU2xSU2.state);
//				cout << "l=" << l << ", <c>=" << c_SU2xSU2(l) << "\t" << c_SU2xSU2(l)/c_U1(l) << endl;
//			}
			
			for (size_t i=0; i<L; ++i) 
			for (size_t j=0; j<L; ++j)
			{
				densityMatrix_SU2xSU2A(i,j) = 0.5*avg(g_SU2xSU2.state, H_SU2xSU2.cdagc(i,j), g_SU2xSU2.state);
			}
			
			for (size_t i=0; i<L; ++i) 
			for (size_t j=0; j<L; ++j)
			{
				//factor 1/2 because we have computed cdagc+cdagc
				densityMatrix_SU2xSU2B(i,j) = 0.5*avg(g_SU2xSU2.state, H_SU2xSU2.cdag(i,0,2.), H_SU2xSU2.c(j), g_SU2xSU2.state);
			}
			
			for (size_t i=0; i<L; ++i) 
			for (size_t j=0; j<L; ++j)
			{
				spin_SU2xSU2(i,j) = avg(g_SU2xSU2.state, H_SU2xSU2.SdagS(i,j), g_SU2xSU2.state);
			}
			for (size_t i=0; i<L; ++i) 
			for (size_t j=0; j<L; ++j)
			{
				isospin_SU2xSU2(i,j) = avg(g_SU2xSU2.state, H_SU2xSU2.TdagT(i,j), g_SU2xSU2.state);
			}
		//	lout << "P SU(2): " << Ptot(0.5*densityMatrix_SU2xSU2,L) << "\t" << Ptot(0.5*densityMatrix_SU2xSU2B,L) << endl;
			
			for (size_t i=0; i<L; ++i) 
			{
				nh_SU2xSU2(i) = avg(g_SU2xSU2.state, H_SU2xSU2.nh(i), g_SU2xSU2.state);
				ns_SU2xSU2(i) = avg(g_SU2xSU2.state, H_SU2xSU2.ns(i), g_SU2xSU2.state);
			}
//			lout << "<nh>=" << endl << nh_SU2xSU2 << endl;
//			lout << "error(<nh>=<h>+<d>)=" << (nh_SU2xSU2-d_ED-h_ED).matrix().norm() << endl;
		}
	}
	
	if (PRINT)
	{
		lout << endl << termcolor::blue << "--------Observables---------" << termcolor::reset << endl << endl;
		
		// cout << "density matrix ED: " << endl;
		// cout << densityMatrix_ED << endl << endl;
		
		// cout << "density matrix U(1)⊗U(1) A: " << endl;
		// cout << densityMatrix_U1A << endl << endl;
		// cout << "density matrix U(1)⊗U(1) B: " << endl;
		// cout << densityMatrix_U1B << endl << endl;
		
		cout << "spin matrix SU(2)⊗U(1) A: " << endl;
		cout << spin_SU2 << endl << endl;
		cout << isospin_SU2 << endl << endl;
		cout << "density matrix SU(2)⊗U(1) B: " << endl;
		cout << densityMatrix_SU2B << endl << endl;
		
		cout << "spin matrix SU(2)⊗SU(2) A: " << endl;
		cout << spin_SU2xSU2 << endl << endl;
		cout << isospin_SU2xSU2 << endl << endl;
		cout << "density matrix SU(2)⊗SU(2) B: " << endl;
		cout << densityMatrix_SU2xSU2B << endl << endl;
	}
	
	//--------output---------
	TextTable T( '-', '|', '+' );
	
	T.add("");
	T.add("ED");
	T.add("U(0)");
	T.add("U(1)⊗U(1)");
	T.add("SU(2)⊗U(1)");
	T.add("SU(2)⊗SU(2)");
	T.endOfRow();
	
	T.add("E/V");
	T.add(to_string_prec(g_ED.energy/Vol));
	T.add(to_string_prec(emin_U0));
	T.add(to_string_prec(g_U1.energy/Vol));
	T.add(to_string_prec(g_SU2.energy/Vol));
	T.add(to_string_prec(emin_SU2xSU2));
	T.endOfRow();
	
	T.add("E/V diff");
	T.add("-");
	T.add(to_string_prec(abs(Emin_U0-g_ED.energy)/Vol,true));
	T.add(to_string_prec(abs(g_U1.energy-g_ED.energy)/Vol,true));
	T.add(to_string_prec(abs(g_SU2.energy-g_ED.energy)/Vol,true));
	T.add(to_string_prec(abs(Emin_SU2xSU2-g_ED.energy)/Vol,true));
	T.endOfRow();
	
	T.add("t/s");
	T.add("-");
	T.add(to_string_prec(t_U0,false,2));
	T.add(to_string_prec(t_U1,false,2));
	T.add(to_string_prec(t_SU2,false,2));
	T.add(to_string_prec(t_SU2xSU2,false,2));
	T.endOfRow();
	
	T.add("t gain");
	T.add("-");
	T.add(to_string_prec(t_U0/t_SU2,false,2));
	T.add(to_string_prec(t_U1/t_SU2,false,2));
	T.add("1");
	T.add(to_string_prec(t_SU2xSU2/t_SU2,false,2));
	T.endOfRow();
	
	if (CORR)
	{
		T.add("d diff");
		T.add("0");
		T.add("-");
		T.add(to_string_prec((d_U1-d_ED).norm(),true));
		T.add(to_string_prec((d_SU2-d_ED).norm(),true));
		T.add(to_string_prec((nh_SU2xSU2-d_ED-h_ED).norm(),true));
		T.endOfRow();
		
		T.add("rhoA diff");
		T.add("0");
		T.add("-");
		T.add(to_string_prec((densityMatrix_U1A-densityMatrix_ED).norm(),true));
		T.add(to_string_prec((densityMatrix_SU2A-densityMatrix_ED).norm(),true));
		T.add(to_string_prec((densityMatrix_SU2xSU2A-densityMatrix_ED).norm(),true));
		T.endOfRow();
		
		T.add("rhoB diff");
		T.add("0");
		T.add("-");
		T.add(to_string_prec((densityMatrix_U1B-densityMatrix_ED).norm(),true));
		T.add(to_string_prec((densityMatrix_SU2B-densityMatrix_ED).norm(),true));
		T.add(to_string_prec((densityMatrix_SU2xSU2B-densityMatrix_ED).norm(),true));
		T.endOfRow();
	}
	
	T.add("Dmax");
	T.add("-");
	T.add(to_string(g_U0.state.calc_Dmax()));
	T.add(to_string(g_U1.state.calc_Dmax()));
	T.add(to_string(g_SU2.state.calc_Dmax()));
	T.add(to_string(g_SU2xSU2.state.calc_Dmax()));
	T.endOfRow();
	
	T.add("Mmax");
	T.add("-");
	T.add(to_string(g_U0.state.calc_Dmax()));
	T.add(to_string(g_U1.state.calc_Mmax()));
	T.add(to_string(g_SU2.state.calc_Mmax()));
	T.add(to_string(g_SU2xSU2.state.calc_Mmax()));
	T.endOfRow();
	
	lout << endl << T;
	
	lout << "ref=" << VMPS::Hubbard::ref({{"n",static_cast<double>((N)/V)},{"U",U},{"t",t},{"Ly",Ly},{"tRung",tRung},{"tPrime",tPrime}}) << endl;
}

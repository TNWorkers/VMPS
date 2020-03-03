#define LANCZOS_MAX_ITERATIONS 1e2

#define USE_HDF5_STORAGE
#define DMRG_DONT_USE_OPENMP
#define MPSQCOMPRESSOR_DONT_USE_OPENMP


#include <iostream>
#include <fstream>
#include <complex>

#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

#include <gsl/gsl_sf_ellint.h>

#include "Logger.h"
Logger lout;
#include "ArgParser.h"

#include "util/LapackManager.h"

#include "StringStuff.h"
#include "Stopwatch.h"

#include <Eigen/Core>
using namespace Eigen;

//-------- Test of the ArnoldiSolver:
//size_t dim (const MatrixXcd &A) {return A.rows();}
//#include "LanczosWrappers.h"
//#include "HxV.h"

#include "VUMPS/VumpsSolver.h"
#include "VUMPS/VumpsLinearAlgebra.h"
#include "VUMPS/UmpsCompressor.h"
#include "models/Heisenberg.h"
#include "models/HeisenbergU1.h"
#include "models/HeisenbergU1XXZ.h"
#include "models/HeisenbergXXZ.h"
#include "models/HeisenbergSU2.h"
// #include "models/HubbardU1xU1.h"
// #include "models/Hubbard.h"
// #include "models/HubbardSU2xSU2.h"
// #include "models/HubbardSU2xU1.h"
// #include "models/KondoSU2xU1.h"
// #include "models/KondoU1xU1.h"
// #include "models/KondoU0xSU2.h"
#include "models/ParamCollection.h"

double Jxy, Jz, J, Jprime, Jprimeprime, Jrung, Jsmall, tPrime, Bx, Bz;
double U, mu;
double dt;
double e_exact;
size_t L, Ly;
int N;
size_t Dinit, max_iter, min_iter;
double tol_eigval, tol_var, tol_state;
bool ISING, HEIS2, HEIS3, SSH, ALL;

//-------- Ising model integrations:
// reference: Pfeuty, Annals of Physics 57, 79-90, 1970
//double IsingGroundIntegrand (double x, void*)
//{
//	if (Bx==0.)
//	{
//		return -0.25;
//	}
//	else
//	{
//		double lambda = Jz/(2.*Bx);
//		double thetasq = 4.*lambda/pow(1.+lambda,2);
//		return -(Bx+0.5*Jz) * M_1_PI * sqrt(1.-thetasq*pow(sin(x),2));
//	}
//}

double IsingGround (double Jz, double Bx)
{
	if (Bx == 0.)
	{
		return -0.25;
	}
	else
	{
		double lambda = Jz/(2.*Bx);
		double theta = sqrt(4.*lambda/pow(1.+lambda,2));
		return -(Bx+0.5*Jz) * M_1_PI *  gsl_sf_ellint_E(0.5*M_PI,theta,GSL_PREC_DOUBLE);
	}
}

// SSH model integrations
struct eSSH
{
	static double v;
	static double w;
	
	static double f (double x, void*)
	{
		return -M_1_PI*sqrt(v*v+w*w+2.*v*w*cos(2.*x));
	}
};
double eSSH::v = -1.;
double eSSH::w = -1.;

template<typename Hamiltonian, typename Eigenstate>
void print_mag (const Hamiltonian &H, const Eigenstate &g)
{
	VectorXd SZcell(L);
	VectorXd SXcell(L);
	lout << endl;
	lout << "magnetization within unit cell: " << endl;
	
	for (size_t l=0; l<L; ++l)
	{
		SZcell(l) = avg(g.state, H.Sz(l), g.state);
		SXcell(l) = avg(g.state, H.Scomp(SX,l), g.state);
		lout << "<Sz("<<l<<")>=" << SZcell(l) << endl;
		lout << "<Sx("<<l<<")>=" << SXcell(l) << endl;
	}
	
	lout << "total magnetization: " << endl;
	lout << "<Sz>=" << SZcell.sum() << endl;
	lout << "<Sx>=" << SXcell.sum() << endl;
}

template<typename MpsType, typename MpoType>
double SvecSvecAvg (const MpsType &Psi, const MpoType &H, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0)
{
	return isReal(avg(Psi,H.SzSz(locx1,locx2,locy1,locy2),Psi))+
	       0.5*(isReal(avg(Psi,H.SpSm(locx1,locx2,locy1,locy2),Psi))+
	            isReal(avg(Psi,H.SmSp(locx1,locx2,locy1,locy2),Psi)));
}

int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",2);
	Ly = args.get<size_t>("Ly",1);
	Jxy = args.get<double>("Jxy",1.);
	Jz = args.get<double>("Jz",1.);
	J = args.get<double>("J",1.);
	Jsmall = args.get<double>("Jsmall",0.);
	Jrung = args.get<double>("Jrung",J);
	Jprime = args.get<double>("Jprime",0.);
	Jprimeprime = args.get<double>("Jprimeprime",0.);
	tPrime = args.get<double>("tPrime",0.);
	Bx = args.get<double>("Bx",1.);
	Bz = args.get<double>("Bz",0.);
	U = args.get<double>("U",10.);
	mu = args.get<double>("mu",0.5*U);
	N = args.get<int>("N",L);
	
	dt = args.get<double>("dt",0.5); // hopping-offset for SSH model
	Dinit = args.get<double>("Dinit",5);    // bond dimension
	tol_eigval = args.get<double>("tol_eigval",1e-9);
	tol_var = args.get<double>("tol_var",1e-8);
	tol_state = args.get<double>("tol_state",1e-7);

	max_iter = args.get<size_t>("max_iter",300ul);
	min_iter = args.get<size_t>("min_iter",1ul);
	size_t Qinit = args.get<size_t>("Qinit",6);
	size_t D = args.get<size_t>("D",3);

	VUMPS::CONTROL::GLOB GlobParams;
	GlobParams.min_iterations = min_iter;
	GlobParams.max_iterations = max_iter;
	GlobParams.Dinit = Dinit;
	GlobParams.Qinit = Qinit;
	GlobParams.tol_eigval = tol_eigval;
	GlobParams.tol_var = tol_var;
	GlobParams.tol_state = tol_state;

	VUMPS::CONTROL::LANCZOS LanczosParams;
	LanczosParams.eps_eigval = 1.e-12;
	LanczosParams.eps_coeff = 1.e-12;
	// VUMPS::CONTROL::DYN DynParams;
	// DynParams.max_deltaD = [] (size_t i) {return (i<lim)? tmp1:0.;};
	// DynParams.Dincr_abs  = [] (size_t i) {return Dabs;};

//	-------- Test of the ArnoldiSolver:
//	MatrixXd A(100,100);
//	A.setRandom();
//	A.triangularView<Upper>() = A.adjoint();
//	SelfAdjointEigenSolver<MatrixXd> Eugen(A);
//	cout << Eugen.eigenvalues().head(5).transpose() << endl;
//	complex<double> lambda;
//	VectorXcd v(100);
//	ArnoldiSolver<MatrixXd,VectorXcd> Arnie(A,v,lambda);
//	cout << Arnie.info() << endl;
	
	bool CALC_SU2 = args.get<bool>("SU2",true);
	bool CALC_U1 = args.get<bool>("U1",true);
	bool CALC_U0 = args.get<bool>("U0",true);
	bool CALC_HUBB = args.get<bool>("HUBB",false);
	bool CALC_KOND = args.get<bool>("KOND",false);
	bool CALC_DOT = args.get<bool>("DOT",false);
	ALL   = args.get<bool>("ALL",false);
	if (ALL)
	{
		CALC_SU2  = true;
		CALC_U1   = true;
		CALC_U0   = true;
		CALC_HUBB = true;
		CALC_KOND = true;
	}
		
	DMRG::VERBOSITY::OPTION VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",2));
	UMPS_ALG::OPTION ALG = static_cast<UMPS_ALG::OPTION>(args.get<int>("ALG",0));
	lout << args.info() << endl;
//	lout.set(make_string("L=",L,"_Nup=",Nup,"_Ndn=",Ndn,".log"),"log");
	
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
		
	typedef VMPS::HeisenbergSU2 HEISENBERG_SU2;

	HEISENBERG_SU2::uSolver DMRG_SU2(VERB);
	Eigenstate<HEISENBERG_SU2::StateUd> g_SU2;
	if (CALC_SU2)
	{
		HEISENBERG_SU2 Heis_SU2;
		Heis_SU2 = HEISENBERG_SU2(L,{{"Ly",Ly},{"J",J},{"Jprime",Jprime},{"OPEN_BC",false},{"CALC_SQUARE",false},{"D",D}}, BC::INFINITE);
		lout << Heis_SU2.info() << endl;
		DMRG_SU2.set_log(L,"e_Heis_SU2.dat","err_eigval_Heis_SU2.dat","err_var_Heis_SU2.dat","err_state_Heis_SU2.dat");
		DMRG_SU2.userSetGlobParam();
		DMRG_SU2.userSetLanczosParam();
		DMRG_SU2.LanczosParam = LanczosParams;
		DMRG_SU2.GlobParam = GlobParams;
		DMRG_SU2.edgeState(Heis_SU2, g_SU2, {1});
	}
	
	typedef VMPS::HeisenbergU1 HEISENBERG_U1;
	HEISENBERG_U1 Heis_U1;
	Heis_U1 = HEISENBERG_U1(L,{{"Ly",Ly},{"Jxy",Jxy},{"Jz",Jz},{"Jprime",Jprime},{"Bz",Bz},{"D",D}},BC::INFINITE);
	
	HEISENBERG_U1 Heis_U1_(L,{{"Ly",Ly},{"Jxy",Jxy},{"Jz",1.2*Jz},{"Jprime",Jprime},{"Bz",Bz},{"D",D}},BC::INFINITE);
	
	HEISENBERG_U1::uSolver DMRG_U1(VERB);
	Eigenstate<HEISENBERG_U1::StateUd> g_U1;
	if (CALC_U1)
	{
		lout << Heis_U1.info() << endl;
		DMRG_U1.set_log(2,"e_Heis_U1.dat","err_eigval_Heis_U1.dat","err_var_Heis_U1.dat","err_state_Heis_U1.dat");
		DMRG_U1.userSetGlobParam();
		DMRG_U1.userSetLanczosParam();
		DMRG_U1.LanczosParam = LanczosParams;
		DMRG_U1.GlobParam = GlobParams;
		DMRG_U1.edgeState(Heis_U1, g_U1, {0});
		
		if (CALC_DOT)
		{
			HEISENBERG_U1::uSolver DMRG_U1_(VERB);
			Eigenstate<HEISENBERG_U1::StateUd> g_U1_;
			DMRG_U1_.set_log(2,"e_Heis_U1_.dat","err_eigval_Heis_U1_.dat","err_var_Heis_U1_.dat","err_state_Heis_U1_.dat");
			DMRG_U1_.userSetGlobParam();
			DMRG_U1_.GlobParam = GlobParams;
			DMRG_U1_.edgeState(Heis_U1_, g_U1_, {0});
			double dot1 = g_U1.state.dot(g_U1.state);
			double dot2 = g_U1_.state.dot(g_U1_.state);
			double dot3 = g_U1.state.dot(g_U1_.state);
			cout << "<ψ|ψ>=" <<  dot1 
			     << ", <φ|φ>=" << dot2 
			     << ", <ψ|φ>=" << dot3 
			<< endl;
		}
	}

// 	typedef VMPS::KondoSU2xU1 KONDO;
// 	KONDO Kond(L,{{"t",1.},{"tPrime",tPrime},{"U",U},{"J",J},{"OPEN_BC",false}});
// 	KONDO::uSolver DMRG_KOND(VERB);

// 	if (CALC_KOND)
// 	{
// 		cout << Kond.info() << endl;

// 		qarray<2> Qc = {1,N};
// 		Kond.transform_base(Qc);
// 		Eigenstate<KONDO::StateUd> g_Kond;
// 		DMRG_KOND.set_log(2,"e_Kond.dat","err_eigval_Kond.dat","err_var_Kond.dat","err_state_Kond.dat");
// 		DMRG_KOND.userSetGlobParam();
// 		DMRG_KOND.GlobParam = GlobParams;
// 		DMRG_KOND.edgeState(Kond, g_Kond, Qc);

// 		cout << termcolor::bold << "e0=" << g_Kond.energy << termcolor::reset << endl;
// 		ArrayXd nvec(L), dvec(L);
// 		for (size_t l=0; l<L; ++l)
// 		{
// 			cout << "l=" << l << endl;
			
// 			nvec(l) = avg(g_Kond.state, Kond.n(l), g_Kond.state);
// 			cout << "n=" << nvec(l) << endl;
			
// 			dvec(l) = avg(g_Kond.state, Kond.d(l), g_Kond.state);
// 			cout << "d=" << dvec(l) << endl;	
// 		}
// 		cout << "SimpSimp(0,1)=" <<  avg(g_Kond.state, Kond.SimpSimp(0,1), g_Kond.state) << endl;
// 		cout << "SimpSimp(1,0)=" <<  avg(g_Kond.state, Kond.SimpSimp(1,0), g_Kond.state) << endl;

// 		KONDO Kond_4(8,{{"t",1.},{"tPrime",tPrime},{"U",U},{"J",J},{"OPEN_BC",false}});
// 		cout << "SimpSimp(0,1)=" <<  avg(g_Kond.state, Kond_4.SimpSimp(0,1), g_Kond.state) << endl;
// 		cout << "SimpSimp(1,2)=" <<  avg(g_Kond.state, Kond_4.SimpSimp(1,2), g_Kond.state) << endl;
// 		cout << "SimpSimp(2,3)=" <<  avg(g_Kond.state, Kond_4.SimpSimp(2,3), g_Kond.state) << endl;
// 		cout << "SimpSimp(0,2)=" <<  avg(g_Kond.state, Kond_4.SimpSimp(0,2), g_Kond.state) << endl;
		
// 		cout << "navg=" << nvec.sum()/L << endl;
// 		cout << "davg=" << dvec.sum()/L << endl;
// 	}

// 	typedef VMPS::HubbardSU2xU1 HUBBARD;
// 	HUBBARD Hubb(L,{{"t",1.},{"tPrime",tPrime},{"U",U},{"J",J},{"OPEN_BC",false}});
// 	HUBBARD::uSolver DMRG_HUBB(VERB);

// 	if (CALC_HUBB)
// 	{
// 		cout << Hubb.info() << endl;
		
// 		qarray<2> Qc = {1,N};
// 		Hubb.transform_base(Qc);
// 		Eigenstate<HUBBARD::StateUd> g_Hubb;
// 		DMRG_HUBB.set_log(2,"e_Hubb.dat","err_eigval_Hubb.dat","err_var_Hubb.dat","err_state_Hubb.dat");
// 		DMRG_HUBB.userSetGlobParam();
// 		DMRG_HUBB.GlobParam = GlobParams;
// 		DMRG_HUBB.edgeState(Hubb, g_Hubb, Qc);
// 		double e_exact = VMPS::Hubbard::ref({{"U",U}}).value;
// 		cout << "e0=" << g_Hubb.energy << endl;
// 		cout << "e_exact=" << e_exact << ", diff=" << abs(e_exact-g_Hubb.energy) << endl;
// 		ArrayXd nvec(L), dvec(L);
// 		for (size_t l=0; l<L; ++l)
// 		{
// 			cout << "l=" << l << endl;
			
// 			nvec(l) = avg(g_Hubb.state, Hubb.n(l), g_Hubb.state);
// 			cout << "n=" << nvec(l) << endl;
			
// 			dvec(l) = avg(g_Hubb.state, Hubb.d(l), g_Hubb.state);
// 			cout << "d=" << dvec(l) << endl;
// 		}
		
// 		cout << "navg=" << nvec.sum()/L << endl;
// 		cout << "davg=" << dvec.sum()/L << endl;
// 	}
	
	
	typedef VMPS::HeisenbergXXZ HEISENBERG0;
	HEISENBERG0 Heis0;
	Heis0 = HEISENBERG0(L,{{"Ly",Ly},{"Jxy",Jxy},{"Jz",Jz},{"Jprime",Jprime},{"Bz",Bz},{"D",D}},BC::INFINITE);

	HEISENBERG0 Heis0_(L,{{"Ly",Ly},{"Jxy",Jxy},{"Jz",1.2*Jz},{"Jprime",Jprime},{"Bz",Bz},{"D",D}},BC::INFINITE);
	
	HEISENBERG0::uSolver DMRG0(VERB);
	Eigenstate<HEISENBERG0::StateUd> g0;
	if (CALC_U0)
	{
		lout << Heis0.info() << endl;
		DMRG0.set_log(2,"e_Heis_U0.dat","err_eigval_Heis_U0.dat","err_var_Heis_U0.dat","err_state_Heis_U0.dat");
		DMRG0.userSetGlobParam();
		DMRG0.GlobParam = GlobParams;
		DMRG0.edgeState(Heis0, g0, {});
		cout << g0.state.info() << endl;
		if (CALC_DOT)
		{
			HEISENBERG0::uSolver DMRG0_(VERB);
			Eigenstate<HEISENBERG0::StateUd> g0_;
			DMRG0_.userSetGlobParam();
			DMRG0_.GlobParam = GlobParams;
			DMRG0_.edgeState(Heis0_, g0_, {});
			double dot1 = g0.state.dot(g0.state);
			double dot2 = g0_.state.dot(g0_.state);
			double dot3 = g0.state.dot(g0_.state);
			cout << "<ψ|ψ>=" <<  dot1 
			     << ", <φ|φ>=" << dot2 
			     << ", <ψ|φ>=" << dot3 << endl;
		}
		
//		typedef VMPS::Hubbard HUBBARD_U0;
//		HUBBARD_U0 Hubb_U0(L,{{"U",U},{"mu",mu},{"OPEN_BC",false}});
//		qarray<0> Qc = {};
//		Hubb_U0.transform_base(Qc);
//		cout << Hubb_U0.info() << endl;
//		HUBBARD_U0::uSolver DMRG_HUBBU0(VERB);
//		Eigenstate<HUBBARD_U0::StateUd> g_U0Hubb;
//		DMRG_HUBBU0.set_log(2,"e_Hubb_U0.dat","err_eigval_Hubb_U0.dat","err_var_Hubb_U0.dat");
//		DMRG_HUBBU0.edgeState(Hubb_U0, g_U0Hubb, Qc, tol_eigval,tol_var, M, Nqmax, max_iter,1);
////		cout << "exact=" << -0.2671549218961211 << endl;
//		double e_exact = LiebWu_E0_L(U,0.01*tol_eigval);
//		cout << "e_exact=" << e_exact-mu << endl;
//		for (size_t l=0; l<L; ++l)
//		{
//			cout << "l=" << l << endl;
//			cout << "n=" << avg(g_U0Hubb.state, Hubb_U0.n(l), g_U0Hubb.state) << endl;
//			cout << "d=" << avg(g_U0Hubb.state, Hubb_U0.d(l), g_U0Hubb.state) << endl;
//			cout << "Sz=" << avg(g_U0Hubb.state, Hubb_U0.Sz(l), g_U0Hubb.state) << endl;
//		}
//		cout << "n(0)n(1)=" << avg(g_U0Hubb.state, Hubb_U0.nn<UPDN,UPDN>(0,1), g_U0Hubb.state) << endl;
	}
	
	cout << setprecision(13);
	refEnergy lit = VMPS::Heisenberg::ref({{"J",J},{"Jprime",Jprime},{"Ly",Ly},{"D",D}}); 
	cout << "e(ref) =" << lit.value << "\t from: " << lit.source << endl;
	if (CALC_SU2) {cout << "e0(SU2)=" << g_SU2.energy << ", diff=" << abs(lit.value-g_SU2.energy) << endl;}
	if (CALC_U1)  {cout << "e0(U1) =" << g_U1.energy << ", diff=" << abs(lit.value-g_U1.energy) << endl;}
	if (CALC_U0)  {cout << "e0(U0) =" << g0.energy << ", diff=" << abs(lit.value-g0.energy) << endl;}
	
	// if (CALC_U0)
	// {
	// 	cout << "-----U0-----" << endl;
	// 	print_mag(Heis0,g0);
	// 	size_t dmax = 10;
	// 	for (size_t d=1; d<dmax; ++d)
	// 	{
	// 		HEISENBERG0 Htmp(d+1,{{"Ly",Ly},{"J",J},{"OPEN_BC",false},{"D",D}});
	// 		double SvecSvec = SvecSvecAvg(g0.state,Htmp,0,d);
	// 		// if (d == L) {lout << "l=" << d-1 << ", " << d << ", <SvecSvec>=" << SvecSvecAvg(g0.state,Htmp,d-1,d) << endl;}
	// 		lout << "d=" << d << ", <SvecSvec>=" << SvecSvec << endl;
	// 	}

	// 	g0.state.truncate();
	// 	cout << endl << endl << "after truncation" << endl;
	// 	cout << g0.state.info() << endl;
	// 	print_mag(Heis0,g0);

	// 	// print_mag(Heis0,g0);
	// 	// for (size_t d=1; d<dmax; ++d)
	// 	// {
	// 	// 	HEISENBERG0 Htmp(d+1,{{"Ly",Ly},{"J",J},{"OPEN_BC",false},{"D",D}});
	// 	// 	double SvecSvec = SvecSvecAvg(g0.state,Htmp,0,d);
	// 	// 	// if (d == L) {lout << "l=" << d-1 << ", " << d << ", <SvecSvec>=" << SvecSvecAvg(g0.state,Htmp,d-1,d) << endl;}
	// 	// 	lout << "d=" << d << ", <SvecSvec>=" << SvecSvec << endl;
	// 	// }

	// 	// Umps<Sym::U0,complex<double> > g0_compl = g0.state.template cast<complex<double> >();
	// 	// Umps<Sym::U0,complex<double> > g0_trunc_compl;
	// 	// UmpsCompressor<Sym::U0,complex<double>, complex<double> > Lana(DMRG::VERBOSITY::HALFSWEEPWISE);
	// 	// Lana.stateCompress(g0_compl, g0_trunc_compl, 25ul, 1ul, 1.e-5, 50ul);
	// 	// Umps<Sym::U0,double> g0_trunc = g0_trunc_compl.real();
	// 	// cout << endl << endl << "after truncation" << endl;
	// 	// cout << g0_trunc_compl.info() << endl;
		
	// 	for (size_t d=1; d<dmax; ++d)
	// 	{
	// 		HEISENBERG0 Htmp(d+1,{{"Ly",Ly},{"J",J},{"OPEN_BC",false},{"D",D}});
	// 		double SvecSvec = SvecSvecAvg(g0.state,Htmp,0,d);
	// 		// if (d == L) {lout << "l=" << d-1 << ", " << d << ", <SvecSvec>=" << SvecSvecAvg(g0.state,Htmp,d-1,d) << endl;}
	// 		lout << "d=" << d << ", <SvecSvec>=" << SvecSvec << endl;
	// 	}

	// }
	// if (CALC_U1)
	// {
	// 	cout << endl;
	// 	cout << "-----U1-----" << endl;
	// 	print_mag(Heis_U1,g_U1);
	// 	size_t dmax = 10;
	// 	for (size_t d=1; d<dmax; ++d)
	// 	{
	// 		HEISENBERG_U1 Htmp(d+1,{{"Ly",Ly},{"Jxy",Jxy},{"Jz",Jz},{"OPEN_BC",false},{"D",D}});
	// 		double SvecSvec = SvecSvecAvg(g_U1.state,Htmp,0,d);
	// 		lout << "d=" << d << ", <SvecSvec>=" << SvecSvec << endl;
	// 	}

	// 	g_U1.state.truncate();
	// 	cout << endl << endl << "after truncation" << endl;
	// 	cout << g_U1.state.info() << endl;
	// 	print_mag(Heis_U1,g_U1);

	// 	for (size_t d=1; d<dmax; ++d)
	// 	{
	// 		HEISENBERG_U1 Htmp(d+1,{{"Ly",Ly},{"Jxy",Jxy},{"Jz",Jz},{"OPEN_BC",false},{"D",D}});
	// 		double SvecSvec = SvecSvecAvg(g_U1.state,Htmp,0,d);
	// 		lout << "d=" << d << ", <SvecSvec>=" << SvecSvec << endl;
	// 	}
		
	// 	g_U1.state.graph("g");
	// }
	
	if (CALC_SU2)
	{
		cout << endl;
		cout << "-----SU2-----" << endl;
		cout << g_SU2.state.info() << endl;
		size_t dmax = 10;
		for (size_t d=1; d<dmax; ++d)
		{
			HEISENBERG_SU2 Htmp(d+1,{{"Ly",Ly},{"J",J},{"CALC_SQUARE",false},{"D",D}}, BC::INFINITE);
			double SvecSvec = avg(g_SU2.state,Htmp.SdagS(0,d),g_SU2.state);
			// if (d == L)
			// {
			// 	lout << "l=" << d-4 << ", " << d-3 << ", <SvecSvec>=" << avg(g_SU2.state,Htmp.SdagS(d-4,d-3),g_SU2.state) << endl;
			// 	lout << "l=" << d-3 << ", " << d-2 << ", <SvecSvec>=" << avg(g_SU2.state,Htmp.SdagS(d-3,d-2),g_SU2.state) << endl;
			// 	lout << "l=" << d-2 << ", " << d-1 << ", <SvecSvec>=" << avg(g_SU2.state,Htmp.SdagS(d-2,d-1),g_SU2.state) << endl;
			// 	lout << "l=" << d-1 << ", " << d << ", <SvecSvec>=" << avg(g_SU2.state,Htmp.SdagS(d-1,d),g_SU2.state) << endl;
			// }

			// double SvecSvec = Htmp.SvecSvecAvg(g.state,0,d);
			lout << "d=" << d << ", <SvecSvec>=" << SvecSvec << endl;
		}

		g_SU2.state.truncate(false);
		cout << endl << endl << "after truncation" << endl;
		cout << g_SU2.state.info() << endl;

		// print_mag(Heis,g);
		for (size_t d=1; d<dmax; ++d)
		{
			HEISENBERG_SU2 Htmp(d+1,{{"Ly",Ly},{"J",J},{"CALC_SQUARE",false},{"D",D}}, BC::INFINITE);
			double SvecSvec = avg(g_SU2.state,Htmp.SdagS(0,d),g_SU2.state);
			// if (d == L)
			// {
			// 	lout << "l=" << d-4 << ", " << d-3 << ", <SvecSvec>=" << avg(g_SU2.state,Htmp.SdagS(d-4,d-3),g_SU2.state) << endl;
			// 	lout << "l=" << d-3 << ", " << d-2 << ", <SvecSvec>=" << avg(g_SU2.state,Htmp.SdagS(d-3,d-2),g_SU2.state) << endl;
			// 	lout << "l=" << d-2 << ", " << d-1 << ", <SvecSvec>=" << avg(g_SU2.state,Htmp.SdagS(d-2,d-1),g_SU2.state) << endl;
			// 	lout << "l=" << d-1 << ", " << d << ", <SvecSvec>=" << avg(g_SU2.state,Htmp.SdagS(d-1,d),g_SU2.state) << endl;
			// }

			// double SvecSvec = Htmp.SvecSvecAvg(g.state,0,d);
			lout << "d=" << d << ", <SvecSvec>=" << SvecSvec << endl;
		}
	
		g_SU2.state.graph("g");
	}
//	//---<transverse Ising>---
//	if (ISING)
//	{
//		XXZ Ising(L,{{"Jz",Jz},{"Bx",Bx},{"OPEN_BC",false}});
//		DMRG.set_log(2,"e_Ising.dat","err_eigval_Ising.dat","err_var_Ising.dat");
//		lout << Ising.info() << endl;
//		
//	//	DMRG.edgeState(Ising, g, {}, tol_eigval,tol_var, M, max_iter,1);
//		
//		e_exact = IsingGround(Jz,Bx); // integrate(IsingGroundIntegrand, 0.,0.5*M_PI, 1e-10,1e-10);
//	//	e_exact = -1.0635444099809814;
//		lout << TCOLOR(BLUE);
//		lout << "Transverse Ising: e0=" << g.energy << ", exact:" << e_exact << ", diff=" << abs(g.energy-e_exact) << endl;
//	
//		for (size_t l=0; l<Ising.length(); ++l)
//		{
//			lout << "<Sz("<<l<<")>=" << avg(g.state, Ising.Sz(l), g.state) << endl;
//			lout << "<Sx("<<l<<")>=" << avg(g.state, Ising.Sx(l), g.state) << endl;
//		}
//		lout << TCOLOR(BLACK) << endl;
//	}
	
	//---<Heisenberg S=1/2>---
//	if (HEIS2)
//	{
//		HEISENBERG Heis(L,{{"Bz",Bz},{"OPEN_BC",false}});
//		DMRG.set_log(2,"e_HeisS1_2.dat","err_eigval_HeisS1_2.dat","err_var_HeisS1_2.dat");
//		lout << Heis.info() << endl;
//		
//		DMRG.edgeState(Heis, g, {}, tol_eigval,tol_var, M, max_iter,1);
//		
//		e_exact = 0.25-log(2);
//		lout << TCOLOR(BLUE);
//		lout << "Heisenberg S=1/2: e0=" << g.energy << ", exact(Δ=0):" << e_exact << ", diff=" << abs(g.energy-e_exact) << endl;
//		print_mag(Heis,g);
//		for (size_t l=0; l<Heis.length(); ++l)
//		{
//			lout << "l=" << l << ", entropy=" << g.state.entropy(l);
//		}
//		lout << TCOLOR(BLACK) << endl;
//		
//		lout << "spin-spin correlations at distance d:" << endl;
//		size_t dmax = 10;
//		for (size_t d=1; d<dmax; ++d)
//		{
//			HEISENBERG Htmp(d+1,{{"Bz",Bz},{"Bx",Bx},{"OPEN_BC",false}});
//			double SvecSvec = Htmp.SvecSvecAvg(g.state,0,d);
//			lout << "d=" << d << ", <SvecSvec>=" << SvecSvec << endl;
//		}
//		lout << endl;
//	}
	
//	//---<Heisenberg S=1>---
//	if (HEIS3)
//	{
//		HEISENBERG Heis(L,{{"Bz",Bz},{"OPEN_BC",false},{"D",3ul}});
//		DMRG.set_log(2,"e_HeisS1.dat","err_eigval_HeisS1.dat","err_var_HeisS1.dat");
//		lout << Heis.info() << endl;
//		
//	//	DMRG.edgeState(Heis.H2site(0,true), Heis.locBasis(0), g, {}, tol_eigval,tol_var, M, max_iter,1);
//		DMRG.edgeState(Heis, g, {}, tol_eigval,tol_var, M, max_iter,1);
//		
//		e_exact = -1.40148403897122; // value from: Haegeman et al. PRL 107, 070601 (2011)
//		lout << TCOLOR(BLUE);
//		lout << "Heisenberg S=1: e0=" << g.energy << ", quasiexact:" << e_exact << ", diff=" << abs(g.energy-e_exact) << endl;
//		print_mag(Heis,g);
//		for (size_t l=0; l<Heis.length(); ++l)
//		{
//			lout << "l=" << l << ", entropy=" << g.state.entropy(l) << endl;
//		}
//		lout << TCOLOR(BLACK) << endl;
//		ofstream SchmidtFiler("Schmidt.dat");
//		if (L==1)
//		{
//			for (size_t i=0; i<M; ++i)
//			{
//				SchmidtFiler << i << "\t" << setprecision(16) << g.state.singularValues(0)(i) << endl;
//			}
//		}
//		SchmidtFiler.close();
//	}
//	
	//---<Hubbard>---
//	if (HUBB)
//	{
//		HUBBARD Hubb(L,{{"U",U},{"mu",mu},{"OPEN_BC",false}});
//		DMRG_HUBB.set_log(2,"e_Hubb.dat","err_eigval_Hubb.dat","err_var_Hubb.dat");
//		lout << Hubb.info() << endl;
//		
//		DMRG_HUBB.edgeState(Hubb, g, {}, tol_eigval,tol_var, M, max_iter,1);
//	//	DMRG_HUBB.edgeState(Hubb.H2site(0,true), Hubb.locBasis(0), g, {}, tol_eigval,tol_var, M, max_iter,1);
//		
//		lout << "half-filling test for μ=U/2: <n>=" << avg(g.state, Hubb.n(0), g.state) << endl;
//		e_exact = LiebWu_E0_L(U,0.01*tol_eigval)-mu;
//	//	e_exact = -0.2671549218961211-mu; // value from: Bethe ansatz code, U=10
//		lout << TCOLOR(BLUE);
//		lout << "Hubbard (half-filling): e0=" << g.energy << ", exact=" << e_exact << ", diff=" << abs(g.energy-e_exact) << endl;
//		print_mag(Hubb,g);
//		lout << TCOLOR(BLACK) << endl;
//	}
//	
//	//---<SSH model to test unit cell>---
//	if (SSH)
//	{
//		HUBBARD Hubb(max(L,2ul),{{"t",1.+dt,0},{"t",1.-dt,1},{"OPEN_BC",false}});
//		DMRG_HUBB.set_log(2,"e_SSH.dat","err_eigval_SSH.dat","err_var_SSH.dat");
//		lout << Hubb.info() << endl;
//		
//		DMRG_HUBB.edgeState(Hubb, g, {}, tol_eigval,tol_var, M, max_iter,1);
//		
//		double n = avg(g.state, Hubb.n(0), g.state);
//		lout << "<n>=" << n << endl;
//		lout << TCOLOR(BLUE);
//		eSSH::v = -(1.+dt);
//		eSSH::w = -(1.-dt);
//		e_exact = integrate(eSSH::f, -0.5*M_PI,+0.5*M_PI, 1e-10,1e-10);
//		
//		lout << "SSH e0=" << g.energy << ", exact=" << e_exact << endl;
//		lout << "diff=" << abs(g.energy-e_exact) << endl;
//		lout << TCOLOR(BLACK) << endl;
//	}
}

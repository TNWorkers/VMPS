#define DONT_USE_LAPACK_SVD
#define DONT_USE_LAPACK_QR
#define LANCZOS_MAX_ITERATIONS 1e2

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

//#include "LanczosWrappers.h"
#include "StringStuff.h"
#include "Stopwatch.h"

#include "tensors/Biped.h"
#include "VUMPS/Umps.h"

#include "VUMPS/VumpsSolver.h"
#include "VUMPS/VumpsLinearAlgebra.h"
//#include "solvers/DmrgSolver.h"
#include "models/Heisenberg.h"
#include "models/HeisenbergU1.h"
#include "models/HeisenbergU1XXZ.h"
#include "models/HeisenbergXXZ.h"
#include "models/HeisenbergSU2.h"
//#include "models/HeisenbergXXZ.h"
//#include "models/Hubbard.h"

// integration files not included in git
//#include "gsl/gsl_integration.h"
//#include "LiebWu.h"

double Jxy, Jz, J, Bx, Bz;
double U, mu;
double dt;
double e_exact;
size_t L;
size_t M, max_iter;
double tol_eigval, tol_var;
bool ISING, HEIS2, HEIS3, HUBB, SSH, ALL;

// Ising model integrations
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
	lout << "magnetization within unit cell (i.e. staggered): " << endl;
	
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

int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",1);
	size_t Ly = args.get<size_t>("Ly",1);
	Jxy = args.get<double>("Jxy",-1.);
	Jz = args.get<double>("Jz",-1.);
	J = args.get<double>("J",-1.);
	Bx = args.get<double>("Bx",1.);
	Bz = args.get<double>("Bz",0.);
	U = args.get<double>("U",10.);
	mu = args.get<double>("mu",0.5*U);
	
	dt = args.get<double>("dt",0.5); // hopping-offset for SSH model
	M = args.get<double>("M",10);    // bond dimension
	tol_eigval = args.get<double>("tol_eigval",1e-6);
	tol_var = args.get<double>("tol_var",1e-7);
	max_iter = args.get<size_t>("max_iter",10);
	size_t Nqmax = args.get<size_t>("Nqmax",6);
	size_t D = args.get<size_t>("D",2);

	bool CALC_SU2 = args.get<bool>("SU2",false);
	bool CALC_U1 = args.get<bool>("U1",true);
	bool CALC_U0 = args.get<bool>("U0",true);

	ISING = args.get<bool>("ISING",true);
	HEIS2 = args.get<bool>("HEIS2",false);
	HEIS3 = args.get<bool>("HEIS3",false);
	HUBB  = args.get<bool>("HUBB",false);
	SSH   = args.get<bool>("SSH",false);
	ALL   = args.get<bool>("ALL",false);
	if (ALL)
	{
		ISING = true;
		HEIS2 = true;
		HEIS3 = true;
		HUBB  = true;
		SSH   = true;
	}
	
	DMRG::VERBOSITY::OPTION VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",2));
	
	lout << args.info() << endl;
//	lout.set(make_string("L=",L,"_Nup=",Nup,"_Ndn=",Ndn,".log"),"log");
	
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
//	typedef VMPS::Heisenberg    HEISENBERG;
//	typedef VMPS::HeisenbergXXZ XXZ;
//	typedef VMPS::Hubbard       HUBBARD;
//	HEISENBERG::uSolver DMRG(VERB);
//	HUBBARD::uSolver DMRG_HUBB(VERB);
//	Eigenstate<Umps<Sym::U0,double> > g;
	
	typedef VMPS::HeisenbergSU2 HEISENBERG_SU2;
	HEISENBERG_SU2 Heis_SU2(L,{{"Ly",Ly},{"J",J},{"OPEN_BC",false},{"D",D}});

	HEISENBERG_SU2::uSolver DMRG_SU2(VERB);
	Eigenstate<HEISENBERG_SU2::StateUd> g_SU2;
	if (CALC_SU2)
	{
		lout << Heis_SU2.info() << endl;
		DMRG_SU2.set_log(2,"e_Heis_SU2.dat","err_eigval_Heis_SU2.dat","err_var_Heis_SU2.dat");
		DMRG_SU2.edgeState(Heis_SU2, g_SU2, tol_eigval,tol_var, M, Nqmax, max_iter,1);
	}
	
	typedef VMPS::HeisenbergU1 HEISENBERG_U1;
	HEISENBERG_U1 Heis_U1(L,{{"Ly",Ly},{"Jxy",Jxy},{"Jz",Jz},{"Bz",Bz},{"OPEN_BC",false},{"D",D}});

	HEISENBERG_U1::uSolver DMRG_U1(VERB);
	Eigenstate<HEISENBERG_U1::StateUd> g_U1;
	if(CALC_U1)
	{
		lout << Heis_U1.info() << endl;
		DMRG_U1.set_log(2,"e_Heis_U1.dat","err_eigval_Heis_U1.dat","err_var_Heis_U1.dat");
		DMRG_U1.edgeState(Heis_U1, g_U1, tol_eigval,tol_var, M, Nqmax, max_iter,1);
	}
	
	typedef VMPS::Heisenberg HEISENBERG0;
	HEISENBERG0 Heis0(L,{{"Ly",Ly},{"J",J},{"Bz",Bz},{"OPEN_BC",false},{"D",D}});
	
	HEISENBERG0::uSolver DMRG0(VERB);
	Eigenstate<HEISENBERG0::StateUd> g0;
	if (CALC_U1)
	{
		lout << Heis0.info() << endl;
		DMRG0.set_log(2,"e_Heis_U0.dat","err_eigval_Heis_U0.dat","err_var_Heis_U0.dat");
		DMRG0.edgeState(Heis0, g0, tol_eigval,tol_var, M, 1, max_iter,1);
		cout << g0.state.info() << endl;
	}
	
	cout << setprecision(13);
	if (CALC_SU2) {cout << "e0(SU2)=" << g_SU2.energy << endl;}
	if (CALC_U1) {cout << "e0(U1) =" << g_U1.energy << endl;}
	if (CALC_U0) {cout << "e0(U0) =" << g0.energy << endl;}
	if (D==2)
	{
		cout << "e(ref) =" << 0.25-log(2) << endl;
	}
	else if (D==3)
	{
		cout << "e(ref) =" << 1.401484038971 << endl;
	}
	
	if (CALC_U0)
	{
		cout << "-----U0-----" << endl;
		print_mag(Heis0,g0);
		size_t dmax = 10;
		for (size_t d=1; d<dmax; ++d)
		{
			HEISENBERG0 Htmp(d+1,{{"Ly",Ly},{"J",J},{"Bz",Bz},{"OPEN_BC",false},{"D",D}});
			double SvecSvec = Htmp.SvecSvecAvg(g0.state,0,d);
			lout << "d=" << d << ", <SvecSvec>=" << SvecSvec << endl;
		}
	}
	if (CALC_U1)
	{
		cout << endl;
		cout << "-----U1-----" << endl;
		print_mag(Heis_U1,g_U1);
		size_t dmax = 10;
		for (size_t d=1; d<dmax; ++d)
		{
			HEISENBERG_U1 Htmp(d+1,{{"Ly",Ly},{"Jxy",Jxy},{"Jz",Jz},{"Bz",Bz},{"OPEN_BC",false},{"D",D}});
			double SvecSvec = Htmp.SvecSvecAvg(g_U1.state,0,d);
			lout << "d=" << d << ", <SvecSvec>=" << SvecSvec << endl;
		}
		
		g_U1.state.graph("g");
	}

	if (CALC_SU2)
	{
		cout << endl;
		cout << "-----SU2-----" << endl;
		// print_mag(Heis,g);
		size_t dmax = 10;
		for (size_t d=1; d<dmax; ++d)
		{
			HEISENBERG_SU2 Htmp(d+1,{{"Ly",Ly},{"J",J},{"OPEN_BC",false},{"D",D}});
			double SvecSvec = avg(g_SU2.state,Htmp.SS(0,d),g_SU2.state);
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
//		DMRG.edgeState(Ising.H2site(0,true), Ising.locBasis(0), g, {}, tol_eigval,tol_var, M, max_iter,1);
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
//	//	DMRG.edgeState(Heis.H2site(0,true), Heis.locBasis(0), g, {}, tol_eigval,tol_var, M, max_iter,1);
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
//	//---<Hubbard>---
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

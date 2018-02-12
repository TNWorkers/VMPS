#define DONT_USE_LAPACK_SVD
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

#include "LanczosWrappers.h"
#include "StringStuff.h"
#include "Stopwatch.h"

#include "VUMPS/VumpsSolver.h"
#include "VUMPS/VumpsLinearAlgebra.h"
#include "solvers/DmrgSolver.h"
#include "models/Heisenberg.h"
#include "models/HeisenbergXXZ.h"
#include "models/Hubbard.h"

// integration files not included in git
#include "gsl/gsl_integration.h"
#include "LiebWu.h"

double Jz, Bx, Bz;
double U, mu;
double dt;
double e_exact;
size_t L;
size_t M, max_iter;
double tol_eigval, tol_var;

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
	if (Bx==0.)
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

int main (int argc, char* argv[]) // usage: -L (int) -Nup (int) -Ndn (int) -U (double) -V (double) -Dinit (int) -Dlimit (int) -D (double)
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",1);
	Jz = args.get<double>("Jz",1.);
	Bx = args.get<double>("Bx",1.);
	Bz = args.get<double>("Bz",0.);
//	eIsing::j = Jz;
//	eIsing::h = Bx;
	U = args.get<double>("U",10.);
	mu = args.get<double>("mu",0.5*U);
	
	dt = args.get<double>("dt",0.5); // hopping-offset beim SSH-Modell
	M = args.get<double>("M",10); // bond dimension
	tol_eigval = args.get<double>("tol_eigval",1e-6);
	tol_var = args.get<double>("tol_var",1e-7);
	max_iter = args.get<size_t>("max_iter",20);
	
	DMRG::VERBOSITY::OPTION VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",2));
	
	lout << args.info() << endl;
//	lout.set(make_string("L=",L,"_Nup=",Nup,"_Ndn=",Ndn,".log"),"log");
	
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	typedef VMPS::Heisenberg    HEIS;
	typedef VMPS::HeisenbergXXZ XXZ;
	typedef VMPS::Hubbard       HUBB;
	HEIS::uSolver DMRG(VERB);
	HUBB::uSolver DMRG_HUBB(VERB);
	Eigenstate<Umps<Sym::U0,double> > g;
	
	//---<transverse Ising>---
	
	XXZ Ising(L,{{"Jz",Jz},{"Bx",Bx},{"OPEN_BC",false}});
	DMRG.set_log(2,"e.dat","err_eigval.dat","err_var.dat");
	lout << Ising.info() << endl;
	
	DMRG.edgeState(Ising.H2site(0,0,true), Ising.locBasis(0), g, {}, tol_eigval,tol_var, M, max_iter,1);
//	DMRG.edgeState(Ising, g, {}, tol_eigval,tol_var, M, max_iter,1);
	
	e_exact = IsingGround(Jz,Bx); // integrate(IsingGroundIntegrand, 0.,0.5*M_PI, 1e-10,1e-10);
//	e_exact = -1.0635444099809814;
	lout << TCOLOR(BLUE);
	lout << "Transverse Ising: e0=" << g.energy << ", exact:" << e_exact << ", diff=" << abs(g.energy-e_exact) << endl;
	
	for (size_t l=0; l<L; ++l)
	{
		lout << "<Sz("<<l<<")>=" << avg(g.state, Ising.Sz(l), g.state) << endl;
		lout << "<Sx("<<l<<")>=" << avg(g.state, Ising.Sx(l), g.state) << endl;
	}
	lout << TCOLOR(BLACK) << endl;
	
//	//---<Heisenberg S=1/2>---
//	
//	HEIS Heis(L,{{"Bz",Bz},{"OPEN_BC",false}});
//	lout << Heis.info() << endl;
//	
////	DMRG.edgeState(Heis.H2site(0,0,true), Heis.locBasis(0), g, {}, tol_eigval,tol_var, M, max_iter,1);
//	DMRG.edgeState(Heis, g, {}, tol_eigval,tol_var, M, max_iter,1);
//	
//	e_exact = 0.25-log(2);
//	lout << TCOLOR(BLUE);
//	lout << "Heisenberg S=1/2: e0=" << g.energy << ", exact(Δ=0):" << e_exact << ", diff=" << abs(g.energy-e_exact) << endl;
//	print_mag(Heis,g);
//	for (size_t l=0; l<L; ++l)
//	{
//		lout << "l=" << l << ", entropy=" << g.state.entropy(l);
//	}
//	lout << TCOLOR(BLACK) << endl;
//	
//	lout << "spin-spin correlations at distance d:" << endl;
//	size_t dmax = 10;
//	for (size_t d=1; d<dmax; ++d)
//	{
//		HEIS Htmp(d+1,{{"Bz",Bz},{"Bx",Bx},{"OPEN_BC",false}});
//		double SvecSvec = Htmp.SvecSvecAvg(g.state,0,d);
//		lout << "d=" << d << ", <SvecSvec>=" << SvecSvec << endl;
//	}
//	lout << endl;
//	
//	//---<Heisenberg S=1>---
//	
//	Heis = HEIS(L,{{"Bz",Bz},{"OPEN_BC",false},{"D",3ul}});
//	lout << Heis.info() << endl;
//	
//	DMRG.edgeState(Heis.H2site(0,0,true), Heis.locBasis(0), g, {}, tol_eigval,tol_var, M, max_iter,1);
////	DMRG.edgeState(Heis, g, {}, tol_eigval,tol_var, M, max_iter,1);
//	
//	e_exact = -1.40148403897122; // value from: Haegeman et al. PRL 107, 070601 (2011)
//	lout << TCOLOR(BLUE);
//	lout << "Heisenberg S=1: e0=" << g.energy << ", quasiexact:" << e_exact << ", diff=" << abs(g.energy-e_exact) << endl;
//	print_mag(Heis,g);
//	for (size_t l=0; l<L; ++l)
//	{
//		lout << "l=" << l << ", entropy=" << g.state.entropy(l) << endl;
//	}
//	lout << TCOLOR(BLACK) << endl;
//	ofstream SchmidtFiler("Schmidt.dat");
//	if (L==1)
//	{
//		for (size_t i=0; i<M; ++i)
//		{
//			SchmidtFiler << i << "\t" << setprecision(16) << g.state.singularValues(0)(i) << endl;
//		}
//	}
//	SchmidtFiler.close();
//	
//	//---<Hubbard>---
//	
//	HUBB Hubb(L,{{"U",U},{"mu",mu},{"OPEN_BC",false}});
//	lout << Hubb.info() << endl;
//	
//	DMRG_HUBB.set_log(10,"e.dat","err_eigval.dat","err_var.dat");
////	DMRG_HUBB.edgeState(Hubb, g, {}, tol_eigval,tol_var, M, max_iter,1);
//	DMRG_HUBB.edgeState(Hubb.H2site(0,0,true), Hubb.locBasis(0), g, {}, tol_eigval,tol_var, M, max_iter,1);
//	
//	lout << "half-filling test for μ=U/2: <n>=" << avg(g.state, Hubb.n(UP,0), g.state) + avg(g.state, Hubb.n(DN,0), g.state) << endl;
//	e_exact = LiebWu_E0_L(U,0.01*tol_eigval)-mu;
////	e_exact = -0.2671549218961211-mu; // value from: Bethe ansatz code, U=10
//	lout << TCOLOR(BLUE);
//	lout << "Hubbard (half-filling): e0=" << g.energy << ", exact=" << e_exact << ", diff=" << abs(g.energy-e_exact) << endl;
//	print_mag(Hubb,g);
//	lout << TCOLOR(BLACK) << endl;
//	
////	ofstream oFiler("overlap.dat");
////	for (double U=5.; U<=15.; U=U+0.2)
////	{
////		HUBB Hubbl(L,U,mu,false,false);
////		HUBB::uSolver DMRGl(DMRG::VERBOSITY::SILENT);
////		Eigenstate<Umps<0,double> > g2;
////		DMRGl.edgeState(Hubbl.H2site(0,0,true), Hubb.locBasis(0), g2, {}, tol_eigval,tol_var, M, max_iter,1);
////		lout << DMRGl.info() << endl;
////		auto overlap = g.state.dot(g2.state);
////		
////		cout << "U=" << U << ", " << overlap << ", norm=" << abs(overlap) << endl;
////		oFiler << U << "\t" << abs(overlap) << endl;
////		
////		for (size_t l=0; l<L; ++l)
////		{
////			lout << "<Sz1("<<l<<")>=" << avg(g.state, Hubb.Sz(l), g.state) << endl;
////			lout << "<Sz2("<<l<<")>=" << avg(g2.state, Hubb.Sz(l), g2.state) << endl;
////		}
////		cout << endl;
////	}
//	
//	//---<SSH model to test unit cell>---
//	
////	HUBB Hubb(L,{1.+dt,1.-dt},U,mu,false,false);
//	Hubb = HUBB(max(L,2ul),{{"t",1.+dt,0},{"t",1.-dt,1},{"OPEN_BC",false}});
//	lout << Hubb.info() << endl;
//	
//	DMRG_HUBB.edgeState(Hubb, g, {}, tol_eigval,tol_var, M, max_iter,1);
//	
//	double n = avg(g.state, Hubb.n(UP,0), g.state)+
//	           avg(g.state, Hubb.n(DN,0), g.state);
//	lout << "<n>=" << n << endl;
//	lout << TCOLOR(BLUE);
//	eSSH::v = -(1.+dt);
//	eSSH::w = -(1.-dt);
//	e_exact = integrate(eSSH::f, -0.5*M_PI,+0.5*M_PI, 1e-10,1e-10);
//	
//	lout << "SSH e0=" << g.energy << ", exact=" << e_exact << endl;
//	lout << "diff=" << abs(g.energy-e_exact) << endl;
////	print_mag(Hubb,g);
//	lout << TCOLOR(BLACK) << endl;
//	
////	lout << "Schmidt values:" << endl << g.state.singularValues(1).head(10) << endl;
}

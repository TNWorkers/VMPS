#include <iostream>
#include <fstream>
#include <complex>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

#include "Logger.h"
Logger lout;
#include "ArgParser.h"

#include "LanczosWrappers.h"
#include "StringStuff.h"
#include "Stopwatch.h"
#include "VumpsSolver.h"
#include "DmrgSolverQ.h"
#include "iDmrgSolver.h"
#include "MpTransverseHeisenbergModel.h"
#include "MpGrandHubbardModel.h"

//// integration files not included in git
//#include "gsl_integration.h"
//#include "LiebWu.h"

//// Ising model integrations
//struct eIsing
//{
//	static double h;
//	static double j;
//	
//	static double f (double x, void*)
//	{
//		return -j*0.5*M_1_PI*sqrt(1.+h*h-2*h*cos(x));
//	}
//};
//double eIsing::h=0.;
//double eIsing::j=1.;

double e_exact;
size_t L;

//// SSH model integrations
//struct eSSH
//{
//	static double v;
//	static double w;
//	
//	static double f (double x, void*)
//	{
//		return -M_1_PI*sqrt(v*v+w*w+2*v*w*cos(2*x));
//	}
//};
//double eSSH::v = -1.;
//double eSSH::w = -1.;

template<typename Hamiltonian, typename Eigenstate>
void print_mag (const Hamiltonian &H, const Eigenstate &g)
{
	VectorXd SZcell(L);
//	VectorXd SXcell(L);
	lout << endl;
	lout << "magnetization within unit cell (i.e. staggered): " << endl;
	for (size_t l=0; l<L; ++l)
	{
		SZcell(l) = avg(g.state, H.Sz(l), g.state);
//		SXcell(l) = avg(g.state, H.Scomp(SX,l), g.state);
		lout << "<Sz("<<l<<")>=" << SZcell(l) << endl;
//		lout << "<Sx("<<l<<")>=" << SXcell(l) << endl;
	}
	lout << "total magnetization: " << endl;
	lout << "<Sz>=" << SZcell.sum() << endl;
//	lout << "<Sx>=" << SXcell.sum() << endl;
}

int main (int argc, char* argv[]) // usage: -L (int) -Nup (int) -Ndn (int) -U (double) -V (double) -Dinit (int) -Dlimit (int) -D (double)
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",1);
	double Bx = args.get<double>("Bx",1.);
//	eIsing::j = 1.;
//	eIsing::h = 0.5*Bx;
	double Delta = args.get<double>("Delta",-1.);
	double U = args.get<double>("U",10.);
	double mu = args.get<double>("mu",0.5*U);
	size_t M = args.get<double>("M",100); // bond dimension
	double err_eigval = args.get<double>("err_eigval",1e-7);
	double err_var = args.get<double>("err_var",1e-9);
	size_t max_iter = args.get<size_t>("max_iter",20);
	
	DMRG::VERBOSITY::OPTION VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",2));
	
	lout << args.info() << endl;
//	lout.set(make_string("L=",L,"_Nup=",Nup,"_Ndn=",Ndn,".log"),"log");
	
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	vector<double> Bzvec; Bzvec.assign(L,0);
	vector<double> Bxvec; Bxvec.assign(L,Bx);
	
	typedef VMPS::TransverseHeisenbergModel HEIS;
	typedef VMPS::GrandHubbardModel         HUBB;
	HEIS::uSolver DMRG(VERB);
	HUBB::uSolver DMRG_HUBB(VERB);
	Eigenstate<UmpsQ<0,double> > g;
	
	// transverse Ising
	
	HEIS Heis(L,0,+4.,Bzvec,Bxvec,2,false);
	lout << Heis.info() << endl;
	
//	DMRG.edgeState(Heis.H2site(0,0,true), Heis.locBasis(0), g, {}, err_eigval,err_var, M, max_iter,1);
	DMRG.edgeState(Heis, g, {}, err_eigval,err_var, M, max_iter,1);
	
//	e_exact = integrate(eIsing::f, -M_PI,M_PI, 0.01*err_eigval,0.01*err_eigval);
	e_exact = -1.0635444099809814;
	lout << TCOLOR(BLUE);
	lout << "Transverse Ising: e0=" << g.energy << ", exact:" << e_exact << ", diff=" << abs(g.energy-e_exact) << endl;
	for (size_t l=0; l<L; ++l)
	{
		lout << "<Sz("<<l<<")>=" << avg(g.state, Heis.Scomp(SZ,l), g.state) << endl;
		lout << "<Sx("<<l<<")>=" << avg(g.state, Heis.Scomp(SX,l), g.state) << endl;
	}
	lout << TCOLOR(BLACK) << endl;
	
	// Heisenberg S=1/2
	
	Bxvec.assign(L,0);
	Heis = HEIS(L,-1.,Delta,Bzvec,Bxvec,2,false);
	lout << Heis.info() << endl;
//	DMRG.edgeState(Heis.H2site(0,0,true), Heis.locBasis(0), g, {}, err_eigval,err_var, M, max_iter,1);
	DMRG.edgeState(Heis, g, {}, err_eigval,err_var, M, max_iter,1);
	e_exact = 0.25-log(2);
	lout << TCOLOR(BLUE);
	lout << "Heisenberg S=1/2: e0=" << g.energy << ", exact(Δ=0):" << e_exact << ", diff=" << abs(g.energy-e_exact) << endl;
	print_mag(Heis,g);
	lout << TCOLOR(BLACK) << endl;
	
	lout << "spin-spin correlations at distance d:" << endl;
	size_t dmax = 10;
	for (size_t d=1; d<dmax; ++d)
	{
		vector<double> Bzvectmp; Bzvectmp.assign(d+1,0);
		vector<double> Bxvectmp; Bxvectmp.assign(d+1,Bx);
		HEIS Htmp(d+1,-1.,-1.,Bzvectmp,Bxvectmp,2,false);
		double SzSz = avg(g.state, Htmp.SzSz(0,d), g.state);
		double SpSm = avg(g.state, Htmp.SaSa(0,SP,d,SM), g.state);
		lout << "d=" << d << ", <SvecSvec>=" << SzSz+SpSm << endl;
	}
	lout << endl;
	
	// Heisenberg S=1
	
	Heis = HEIS(L,-1.,-1.,Bzvec,Bxvec,3,false);
	lout << Heis.info() << endl;
//	DMRG.edgeState(Heis.H2site(0,0,true), Heis.locBasis(0), g, {}, err_eigval,err_var, M, max_iter,1);
	DMRG.edgeState(Heis, g, {}, err_eigval,err_var, M, max_iter,1);
	e_exact = -1.40148403897122; // value from: Haegeman et al. PRL 107, 070601 (2011)
	lout << TCOLOR(BLUE);
	lout << "Heisenberg S=1: e0=" << g.energy << ", quasiexact:" << e_exact << ", diff=" << abs(g.energy-e_exact) << endl;
	print_mag(Heis,g);
	lout << TCOLOR(BLACK) << endl;
	
	// Hubbard
	
	HUBB Hubb(L,U,mu,false,false);
	lout << Hubb.info() << endl;
	DMRG_HUBB.edgeState({Hubb.H2site(0,0,true)}, Hubb.locBasis(0), g, {}, err_eigval,err_var, M, max_iter,1);
//	DMRG_HUBB.edgeState(Hubb, g, {}, err_eigval,err_var, M, max_iter,1);
	lout << "half-filling test for μ=U/2: <n>=" << avg(g.state, Hubb.n(UP,0), g.state) + avg(g.state, Hubb.n(DN,0), g.state) << endl;
//	e_exact = LiebWu_E0_L(U,0.01*err_eigval)-mu;
	e_exact = -0.2671549218961211-mu; // value from: Bethe ansatz code, U=10
	lout << TCOLOR(BLUE);
	lout << "Hubbard (half-filling): e0=" << g.energy << ", exact:" << e_exact << ", diff=" << abs(g.energy-e_exact) << endl;
	print_mag(Hubb,g);
	lout << TCOLOR(BLACK) << endl;
	
	// SSH model to test unit cell
	
//	HUBB Hubb(L,{1.,0.5},U,mu,false,false);
//	lout << Hubb.info() << endl;
//	DMRG_HUBB.edgeState(Hubb, g, {}, err_eigval,err_var, M, max_iter,1);
//	double n = avg(g.state, Hubb.n(UP,0), g.state) + avg(g.state, Hubb.n(DN,0), g.state);
//	lout << "<n>=" << n << endl;
//	lout << TCOLOR(BLUE);
//	eSSH::v = -1.;
//	eSSH::w = -0.5;
//	e_exact = integrate(eSSH::f, -0.5*M_PI,+0.5*M_PI, 0.01*err_eigval,0.01*err_eigval);
//	
//	lout << "Hubbard (half-filling): e0=" << g.energy << endl;
//	lout << "Hubbard (half-filling): e0+mu*n=" << g.energy+mu*n << endl;
//	lout << "diff=" << abs(g.energy+mu*n-e_exact) << endl;
//	print_mag(Hubb,g);
//	lout << TCOLOR(BLACK) << endl;
}

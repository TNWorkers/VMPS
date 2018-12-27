#define LANCZOS_MAX_ITERATIONS 1e2

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
#include "models/HubbardU1xU1.h"
#include "models/Hubbard.h"
#include "models/HubbardSU2xSU2.h"
#include "models/HubbardSU2xU1.h"
#include "models/KondoSU2xU1.h"
#include "models/KondoU1xU1.h"
#include "models/KondoU0xSU2.h"

double Jxy, Jz, J, Jprime, tPrime, Bx, Bz;
double U, mu;
double dt;
double e_exact;
size_t L, Ly;
int N;
size_t Dinit, max_iter, min_iter;
double tol_eigval, tol_var, tol_state;
bool ISING, HEIS2, HEIS3, SSH, ALL;

int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",1);
	Ly = args.get<size_t>("Ly",1);
	Jxy = args.get<double>("Jxy",0.);
	Jz = args.get<double>("Jz",-1.);
	J = args.get<double>("J",0.);
	Jprime = args.get<double>("Jprime",0.);
	tPrime = args.get<double>("tPrime",0.);
	Bx = args.get<double>("Bx",0.25);
	Bz = args.get<double>("Bz",0.);
	U = args.get<double>("U",10.);
	mu = args.get<double>("mu",0.5*U);
	N = args.get<int>("N",L);
	
	size_t i0 = args.get<int>("i0",0);
	size_t j0 = args.get<int>("j0",0);
	
	dt = args.get<double>("dt",0.5); // hopping-offset for SSH model
	Dinit = args.get<double>("Dinit",5);    // bond dimension
	tol_eigval = args.get<double>("tol_eigval",1e-9);
	tol_var = args.get<double>("tol_var",1e-8);
	tol_state = args.get<double>("tol_state",1e-7);

	max_iter = args.get<size_t>("max_iter",300ul);
	min_iter = args.get<size_t>("min_iter",1ul);
	size_t Qinit = args.get<size_t>("Qinit",6);
	size_t D = args.get<size_t>("D",2);

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
	
//	//	-------- Test of the DMRES Solver:
//	MatrixXcd A(100,100);
//	A.setRandom();
////	A.triangularView<Upper>() = A.adjoint();
//	VectorXcd b(100);
//	b.setRandom();
//	VectorXcd x = A.colPivHouseholderQr().solve(b);
////	cout << x.head(5).transpose() << endl;
//	VectorXcd y(100);
//	GMResSolver<MatrixXcd,VectorXcd> Gimli(A,b,y);
//	cout << Gimli.info() << endl;
////	cout << y.head(5).transpose() << endl;
//	cout << "(x-y).norm()=" << (x-y).norm() << endl;
//	cout << "Eigen: " << (A*x-b).norm() << endl;
//	cout << "GMRes: " << (A*y-b).norm() << endl;
//	assert(1!=1);
	
	bool CALC_SU2 = args.get<bool>("SU2",false);
	bool CALC = args.get<bool>("U1",true);
	bool CALC_U0 = args.get<bool>("U0",true);
	bool CALC_HUBB = args.get<bool>("HUBB",false);
	bool CALC_KOND = args.get<bool>("KOND",false);
	bool CALC_DOT = args.get<bool>("DOT",false);
	ALL   = args.get<bool>("ALL",false);
	if (ALL)
	{
		CALC_SU2  = true;
		CALC   = true;
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
	
	VUMPS::CONTROL::GLOB GlobParams;
	GlobParams.tol_eigval = 1e-7;
	GlobParams.tol_var = 1e-5;
	GlobParams.tol_state = 1e-3;
	GlobParams.min_iterations = 24;
	GlobParams.max_iterations = 150;
	GlobParams.Dinit = 10;
	
	typedef VMPS::HeisenbergU1XXZ MODEL;
	MODEL Heis(L,{{"Jxy",Jxy},{"Jz",Jz},{"OPEN_BC",false}}); //{"Bx",Bx},
	
	MODEL::uSolver DMRG(VERB);
	Eigenstate<MODEL::StateUd> g;
	lout << Heis.info() << endl;
	DMRG.set_log(2,"e_Heis.dat","err_eigval_Heis.dat","err_var_Heis.dat","err_state_Heis.dat");
	DMRG.userSetGlobParam();
	DMRG.GlobParam = GlobParams;
//		DMRG.set_verbosity(DMRG::VERBOSITY::SILENT);
	DMRG.edgeState(Heis, g, {});
	
	int N = 51;
	ofstream Filer(make_string("SF_Sym=",MODEL::Symmetry::name(),"_L=",L,"_i0=",i0,"_j0=",j0,".dat"));
	
	ArrayXcd SFzz(N); SFzz=0;
	ArrayXcd SFxx(N); SFxx=0;
	
	ArrayXd Szavg(L);
	ArrayXd Sxavg(L);
	
	vector<Mpo<MODEL::Symmetry> > Sz(L); 
	vector<Mpo<MODEL::Symmetry> > Sx(L);
	
	for (size_t i=0; i<L; ++i)
	{
		Szavg(i) = avg(g.state, Heis.Sz(i), g.state);
		Sxavg(i) = avg(g.state, Heis.Sx(i), g.state);
		
		Sz[i] = Heis.Sz(i);
		Sx[i] = Heis.Sx(i);
		
		Sz[i].scale(1.,-Szavg(i));
		Sx[i].scale(1.,-Sxavg(i));
		
		cout << "i=" << i << endl;
		cout << "<Sz>=" << avg(g.state, Sz[i], g.state) << ", shifted by " << Szavg(i) << endl;
		cout << "<Sx>=" << avg(g.state, Sx[i], g.state) << ", shifted by " << Sxavg(i) << endl;
	}
	
//		for (size_t i=0; i<L; ++i)
//		for (size_t j=0; j<L; ++j)
//		{
//			SFzz += g.state.structure_factor(Sz[i],Sz[j]);
//			SFxx += g.state.structure_factor(Sx[i],Sx[j]);
//		}
	SFzz += g.state.structure_factor(Sz[i0],Sz[j0]);
//	SFxx += g.state.structure_factor(Sx[i0],Sx[j0]);
	
	for (int ik=0; ik<SFzz.rows(); ++ik)
	{
		Filer << ik*2.*M_PI/(SFzz.rows()-1) << "\t" 
//		      << SFzz(ik).real() << "\t" << SFzz(ik).imag() << "\t" 
//		      << SFxx(ik).real() << "\t" << SFxx(ik).imag() << endl;
		      << abs(SFzz(ik)) << endl;
//		      << abs(SFxx(ik)) << endl;
	}
	Filer.close();
	
	N = 100; // Ncell (number of unit cells), L=Lcell (size of unit cell)
	
	MODEL Htmp(L*N,{{"Jxy",Jxy},{"Jz",Jz},{"OPEN_BC",false}}); // ,{"Bx",Bx}
	ArrayXd SzSzR(N);
	ArrayXd SxSxR(N);
	#pragma omp parallel for
	for (size_t n=0; n<N; ++n)
	{
		size_t l = L*n;
		SzSzR(n) = avg(g.state, Htmp.SzSz(i0,j0+l), g.state) - Szavg(i0%L)*Szavg((j0+l)%L);
		SxSxR(n) = avg(g.state, Htmp.SxSx(i0,j0+l), g.state) - Sxavg(i0%L)*Sxavg((j0+l)%L);
	}
	
	ArrayXd SzSzL(N);
	ArrayXd SxSxL(N);
	#pragma omp parallel for
	for (size_t n=0; n<N; ++n)
	{
		size_t l = L*n;
		SzSzL(n) = avg(g.state, Htmp.SzSz(i0+l,j0), g.state) - Szavg((i0+l)%L)*Szavg(j0%L);
		SxSxL(n) = avg(g.state, Htmp.SxSx(i0+l,j0), g.state) - Sxavg((i0+l)%L)*Sxavg(j0%L);
	}
	
	ArrayXcd Szk(2*N);
	ArrayXcd Sxk(2*N);
	
	for (size_t ik=0; ik<2*N; ++ik)
	{
		Szk(ik) = 0;
		Sxk(ik) = 0;
		double k = ik * 2.*M_PI/(2*N);
		for (size_t n=0; n<N; ++n)
		{
			if (n!=0)
			{
				Szk(ik) += SzSzR(n) * exp(-1.i*k*static_cast<double>(n));
				Sxk(ik) += SxSxR(n) * exp(-1.i*k*static_cast<double>(n));
			}
		}
		for (size_t n=0; n<N; ++n)
		{
			if (n!=0)
			{
				Szk(ik) += SzSzL(n) * exp(+1.i*k*static_cast<double>(n));
				Sxk(ik) += SxSxL(n) * exp(+1.i*k*static_cast<double>(n));
			}
		}
	}
	
	ofstream FilerFT(make_string("SF_FT","_L=",L,"_i0=",i0,"_j0=",j0,".dat"));
	for (size_t i=0; i<2*N; i++)
	{
		double k = i * 2.*M_PI/(2*N);
//		FilerFT << k << "\t" << Szk(i).real() << "\t" << Szk(i).imag() << "\t" << Sxk(i).real() << "\t" << Sxk(i).imag() << endl;
		FilerFT << k << "\t" << abs(Szk(i)) << "\t" << abs(Sxk(i)) << endl;
	}
	FilerFT.close();
}

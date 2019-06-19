#ifdef BLAS
#include "util/LapackManager.h"
#pragma message("LapackManager")
#endif

//#define USE_HDF5_STORAGE

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
#include "models/SpinlessFermionsU1.h"
#include "models/SpinlessFermionsZ2.h"

template<typename Scalar>
string to_string_prec (Scalar x, bool COLOR=false, int n=14)
{
	ostringstream ss;
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

size_t L;
int N;
double t, tPrime, V, Vprime, Vph;

typedef Sym::U1<Sym::ChargeU1> Symmetry;
Eigenstate<VMPS::SpinlessFermionsU1::StateXd> g_U1;
//Eigenstate<VMPS::SpinlessFermionsZ2::StateXd> g_Z2;
Eigenstate<VectorXd> g_ED;

double fRand(double fMin, double fMax)
{
	double f = (double)rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}

int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",4);
	N = args.get<size_t>("N",L/2);
	t = args.get<double>("t",1.);
	tPrime = args.get<double>("tPrime",0.);
	V = args.get<double>("V",0.);
	Vprime = args.get<double>("Vprime",0.);
	Vph = args.get<double>("Vph",0.);
	
	DMRG::CONTROL::GLOB GlobParam;
	DMRG::CONTROL::DYN  DynParam;
	
	lout << endl << termcolor::red << "--------U(1)---------" << termcolor::reset << endl << endl;
	
	Stopwatch<> Watch_U1;
	
	ArrayXXd RandomHopping(L,L); RandomHopping.setZero();
	for (int i=0; i<L; ++i)
	for (int j=0; j<i; ++j)
	{
		RandomHopping(i,j) = fRand(0.,1.);
		RandomHopping(j,i) = RandomHopping(i,j);
	}
	
	cout << RandomHopping << endl << endl;
	
//	VMPS::SpinlessFermionsU1 H_U1(L,{{"t",t},{"tPrime",tPrime},{"V",V},{"Vprime",Vprime},{"Vph",Vph}});
	VMPS::SpinlessFermionsU1 H_U1(L,{{"tFull",RandomHopping}});
	lout << H_U1.info() << endl;
	
	VMPS::SpinlessFermionsU1::Solver DMRG_U1(DMRG::VERBOSITY::HALFSWEEPWISE);
	DMRG_U1.edgeState(H_U1, g_U1, {N}, LANCZOS::EDGE::GROUND);
	g_U1.state.graph("U1");
	
	MatrixXd rhoA(L,L);
	MatrixXd rhoB(L,L);
	#pragma omp parallel for collapse(2)
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		rhoA(i,j) = avg(g_U1.state, H_U1.cdagc(i,j), g_U1.state);
		rhoB(i,j) = avg(g_U1.state, H_U1.cdag(i), H_U1.c(j), g_U1.state);
	}
	
	lout << "(rhoA-rhoB).norm()=" << (rhoA-rhoB).norm() << endl;
	
	
	InteractionParams params;
	(tPrime!=0.) ? params.set_hoppings({-t,-tPrime}):params.set_hoppings({-t});
	
//	HoppingParticles H_ED(L, N, params, BC_DANGLING, BS_FULL, DIM1, FERMIONS);
	HoppingParticles H_ED(L, N, -RandomHopping.matrix().sparseView());
	H_ED.switch_Vnn(V);
	lout << H_ED.info() << endl;
	LanczosSolver<HoppingParticles,VectorXd,double> Lutz;
	Lutz.edgeState(H_ED, g_ED, LANCZOS::EDGE::GROUND);
	lout << Lutz.info() << endl;
	
	lout << endl;
	lout << "E_ED=" << g_ED.energy << ", diff=" << abs(g_ED.energy-g_U1.energy) << endl;
	lout << endl;
	
//	cout << H_ED.eigenvalues() << endl;
	
	MatrixXd rhoED_A(L,L);
	MatrixXd rhoED_B(L,L);
	#pragma omp parallel for collapse(2)
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		if (i == j)
		{
			rhoED_A(i,j) = g_ED.state.dot(H_ED.n(i) * g_ED.state);
			rhoED_B(i,j) = H_ED.eigenvectors().col(0).dot(H_ED.n(i) * H_ED.eigenvectors().col(0));
		}
		else
		{
			rhoED_A(i,j) = g_ED.state.dot(H_ED.hopping_element(i,j,1.,false) * g_ED.state);
			rhoED_B(i,j) = H_ED.eigenvectors().col(0).dot(H_ED.hopping_element(i,j,1.,false) * H_ED.eigenvectors().col(0));
		}
	}
	
	cout << rhoA << endl << endl;
	cout << rhoB << endl << endl;
	cout << rhoED_A << endl << endl;
	cout << rhoED_B << endl << endl;
	
	lout << endl;
	lout << "(rhoA-rhoB).norm()=" << (rhoA-rhoB).norm() << endl;
	lout << "(rhoED_A-rhoED_B).norm()=" << (rhoA-rhoB).norm() << endl;
	lout << "(rhoA-rhoED_Lanczos).norm()=" << (rhoA-rhoED_A).norm() << endl;
	lout << "(rhoA-rhoED_fullDiag).norm()=" << (rhoB-rhoED_B).norm() << endl;
	lout << endl;
	
	lout << endl;
	
	SelfAdjointEigenSolver<MatrixXd> Eugen(-RandomHopping.matrix());
	double E0_exact = Eugen.eigenvalues().head(N).sum(); // fill up the Fermi sea
	cout << "1-particle energies: " << Eugen.eigenvalues().transpose() << endl;
//	cout << "eigenvalues: " << H_ED.eigenvalues().transpose() << endl;
	cout << "E0 exact for V=0: " << E0_exact 
	     << ", diffDMRG=" << abs(E0_exact-g_U1.energy) 
	     << ", diffED(Lanczos)=" << abs(E0_exact-g_ED.energy) 
	     << ", diffED(full)=" << abs(E0_exact-H_ED.eigenvalues()(0)) 
	     << endl;
	
	
	MatrixXd rhoED_free(L,L); rhoED_free.setZero();
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	for (int k=0; k<N; ++k)
	{
		rhoED_free(i,j) += Eugen.eigenvectors().col(k)(i) * Eugen.eigenvectors().col(k)(j);
	}
	
	cout << endl << rhoED_free << endl << endl;
	
//	VMPS::SpinlessFermionsZ2 H_Z2(L,{{"t",t},{"tPrime",tPrime},{"V",V},{"Vprime",Vprime},{"Vph",Vph},{"Delta",-t}});
//	lout << H_Z2.info() << endl;
//	VMPS::SpinlessFermionsZ2::Solver DMRG_Z2(DMRG::VERBOSITY::HALFSWEEPWISE);
//	DMRG_Z2.edgeState(H_Z2, g_Z2, {0}, LANCZOS::EDGE::GROUND);
	
}

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
	Jxy = args.get<double>("Jxy",0.1);
	Jz = args.get<double>("Jz",1.);
	J = args.get<double>("J",1.);
	Jprime = args.get<double>("Jprime",0.);
	tPrime = args.get<double>("tPrime",0.);
	Bx = args.get<double>("Bx",0.25);
	Bz = args.get<double>("Bz",0.);
	U = args.get<double>("U",10.);
	mu = args.get<double>("mu",0.5*U);
	N = args.get<int>("N",L);
	int Nk = args.get<int>("Nk",31); // amount of k-points
	
	size_t i0 = (args.get<int>("i0",0))%L;
	size_t j0 = (args.get<int>("j0",0))%L;
	
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
	
	bool SU2 = args.get<bool>("SU2",true);
	bool U1  = args.get<bool>("U1",false);
	bool U0  = args.get<bool>("U0",false);
	if (ALL)
	{
		SU2 = true;
		U1  = true;
		U0  = true;
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
	GlobParams.tol_eigval = 1e-5;
	GlobParams.tol_var = 1e-5;
	GlobParams.tol_state = 1e-2;
	GlobParams.min_iterations = 10;
	GlobParams.max_iterations = 120;
	GlobParams.Dinit = 10;
	
	if (U1)
	{
		typedef VMPS::HeisenbergU1XXZ MODEL;
		MODEL H(L,{{"Jxy",Jxy},{"Jz",Jz},{"OPEN_BC",false}});
		qarray<1> Qc = {0};
//		typedef VMPS::HeisenbergXXZ MODEL;
//		MODEL H(L,{{"Bx",Bx},{"Jz",Jz},{"OPEN_BC",false}});
		lout << H.info() << endl;
		
		MODEL::uSolver DMRG(VERB);
		Eigenstate<MODEL::StateUd> g;
		DMRG.userSetGlobParam();
		DMRG.GlobParam = GlobParams;
		DMRG.edgeState(H, g, Qc);
		
		ArrayXcd SF(Nk); SF=0;
		
		vector<Mpo<MODEL::Symmetry> > O(L); 
		vector<Mpo<MODEL::Symmetry> > Odag(L);
		
		ArrayXd Oavg(L);
		ArrayXd Odagavg(L);
		
		for (size_t l=0; l<L; ++l)
		{
			O[l]    = H.Scomp(SM,l);
			Odag[l] = H.Scomp(SP,l);
		}
		
		for (size_t l=0; l<L; ++l)
		{
			Oavg(l)    = avg(g.state, O[l],    g.state);
			Odagavg(l) = avg(g.state, Odag[l], g.state);
			
			O[l].scale(1.,-Oavg(l));
			Odag[l].scale(1.,-Odagavg(l));
			
			O[l].transform_base(Qc);
			Odag[l].transform_base(Qc);
			
			cout << "l=" << l << endl;
			cout << "<O>="    << avg(g.state, O[l],    g.state) << ", shifted by " << Oavg(l)    << endl;
			cout << "<Odag>=" << avg(g.state, Odag[l], g.state) << ", shifted by " << Odagavg(l) << endl;
		}
		
		SF += g.state.structure_factor(Odag[i0], O[j0], 0., 2.*M_PI, Nk, DMRG::VERBOSITY::STEPWISE);
		
		ofstream Filer(make_string("SF_Sym=",MODEL::Symmetry::name(),"_L=",L,"_i0=",i0,"_j0=",j0,".dat"));
		for (int ik=0; ik<SF.rows(); ++ik)
		{
			Filer << ik*2.*M_PI/(SF.rows()-1) << "\t" << abs(SF(ik)) << endl;
		}
		Filer.close();
		lout << make_string("SF_Sym=",MODEL::Symmetry::name(),"_L=",L,"_i0=",i0,"_j0=",j0,".dat") << " saved!" << endl;
		
		N = 100; // Ncell (number of unit cells), L=Lcell (size of unit cell)
		
		MODEL Htmp(L*N,{{"Jxy",Jxy},{"Jz",Jz},{"OPEN_BC",false}}); // ,{"Bx",Bx}
		ArrayXd OdagO_R(N); OdagO_R=0;
		#pragma omp parallel for
		for (size_t n=0; n<N; ++n)
		{
			size_t l = L*n;
			OdagO_R(n) = avg(g.state, Htmp.SmSp(i0,j0+l), g.state) - Odagavg(i0%L)*Oavg((j0+l)%L);
		}
		
		ArrayXd OdagO_L(N); OdagO_L=0;
		#pragma omp parallel for
		for (size_t n=1; n<N; ++n)
		{
			size_t l = L*n;
			OdagO_L(n) = avg(g.state, Htmp.SmSp(i0+l,j0), g.state) - Odagavg((i0+l)%L)*Oavg(j0%L);
		}
		
		ArrayXcd Sk(2*N);
		
		for (size_t ik=0; ik<2*N; ++ik)
		{
			Sk(ik) = 0;
			double k = ik * 2.*M_PI/(2*N);
			for (size_t n=1; n<N; ++n)
			{
				Sk(ik) += OdagO_R(n) * exp(-1.i*k*static_cast<double>(n));
				Sk(ik) += OdagO_L(n) * exp(+1.i*k*static_cast<double>(n));
			}
		}
		
		ofstream FilerFT(make_string("FT_Sym=",MODEL::Symmetry::name(),"_L=",L,"_i0=",i0,"_j0=",j0,".dat"));
		for (size_t i=0; i<2*N; i++)
		{
			double k = i * 2.*M_PI/(2*N);
			FilerFT << k << "\t" << abs(Sk(i)) << endl;
		}
		FilerFT.close();
		
		lout << make_string("FT_Sym=",MODEL::Symmetry::name(),"_L=",L,"_i0=",i0,"_j0=",j0,".dat") << " saved!" << endl;
	}
	
	if (SU2)
	{
//		typedef VMPS::HubbardSU2xU1 MODEL;
//		MODEL H(L,{{"U",20.},{"OPEN_BC",false},{"CALC_SQUARE",false}});
//		qarray<2> Qc = {1,N};
		typedef VMPS::HeisenbergSU2 MODEL;
		MODEL H(L,{{"J",J},{"OPEN_BC",false},{"CALC_SQUARE",false}});
		qarray<1> Qc = {1};
		
		H.transform_base(Qc);
		lout << H.info() << endl;
		
		MODEL::uSolver DMRG(VERB);
		Eigenstate<MODEL::StateUd> g;
		GlobParams.Qinit = 6;
		DMRG.userSetGlobParam();
		DMRG.GlobParam = GlobParams;
		DMRG.edgeState(H, g, Qc);
		
		ArrayXcd SF(Nk); SF=0;
		
		vector<Mpo<MODEL::Symmetry> > O(L); 
		vector<Mpo<MODEL::Symmetry> > Odag(L);
		
		ArrayXd Oavg(L);
		ArrayXd Odagavg(L);
		
		for (size_t l=0; l<L; ++l)
		{
			O[l]    = H.S(l);
			Odag[l] = H.Sdag(l);
			
//			O[l]    = H.n(l);
//			Odag[l] = H.n(l);
		}
		
		for (size_t l=0; l<L; ++l)
		{
			Oavg(l)    = avg(g.state, O[l], g.state);
			Odagavg(l) = avg(g.state, Odag[l], g.state);
			
			O[l].scale(1.,-Oavg(l));
			Odag[l].scale(1.,-Odagavg(l));
			
			O[l].transform_base(Qc);
			Odag[l].transform_base(Qc);
			
			cout << "l=" << l << endl;
			cout << "<O>="    << avg(g.state, O[l],    g.state) << ", shifted by " << Oavg(l)    << endl;
			cout << "<Odag>=" << avg(g.state, Odag[l], g.state) << ", shifted by " << Odagavg(l) << endl;
		}
		
//		vector<vector<qarray<MODEL::Symmetry::Nq> > > fullBasis(L);
//		
//		for (size_t l=0; l<L; ++l)
//		{
////			lout << "O[0].opBasis(l).size()=" << O[0].opBasis(l).size() << endl;
//			for (size_t r=0; r<L; ++r)
//			for (size_t i=0; i<O[0].opBasis(l).size(); ++i)
//			{
////				cout << "l=" << l << ", i=" << i << ", O[0].opBasis(l)=" << O[0].opBasis(l)[i] << endl;
//				fullBasis[l].push_back(O[l].opBasis(r)[i]);
//			}
//		}
//		
//		for (size_t l=0; l<L; ++l)
//		{
//			for (size_t r=0; r<L; ++r)
//			{
//				O[l].setOpBasis(fullBasis[l],r);
//				Odag[l].setOpBasis(fullBasis[l],r);
//				cout << "l=" << l << ", r=" << r << ", O[l].opBasis(r).size()=" << O[l].opBasis(r).size() << endl;
//			}
//		}
		
		SF += g.state.structure_factor(Odag[i0], O[j0], 0., 2.*M_PI, Nk, DMRG::VERBOSITY::STEPWISE);
		
		ofstream Filer(make_string("SF_Sym=",MODEL::Symmetry::name(),"_L=",L,"_i0=",i0,"_j0=",j0,".dat"));
		for (int ik=0; ik<SF.rows(); ++ik)
		{
			Filer << ik*2.*M_PI/(SF.rows()-1) << "\t" 
			      << abs(SF(ik)) << endl;
		}
		Filer.close();
		lout << make_string("SF_Sym=",MODEL::Symmetry::name(),"_L=",L,"_i0=",i0,"_j0=",j0,".dat") << " saved!" << endl;
		
		N = 100; // Ncell (number of unit cells), L=Lcell (size of unit cell)
		
		MODEL Htmp(L*N,{{"J",J},{"OPEN_BC",false},{"CALC_SQUARE",false}});
		ArrayXd OdagO_R(N); OdagO_R=0;
		ArrayXd OdagO_L(N); OdagO_L=0;
		#pragma omp parallel for
		for (size_t n=1; n<N; ++n)
		{
			size_t l = L*n;
			
//			OdagO_R(n) = avg(g.state, Htmp.nn(i0,j0+l), g.state) - Odagavg(i0%L)*Oavg((j0+l)%L);
			OdagO_R(n) = avg(g.state, Htmp.SdagS(i0,j0+l), g.state) - Odagavg(i0%L)*Oavg((j0+l)%L);
			
//			OdagO_L(n) = avg(g.state, Htmp.nn(i0+l,j0), g.state) - Odagavg((i0+l)%L)*Oavg(j0%L);
			OdagO_L(n) = avg(g.state, Htmp.SdagS(i0+l,j0), g.state) - Odagavg((i0+l)%L)*Oavg(j0%L);
		}
		
		ArrayXcd Ok(2*N);
		
		for (size_t ik=0; ik<2*N; ++ik)
		{
			Ok(ik) = 0;
			double k = ik * 2.*M_PI/(2*N);
			for (size_t n=1; n<N; ++n)
			{
				Ok(ik) += OdagO_R(n) * exp(-1.i*k*static_cast<double>(n));
				Ok(ik) += OdagO_L(n) * exp(+1.i*k*static_cast<double>(n));
			}
		}
		
		ofstream FilerFT(make_string("FT_Sym=",MODEL::Symmetry::name(),"_L=",L,"_i0=",i0,"_j0=",j0,".dat"));
		for (size_t i=0; i<2*N; i++)
		{
			double k = i * 2.*M_PI/(2*N);
			FilerFT << k << "\t" << abs(Ok(i)) << endl;
		}
		FilerFT.close();
		lout << make_string("FT_Sym=",MODEL::Symmetry::name(),"_L=",L,"_i0=",i0,"_j0=",j0,".dat") << " saved!" << endl;
	}
}

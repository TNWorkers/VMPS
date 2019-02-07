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
size_t D;

//ArrayXcd calc_FT_SU2 (size_t i0, size_t j0, const Eigenstate<VMPS::HeisenbergSU2::StateUd> &g, const ArrayXd &Oavg, const ArrayXd &Odagavg)
//{
//	VMPS::HeisenbergSU2 Htmp(2*N*L+4,{{"J",J},{"OPEN_BC",false},{"CALC_SQUARE",false},{"Ly",Ly},{"D",D}});
//	ArrayXd OdagO_R(N); OdagO_R=0;
//	ArrayXd OdagO_L(N); OdagO_L=0;
//	
//	#pragma omp parallel for
//	for (size_t n=1; n<N; ++n)
//	{
//		size_t l = L*n;
//		
////		OdagO_R(n) = avg(g.state, Htmp.nn(i0,j0+l), g.state) - Odagavg(i0%L)*Oavg((j0+l)%L);
//		OdagO_R(n) = avg(g.state, Htmp.SdagS(i0,j0+l), g.state) - Odagavg(i0%L)*Oavg((j0+l)%L);
//		
////		OdagO_L(n) = avg(g.state, Htmp.nn(i0+l,j0), g.state) - Odagavg((i0+l)%L)*Oavg(j0%L);
//		OdagO_L(n) = avg(g.state, Htmp.SdagS(i0+l,j0), g.state) - Odagavg((j0+l)%L)*Oavg(i0%L);
//		
////		cout << "calc_FT_SU2: (" << i0 << ", " << j0+l << "), (" << i0+l << "," << j0 << ")\t" << OdagO_R(n) << "\t" << OdagO_L(n) << endl;
//	}
//	
//	ArrayXcd Ok(N);
//	
////	cout << "plain sum=" << OdagO_L.sum() + OdagO_R.sum() << endl;
//	
//	for (size_t ik=0; ik<N; ++ik)
//	{
//		Ok(ik) = 0;
//		double k = ik * 2.*M_PI/N;
//		for (size_t n=1; n<N; ++n)
//		{
//			Ok(ik) += OdagO_R(n) * exp(-1.i*k*static_cast<double>(n));
//			Ok(ik) += OdagO_L(n) * exp(+2.*1.i*k*static_cast<double>(n));
//		}
//		
////		if (abs(k) < 1e-14)
////		{
////			cout << "CELLS" << i0 << j0 << " k=0: " << Ok(ik) << endl;
////		}
////		else if (abs(k-M_PI) < 1e-14)
////		{
////			cout << "CELLS" << i0 << j0 << " k=pi: " << Ok(ik) << endl;
////		}
//	}
//	
//	return Ok;
//}

//ArrayXcd calc_FT_SU2_full (const Eigenstate<VMPS::HeisenbergSU2::StateUd> &g, const ArrayXd &Oavg, const ArrayXd &Odagavg)
//{
//	VMPS::HeisenbergSU2 Htmp(2*N*L+4,{{"J",J},{"OPEN_BC",false},{"CALC_SQUARE",false},{"Ly",Ly},{"D",D}});
//	ArrayXd OdagO_R(L*N); OdagO_R=0;
//	ArrayXd OdagO_L(L*N); OdagO_L=0;
//	
//	#pragma omp parallel for
//	for (size_t l=0; l<N*L; ++l)
//	{
//		if (l!=0 and l!=1)
//		{
//			OdagO_R(l) = avg(g.state, Htmp.SdagS(0,l), g.state) - Odagavg(0)*Oavg(l%L);
////			OdagO_L(l) = avg(g.state, Htmp.SdagS(l,0), g.state) - Odagavg(l%L)*Oavg(0);
////			cout << "calc_FT_SU2_full: " << "(0" << ", " << l << "), (" << l << ",0)\t" << OdagO_R(l) << "\t" << OdagO_L(l) << endl;
//		}
//	}
//	
//	ArrayXcd Ok(N*L);
//	
////	cout << "plain sum=" << OdagO_L.sum() + OdagO_R.sum() << endl;
//	
//	for (size_t ik=0; ik<N*L; ++ik)
//	{
//		Ok(ik) = 0;
//		double k = ik * 2.*M_PI/(N*L);
//		for (size_t l=0; l<N*L; ++l)
//		{
//			Ok(ik) += OdagO_R(l) * exp(-1.i*k*static_cast<double>(l));
////			Ok(ik) += OdagO_L(l) * exp(+1.i*k*static_cast<double>(l));
//		}
//		
//		if (abs(k) < 1e-14)
//		{
//			cout << "FULL k=0: " << Ok(ik) << endl;
//		}
//		else if (abs(k-M_PI) < 1e-14)
//		{
//			cout << "FULL k=pi: " << Ok(ik) << endl;
//		}
//	}
//	
//	return Ok;
//}

vector<vector<ArrayXd> > OdagO;

void fill_OdagO_SU2 (const Eigenstate<VMPS::HeisenbergSU2::StateUd> &g, const ArrayXd &Oavg, const ArrayXd &Odagavg)
{
	VMPS::HeisenbergSU2 Htmp(2*N*L+4,{{"J",J},{"OPEN_BC",false},{"CALC_SQUARE",false},{"Ly",Ly},{"D",D}});
	
	OdagO.resize(L);
	for (size_t i0=0; i0<L; ++i0)
	{
		OdagO[i0].resize(L);
		for (size_t j0=0; j0<L; ++j0)
		{
			OdagO[i0][j0].resize(N);
		}
	}
	
	#pragma omp parallel for collapse(3)
	for (size_t i0=0; i0<L; ++i0)
	for (size_t j0=0; j0<L; ++j0)
	for (size_t n=0; n<N; ++n)
	{
		OdagO[i0][j0](n) = avg(g.state, Htmp.SdagS(i0,j0+L*n), g.state) - Odagavg(i0%L)*Oavg((j0+L*n)%L);
	}
}

ArrayXXcd calc_FT (double k, const Eigenstate<VMPS::HeisenbergSU2::StateUd> &g, const ArrayXd &Oavg, const ArrayXd &Odagavg)
{
	ArrayXXcd res(L,L); res=0;
	
	for (size_t i0=0; i0<L; ++i0)
	for (size_t j0=0; j0<L; ++j0)
	{
		res(i0,j0) = OdagO[i0][j0](0);
		
		for (size_t n=1; n<N; ++n)
		{
			res(i0,j0) += OdagO[i0][j0](n) * exp(-1.i*k*static_cast<double>(L*n)) + OdagO[j0][i0](n) * exp(+1.i*k*static_cast<double>(L*n));
		}
	}
	
	return res;
}

complex<double> calc_FT_full (double k, const Eigenstate<VMPS::HeisenbergSU2::StateUd> &g, const ArrayXd &Oavg, const ArrayXd &Odagavg)
{
	complex<double> res = 0;
	
	for (size_t i0=0; i0<L; ++i0)
	for (size_t j0=0; j0<L; ++j0)
	{
		double i0d = i0;
		double j0d = j0;
		res += exp(-1.i*k*(i0d-j0d)) * calc_FT(k,g,Oavg,Odagavg)(i0,j0);
	}
	
	return res;
}

complex<double> calc_FT_full (double k, const ArrayXXcd &Sijk)
{
	complex<double> res = 0;
	
	for (size_t i0=0; i0<L; ++i0)
	for (size_t j0=0; j0<L; ++j0)
	{
		double i0d = i0;
		double j0d = j0;
		res += exp(-1.i*k*(i0d-j0d)) * Sijk(i0,j0);
	}
	
	return res;
}

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
	tol_eigval = args.get<double>("tol_eigval",1e-5);
	tol_var = args.get<double>("tol_var",1e-4);
	tol_state = args.get<double>("tol_state",1.);
	
	max_iter = args.get<size_t>("max_iter",300ul);
	min_iter = args.get<size_t>("min_iter",1ul);
	size_t Qinit = args.get<size_t>("Qinit",6);
	D = args.get<size_t>("D",3);

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
	GlobParams.tol_eigval = tol_eigval;
	GlobParams.tol_var = tol_var;
	GlobParams.tol_state = tol_state;
	GlobParams.min_iterations = args.get<size_t>("min_iter",10);
	GlobParams.max_iterations = args.get<size_t>("max_iter",100);
	GlobParams.Dinit = args.get<size_t>("Dinit",10);

	VUMPS::CONTROL::LANCZOS LanczosParams;
	LanczosParams.eps_eigval = 1.e-12;
	LanczosParams.eps_coeff = 1.e-12;

//	if (U1)
//	{
//		typedef VMPS::HeisenbergU1XXZ MODEL;
//		MODEL H(L,{{"Jxy",Jxy},{"Jz",Jz},{"OPEN_BC",false},{"D",D},{"Ly",Ly}});
//		qarray<1> Qc = {0};
////		typedef VMPS::HeisenbergXXZ MODEL;
////		MODEL H(L,{{"Bx",Bx},{"Jz",Jz},{"OPEN_BC",false}});
//		lout << H.info() << endl;
//		
//		MODEL::uSolver DMRG(VERB);
//		Eigenstate<MODEL::StateUd> g;
//		DMRG.userSetGlobParam();
//		DMRG.GlobParam = GlobParams;
//		DMRG.edgeState(H, g, Qc);
//		
//		ArrayXcd SF(Nk); SF=0;
//		
//		vector<Mpo<MODEL::Symmetry> > O(L); 
//		vector<Mpo<MODEL::Symmetry> > Odag(L);
//		
//		ArrayXd Oavg(L);
//		ArrayXd Odagavg(L);
//		
//		for (size_t l=0; l<L; ++l)
//		{
//			O[l]    = H.Scomp(SM,l);
//			Odag[l] = H.Scomp(SP,l);
////			O[l]    = H.Scomp(SZ,l);
////			Odag[l] = H.Scomp(SZ,l);
//		}
//		
//		for (size_t l=0; l<L; ++l)
//		{
//			Oavg(l)    = avg(g.state, O[l],    g.state);
//			Odagavg(l) = avg(g.state, Odag[l], g.state);
//			
//			O[l].scale(1.,-Oavg(l));
//			Odag[l].scale(1.,-Odagavg(l));
//			
//			O[l].transform_base(Qc);
//			Odag[l].transform_base(Qc);
//			
//			cout << "l=" << l << endl;
//			cout << "<O>="    << avg(g.state, O[l],    g.state) << ", shifted by " << Oavg(l)    << endl;
//			cout << "<Odag>=" << avg(g.state, Odag[l], g.state) << ", shifted by " << Odagavg(l) << endl;
//		}
//		
//		SF += g.state.structure_factor(Odag[i0], O[j0], 0., 2.*M_PI, Nk, DMRG::VERBOSITY::STEPWISE);
//		
//		ofstream Filer(make_string("SF_Sym=",MODEL::Symmetry::name(),"_L=",L,"_i0=",i0,"_j0=",j0,".dat"));
//		for (int ik=0; ik<SF.rows(); ++ik)
//		{
//			Filer << ik*2.*M_PI/(SF.rows()-1) << "\t" << SF(ik).real() << "\t" << SF(ik).imag() << endl;
//		}
//		Filer.close();
//		lout << make_string("SF_Sym=",MODEL::Symmetry::name(),"_L=",L,"_i0=",i0,"_j0=",j0,".dat") << " saved!" << endl;
//		
//		N = 100; // Ncell (number of unit cells), L=Lcell (size of unit cell)
//		
//		double a00 = avg(g.state, H.SmSp(0,0), g.state) - Odagavg(0)*Oavg(1);
//		double a01 = avg(g.state, H.SmSp(0,1), g.state) - Odagavg(0)*Oavg(1);
//		cout << a00 << "\t" << a01 << endl;
////		cout << avg(g.state, H.SmSp(0,1), g.state) << endl;
////		cout << avg(g.state, H.SmSp(1,0), g.state) << endl;
//		
//		cout << "k=0" << endl;
//		double c00 = g.state.structure_factor_point(Odag[0], O[0], 0., DMRG::VERBOSITY::SILENT).real();
//		double c01 = g.state.structure_factor_point(Odag[0], O[1], 0., DMRG::VERBOSITY::SILENT).real();
//		cout << c00 << "\t" << c01 << endl;
//		cout << "sum=" << 2.*(c00+c01+a00+a01) << endl;
//		
//		cout << "k=pi" << endl;
//		double d00 = g.state.structure_factor_point(Odag[0], O[0], M_PI, DMRG::VERBOSITY::SILENT).real();
//		double d01 = g.state.structure_factor_point(Odag[0], O[1], M_PI, DMRG::VERBOSITY::SILENT).real();
//		cout << d00 << "\t" << d01 << endl;
//		cout << "staggered sum=" << 2.*(d00-d01+a00-a01) << endl;
//		
//		MODEL Htmp(L*N+4,{{"Jxy",Jxy},{"Jz",Jz},{"OPEN_BC",false},{"D",D},{"Ly",Ly}}); // ,{"Bx",Bx}
//		ArrayXd OdagO_R(N); OdagO_R=0;
//		#pragma omp parallel for
//		for (size_t n=0; n<N; ++n)
//		{
//			size_t l = L*n;
//			OdagO_R(n) = avg(g.state, Htmp.SmSp(i0,j0+l), g.state) - Odagavg(i0%L)*Oavg((j0+l)%L);
//		}
//		
//		ArrayXd OdagO_L(N); OdagO_L=0;
//		#pragma omp parallel for
//		for (size_t n=1; n<N; ++n)
//		{
//			size_t l = L*n;
//			OdagO_L(n) = avg(g.state, Htmp.SmSp(i0+l,j0), g.state) - Odagavg((i0+l)%L)*Oavg(j0%L);
//		}
//		
//		ArrayXcd Sk(2*N);
//		
//		for (size_t ik=0; ik<2*N; ++ik)
//		{
//			Sk(ik) = 0;
//			double k = ik * 2.*M_PI/(2*N);
//			for (size_t n=1; n<N; ++n)
//			{
//				Sk(ik) += OdagO_R(n) * exp(-1.i*k*static_cast<double>(n));
//				Sk(ik) += OdagO_L(n) * exp(+1.i*k*static_cast<double>(n));
//			}
//		}
//		
//		ofstream FilerFT(make_string("FT_Sym=",MODEL::Symmetry::name(),"_L=",L,"_i0=",i0,"_j0=",j0,".dat"));
//		for (size_t i=0; i<2*N; i++)
//		{
//			double k = i * 2.*M_PI/(2*N);
//			FilerFT << k << "\t" << Sk(i).real() << "\t" << Sk(i).imag() << endl;
//		}
//		FilerFT.close();
//		
//		lout << make_string("FT_Sym=",MODEL::Symmetry::name(),"_L=",L,"_i0=",i0,"_j0=",j0,".dat") << " saved!" << endl;
//		
////		ArrayXd OdagO_R(L*N); OdagO_R=0;
////		ArrayXd OdagO_L(L*N); OdagO_L=0;
////		#pragma omp parallel for
////		for (size_t l=0; l<N*L; ++l)
////		{
////			OdagO_R(l) = avg(g.state, Htmp.SmSp(0,l), g.state) - Odagavg(0)*Oavg(l%L);
////			OdagO_L(l) = avg(g.state, Htmp.SmSp(l,0), g.state) - Odagavg(l%L)*Oavg(0);
//////			cout << "l=" << l << "\t" << OdagO_R(l) << "\t" << OdagO_L(l) << endl;
////		}
////		
////		ArrayXcd Ok(N*L);
////		
////		for (size_t ik=0; ik<N*L; ++ik)
////		{
////			Ok(ik) = 0;
////			double k = ik * 2.*M_PI/(N*L);
////			for (size_t l=0; l<N*L; ++l)
////			{
////				Ok(ik) += OdagO_R(l) * exp(-1.i*k*static_cast<double>(l));
////				Ok(ik) += OdagO_L(l) * exp(+1.i*k*static_cast<double>(l));
////			}
////			
////			if (abs(k) < 1e-14)
////			{
////				cout << "k=0: " << Ok(ik) << endl;
////			}
////			else if (abs(k-M_PI) < 1e-14)
////			{
////				cout << "k=pi: " << Ok(ik) << endl;
////			}
////		}
//	}
	
	if (SU2)
	{
//		typedef VMPS::HubbardSU2xU1 MODEL;
//		MODEL H(L,{{"U",20.},{"OPEN_BC",false},{"CALC_SQUARE",false}});
//		qarray<2> Qc = {1,N};
		typedef VMPS::HeisenbergSU2 MODEL;
		MODEL H(L,{{"J",J},{"OPEN_BC",false},{"CALC_SQUARE",false},{"Ly",Ly},{"D",D}});
		qarray<1> Qc = {1};
		
		H.transform_base(Qc);
		lout << H.info() << endl;
		
		MODEL::uSolver DMRG(VERB);
		Eigenstate<MODEL::StateUd> g;
		GlobParams.Qinit = 6;
		DMRG.userSetGlobParam();
		DMRG.GlobParam = GlobParams;
		DMRG.userSetLanczosParam();
		DMRG.LanczosParam = LanczosParams;
		DMRG.edgeState(H, g, Qc);
		
		vector<Mpo<MODEL::Symmetry> > Odag(L);
		vector<Mpo<MODEL::Symmetry> > O(L); 
		
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
		cout << endl;
		
//		SF += g.state.structure_factor(Odag[i0], O[j0], 0., 2.*M_PI, Nk, DMRG::VERBOSITY::STEPWISE);
//		
//		ofstream Filer(make_string("SF_Sym=",MODEL::Symmetry::name(),"_L=",L,"_i0=",i0,"_j0=",j0,".dat"));
//		for (int ik=0; ik<SF.rows(); ++ik)
//		{
//			// the result has the last point repeated: 0 = 2*pi
//			Filer << ik*2.*M_PI/(SF.rows()-1) << "\t" << SF(ik).real() << "\t" << SF(ik).imag() << endl;
//		}
//		Filer.close();
//		lout << make_string("SF_Sym=",MODEL::Symmetry::name(),"_L=",L,"_i0=",i0,"_j0=",j0,".dat") << " saved!" << endl;
		
//		N = 50; // Ncell (number of unit cells), L=Lcell (size of unit cell)
		
//		ofstream FilerFT(make_string("FT_Sym=",MODEL::Symmetry::name(),"_L=",L,"_i0=",i0,"_j0=",j0,".dat"));
//		for (size_t i=0; i<N; i++)
//		{
//			double k = i * 2.*M_PI/N;
//			FilerFT << k << "\t" << Ok(i).real() << "\t" << Ok(i).imag() << endl;
//		}
//		FilerFT.close();
//		lout << make_string("FT_Sym=",MODEL::Symmetry::name(),"_L=",L,"_i0=",i0,"_j0=",j0,".dat") << " saved!" << endl;
		
		ArrayXXcd SF(Nk,2);
		
		ArrayXXd Sij_cell(2,2);
		Sij_cell(0,0) = avg(g.state, H.SdagS(0,0), g.state);
		Sij_cell(0,1) = avg(g.state, H.SdagS(0,1), g.state);
		Sij_cell(1,0) = avg(g.state, H.SdagS(1,0), g.state);
		Sij_cell(1,1) = avg(g.state, H.SdagS(1,1), g.state);
		
		// variant 1:
//		#pragma omp parallel for
//		for (size_t ik=0; ik<Nk; ++ik)
//		{
//			double k = ik * 2.*M_PI/(Nk-1); // last point is repeated (0 and 2*pi), hence the amount of k-points is Nk-1
//			
//			complex<double> val_ssf = g.state.SFpoint(Sij_cell, Odag,O, k, DMRG::VERBOSITY::SILENT);
//			
//			cout << "k=" << k << ", ssf=" << val_ssf << endl;
//			
//			SF(ik,0) = k;
//			SF(ik,1) = val_ssf;
//		}
		// variant 2:
		SF = g.state.SF(Sij_cell, Odag,O, 0.,2.*M_PI,Nk, DMRG::VERBOSITY::ON_EXIT);
		
		ofstream FilerSF(make_string("SF_Sym=",MODEL::Symmetry::name(),"_L=",L,"_J=",J,".dat")); 
		for (size_t ik=0; ik<SF.rows(); ++ik)
		{
			FilerSF << SF(ik,0).real() << "\t" << SF(ik,1).real() << "\t" << SF(ik,1).imag() << endl;
		}
		FilerSF.close();
		cout << make_string("SF_Sym=",MODEL::Symmetry::name(),"_L=",L,"_J=",J,".dat") << " saved!" << endl;
		
		N = 100;
		ArrayXXcd FT(N+1,2);
		fill_OdagO_SU2(g,Oavg,Odagavg);
		
		#pragma omp parallel for
		for (size_t ik=0; ik<=N; ++ik)
		{
			double k = ik * 2.*M_PI/N;
			
			complex<double> val_fft = calc_FT_full(k,g,Oavg,Odagavg);
			
			cout << "k=" << k << ", fft=" << val_fft << endl;
			
			FT(ik,0) = k;
			FT(ik,1) = val_fft;
		}
		
		ofstream FilerFT(make_string("FT_Sym=",MODEL::Symmetry::name(),"_L=",L,"_J=",J,".dat")); 
		for (size_t ik=0; ik<FT.rows(); ++ik)
		{
			FilerFT << FT(ik,0).real() << "\t" << FT(ik,1).real() << "\t" << FT(ik,1).imag() << endl;
		}
		FilerFT.close();
		cout << make_string("FT_Sym=",MODEL::Symmetry::name(),"_L=",L,"_J=",J,".dat") << " saved!" << endl;
	}
}

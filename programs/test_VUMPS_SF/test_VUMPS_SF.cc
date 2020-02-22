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

#include "VUMPS/VumpsSolver.h"
#include "VUMPS/VumpsLinearAlgebra.h"
//#include "VUMPS/UmpsCompressor.h"
//#include "models/Heisenberg.h"
// #include "models/HeisenbergU1.h"
// #include "models/HeisenbergU1XXZ.h"
//#include "models/HeisenbergXXZ.h"
#include "models/HeisenbergSU2.h"
//#include "models/HubbardU1xU1.h"
//#include "models/Hubbard.h"
// #include "models/HubbardSU2xSU2.h"
// #include "models/HubbardSU2.h"
//#include "models/HubbardSU2xU1.h"
//#include "models/KondoSU2xU1.h"
//#include "models/KondoU1xU1.h"
//#include "models/KondoU0xSU2.h"

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

vector<vector<vector<vector<ArrayXd> > > > OdagO;

template<typename Model>
void fill_OdagO_SU2 (const Eigenstate<typename Model::StateUd> &g)
{
	OdagO.resize(L);
	for (size_t x0=0; x0<L; ++x0)
	{
		OdagO[x0].resize(Ly);
		for (size_t y0=0; y0<Ly; ++y0)
		{
			OdagO[x0][y0].resize(L);
			for (size_t x1=0; x1<L; ++x1)
			{
				OdagO[x0][y0][x1].resize(Ly);
				for (size_t y1=0; y1<Ly; ++y1)
				{
					OdagO[x0][y0][x1][y1].resize(N);
				}
			}
		}
	}
	
	Geometry2D Geo(SNAKE,L,Ly,1.,true);
	
	#pragma omp parallel for collapse(5)
	for (size_t x0=0; x0<L; ++x0)
	for (size_t x1=0; x1<L; ++x1)
	for (size_t y0=0; y0<Ly; ++y0)
	for (size_t y1=0; y1<Ly; ++y1)
	for (size_t n=0; n<N; ++n)
	{
		int i0 = Geo(x0,y0);
		int i1 = Geo(x1,y1);
		
		Model Htmp(L*Ly*n+i1+i0+4,{{"OPEN_BC",false},{"D",D},{"CALC_SQUARE",false}});
		OdagO[x0][y0][x1][y1](n) = avg(g.state, Htmp.SdagS(i0,L*Ly*n+i1), g.state);
	}
}

complex<double> calc_FT (double kx, int iky)
{
	ArrayXXcd FTintercell(L,L);
	
	Geometry2D Geo(SNAKE,L,Ly,1.,true);
	
	for (size_t x0=0; x0<L; ++x0)
	for (size_t x1=0; x1<L; ++x1)
	{
		FTintercell(x0,x1) = 0;
		vector<complex<double> > phases_m0 = Geo.FTy_phases(x0,iky,1);
		vector<complex<double> > phases_p1 = Geo.FTy_phases(x1,iky,0);
		
		for (size_t y0=0; y0<Ly; ++y0)
		for (size_t y1=0; y1<Ly; ++y1)
		{
			int i0 = Geo(x0,y0);
			int i1 = Geo(x1,y1);
			
			FTintercell(x0,x1) += phases_m0[i0] * phases_p1[i1] * OdagO[x0][y0][x1][y1](0);
			
			for (size_t n=1; n<N; ++n)
			{
				FTintercell(x0,x1) += phases_m0[i0] * phases_p1[i1] *
				                      (
				                       OdagO[x0][y0][x1][y1](n) * exp(-1.i*kx*static_cast<double>(L*n)) + 
				                       OdagO[x1][y1][x0][y0](n) * exp(+1.i*kx*static_cast<double>(L*n)) // careful: 0-1 exchange here!
				                      );
			}
		}
	}
	
	complex<double> res = 0;
	
	for (size_t x0=0; x0<L; ++x0)
	for (size_t x1=0; x1<L; ++x1)
	{
		double x0d = x0;
		double x1d = x1;
		
		res += 1./L * exp(-1.i*kx*(x0d-x1d)) * FTintercell(x1,x0); // Attention: (x1,x0) in argument is correct!
	}
	
	return res;
}

//==============================
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
	double V = args.get<double>("V",0.);
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
	
	max_iter = args.get<size_t>("max_iter",150ul);
	min_iter = args.get<size_t>("min_iter",50ul);
	size_t Qinit = args.get<size_t>("Qinit",6);
	D = args.get<size_t>("D",3);
	
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
	GlobParams.min_iterations = min_iter;
	GlobParams.max_iterations = max_iter;
	GlobParams.Dinit = Dinit;
	
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
//		ofstream FilerFT(make_string("FT_Sym=",MODEL::Symmetry::name(),"_L=",L,"_Ly=",Ly,"_i0=",i0,"_j0=",j0,".dat"));
//		for (size_t i=0; i<2*N; i++)
//		{
//			double k = i * 2.*M_PI/(2*N);
//			FilerFT << k << "\t" << Sk(i).real() << "\t" << Sk(i).imag() << endl;
//		}
//		FilerFT.close();
//		
//		lout << make_string("FT_Sym=",MODEL::Symmetry::name(),"_L=",L,"_Ly=",Ly,"_i0=",i0,"_j0=",j0,".dat") << " saved!" << endl;
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
		Geometry2D Geo1cell(SNAKE,  L,Ly,1.,true);
		Geometry2D Geo2cell(SNAKE,2*L,Ly,1.,true);
		
		ArrayXXd Jarray = J * Geo2cell.hopping();
		
		cout << endl << Geo1cell.hopping() << endl << endl;
		cout << endl << Geo2cell.hopping() << endl << endl;
		
		vector<Param> params;
//		params.push_back({"Jfull",Jarray});
//		params.push_back({"OPEN_BC",false});
//		params.push_back({"D",D});
//		params.push_back({"CALC_SQUARE",false});
		
		// typedef VMPS::HubbardSU2 MODEL;
		ArrayXXd tArray = 1. * Geo2cell.hopping();
		ArrayXXd Vxyarray = V * Geo2cell.hopping();
		ArrayXXd Vzarray  = V * Geo2cell.hopping();
		params.push_back({"tFull",tArray});
		params.push_back({"Vxyfull",Vxyarray});
		params.push_back({"Vzfull",Vzarray});
		params.push_back({"Uph",U});
		params.push_back({"OPEN_BC",false});
		params.push_back({"CALC_SQUARE",false});
		// MODEL H(L,params);
//		MODEL H(L,{{"U",2.},{"Vxy",V},{"Vz",V},{"OPEN_BC",false},{"CALC_SQUARE",false}});
		
		typedef VMPS::HeisenbergSU2 MODEL;
		MODEL H(L,{{"J",J},{"OPEN_BC",false},{"CALC_SQUARE",false},{"Ly",Ly},{"D",D}});
		
		qarray<MODEL::Symmetry::Nq> Qc = MODEL::singlet();
		H.transform_base(Qc);
		lout << H.info() << endl;
		
		MODEL::uSolver DMRG(VERB);
		Eigenstate<MODEL::StateUd> g;
		GlobParams.Qinit = 6;
		GlobParams.Dinit = 10;
		DMRG.userSetGlobParam();
		DMRG.GlobParam = GlobParams;
		DMRG.edgeState(H, g, Qc);
		
		for (int iky=0; iky<Ly; ++iky)
//		int iky = Ly/2;
		{
			Geometry2D GeoSnake(SNAKE,     L,Ly,1.,true);
			Geometry2D GeoChess(CHESSBOARD,L,Ly,1.,true);
			for (size_t x=0; x<L; ++x)
			{
				auto phasesSnake = GeoSnake.FTy_phases(x,iky,0);
				auto phasesChess = GeoChess.FTy_phases(x,iky,0);
				
				for (int i=0; i<phasesSnake.size(); ++i)
				{
					cout << "i=" << i << ", x=" << x << ", iky=" << iky << ", snake=" << phasesSnake[i] << ", chess=" << phasesChess[i] << endl;
				}
			}
			
			vector<Mpo<MODEL::Symmetry> > Odag(L*Ly);
			vector<Mpo<MODEL::Symmetry> > O(L*Ly); 
			
			vector<Mpo<MODEL::Symmetry,complex<double> > > Odag_ky(L);
			vector<Mpo<MODEL::Symmetry,complex<double> > > O_ky(L); 
			
			ArrayXd Oavg(L*Ly);
			ArrayXd Odagavg(L*Ly);
			
			for (size_t x=0; x<L; ++x)
			{
				vector<complex<double> > phases_p = Geo1cell.FTy_phases(x,iky,0);
				vector<complex<double> > phases_m = Geo1cell.FTy_phases(x,iky,1);

				O_ky[x]    = H.S_ky   (phases_p);
				Odag_ky[x] = H.Sdag_ky(phases_m);
				
	//			O[l]    = H.n(l);
	//			Odag[l] = H.n(l);
			}
			
			for (size_t l=0; l<L*Ly; ++l)
			{

				O[l]    = H.S(l);
				cout << "Odag:" << endl;
				Odag[l] = H.Sdag(l);
				
				Oavg(l)    = avg(g.state, O[l],    g.state);
				Odagavg(l) = avg(g.state, Odag[l], g.state);
//				
//				O_ky[l].scale(1.,-Oavg(l));
//				Odag_ky[l].scale(1.,-Odagavg(l));
//				
//				O_ky[l].transform_base(Qc);
//				Odag_ky[l].transform_base(Qc);
//				
//				cout << "l=" << l << endl;
//				cout << "<O>="    << avg(g.state, O_ky[l],    g.state) << ", shifted by " << Oavg(l)    << endl;
//				cout << "<Odag>=" << avg(g.state, Odag_ky[l], g.state) << ", shifted by " << Odagavg(l) << endl;
			}
			
			ArrayXXcd SF(Nk,2);
			
			ArrayXXcd Sij_cell(L,L); Sij_cell = 0;
			
			// new 2d:
			for (int x1=0; x1<L; ++x1)
			for (int x2=0; x2<L; ++x2)
			{
				auto phases_m1 = Geo1cell.FTy_phases(x1,iky,1);
				auto phases_p2 = Geo1cell.FTy_phases(x2,iky,0);
				
				for (int y1=0; y1<Ly; ++y1)
				for (int y2=0; y2<Ly; ++y2)
				{
					int index1 = Geo1cell(x1,y1);
					cout << "x1=" << x1 << ", y1=" << y1 << ", index1=" << index1 << endl;
					int index2 = Geo1cell(x2,y2);
					
					if (phases_m1[index1] * phases_p2[index2] != 0.)
					{
						Sij_cell(x1,x2) += phases_m1[index1] * phases_p2[index2] * 
						                   avg(g.state, H.SdagS(index1,index2), g.state);
					}
				}
			}
			
			// old 1d:
//			for (size_t x1=0; x1<L; ++x1)
//			for (size_t x2=0; x2<L; ++x2)
//			{
//				Sij_cell(x1,x2) += avg(g.state, H.SdagS(x1,x2), g.state);
//			}
			
			// new 2d:
			SF = g.state.SF(Sij_cell, Odag_ky,O_ky, L, 0.,2.*M_PI,Nk, DMRG::VERBOSITY::ON_EXIT);
			
			// old 1d:
//			SF = g.state.SF(Sij_cell, Odag,O, L, 0.,2.*M_PI,Nk, DMRG::VERBOSITY::ON_EXIT);
			
			ofstream FilerSF(make_string("SF_Sym=",MODEL::Symmetry::name(),"_iky=",iky,"_L=",L,"_Ly=",Ly,".dat")); 
			for (size_t ik=0; ik<SF.rows(); ++ik)
			{
				FilerSF << SF(ik,0).real() << "\t" << SF(ik,1).real() << "\t" << SF(ik,1).imag() << endl;
			}
			FilerSF.close();
			cout << make_string("SF_Sym=",MODEL::Symmetry::name(),"_iky=",iky,"_L=",L,"_Ly=",Ly,".dat") << " saved!" << endl;
			
			//-------------
			
			N = 20;
			ArrayXXcd FT(N+1,2);
			cout << "beginning fill_OdagO_SU2..." << endl;
			fill_OdagO_SU2<MODEL>(g);
			cout << "fill_OdagO_SU2 done!" << endl;
			
			/////////// test 1D with exlicit loop ///////////
			VectorXd SdagSvec(100);
			for (int l=0; l<100; ++l)
			{
				MODEL Htmp(l+2,{{"OPEN_BC",false},{"D",D},{"CALC_SQUARE",false}});
				SdagSvec(l) = avg(g.state, Htmp.SdagS(0,l), g.state);
				lout << "l=" << l << "\t" << SdagSvec(l) << endl;
			}
			VectorXcd Skvec(100); Skvec.setZero();
			int Lcell = 2;
			for (int ik=0; ik<100; ++ik)
			{
				double k = 2.*M_PI/100.*ik;
				Skvec(ik) = 0;
				for (int l=-38; l<38; ++l)
				for (int j=0; j<Lcell; ++j)
				{
					Skvec(ik) += 1./Lcell * exp(-1.i*k*(double(l)-double(j))) * SdagSvec(abs(l-j));
				}
			}
			ofstream FilerFTexplicit(make_string("FTexplicit_Sym=",MODEL::Symmetry::name(),"_iky=",iky,"_L=",Lcell,"_Ly=",Ly,".dat")); 
			for (int ik=0; ik<100; ++ik)
			{
				double k = 2.*M_PI/100.*ik;
				FilerFTexplicit << k << "\t" << Skvec(ik).real() << "\t" << Skvec(ik).imag() << "\t" << abs(Skvec(ik)) << endl;
			}
			FilerFTexplicit.close();
			lout << make_string("FTexplicit_Sym=",MODEL::Symmetry::name(),"_iky=",iky,"_L=",Lcell,"_Ly=",Ly,".dat") << " saved!" << endl;
			/////////// test 1D with exlicit loop ///////////
			
			
			#pragma omp parallel for
			for (size_t ikx=0; ikx<=N; ++ikx)
			{
				double kx = ikx * 2.*M_PI/N;
				
				complex<double> val_FT = calc_FT(kx,iky);
				
				#pragma omp critical
				{
					cout << "k=" << kx << ", FT=" << val_FT << endl;
				}
				
				FT(ikx,0) = kx;
				FT(ikx,1) = val_FT;
			}
			
			ofstream FilerFT(make_string("FT_Sym=",MODEL::Symmetry::name(),"_iky=",iky,"_L=",L,"_Ly=",Ly,".dat")); 
			for (size_t ik=0; ik<FT.rows(); ++ik)
			{
				FilerFT << FT(ik,0).real() << "\t" << FT(ik,1).real() << "\t" << FT(ik,1).imag() << endl;
			}
			FilerFT.close();
			cout << make_string("FT_Sym=",MODEL::Symmetry::name(),"_iky=",iky,"_L=",L,"_Ly=",Ly,".dat") << " saved!" << endl;
			
			/////////// test 1D back transform ///////////
			lout << endl << "back transform:" << endl;
			for (int l=0; l<min(FT.rows(),SF.rows()); ++l)
			{
				complex<double> resFT = 0;
				complex<double> resSF = 0;
				// because k=0 and k=2*pi is repeated
				for (int ik=0; ik<FT.rows()-1; ++ik)
				{
					resFT += exp(+1.i*double(l)*FT(ik,0)) * FT(ik,1);
				}
				for (int ik=0; ik<SF.rows()-1; ++ik)
				{
					resSF += exp(+1.i*double(l)*SF(ik,0)) * SF(ik,1);
				}
				resFT /= FT.rows()-1; 
				resSF /= SF.rows()-1;
				cout << "l=" << l << ", from FT=" << resFT.real() << ", from SF=" << resSF.real() << ", exact=" << SdagSvec(l) << endl;
			}
		}
	}
}

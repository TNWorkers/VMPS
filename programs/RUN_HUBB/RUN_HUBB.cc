#define USE_HDF5_STORAGE
#include "util/LapackManager.h"
//#define EIGEN_DONT_PARALLELIZE
#define DMRG_DONT_USE_OPENMP

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_DEFAULT_INDEX_TYPE int

#include <iostream>
#include <fstream>
#include <complex>
#include <variant>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

#include "ArgParser.h"

#include "solvers/DmrgSolver.h"
#include "VUMPS/VumpsSolver.h"
#include "VUMPS/VumpsLinearAlgebra.h"
#include "models/HubbardSU2xSU2.h"
#include "models/HubbardSU2xU1.h"
#include "models/Hubbard.h"

#include "Geometry2D.h" // from TOOLS
#include "NestedLoopIterator.h" // from TOOLS
#include "models/ParamCollection.h"

size_t L, Ncells, Ly;
int volume;
double t, tRung, U, J, V, Vxy, Vz, Vext, X;
int fullMmax;
int M, N, S, T;
double alpha;
double str_tol;
DMRG::VERBOSITY::OPTION VERB;
double Emin = 0.;
double emin = 0.;
bool STRUCTURE, CONTRACTIONS, CALC_TSQ, CALC_BOW, VUMPS, PBC, CALC_TGAP, CALC_SGAP;
string wd; // working directory
string base;

Eigenstate<MODEL::StateXd> g_fix;
Eigenstate<MODEL::StateUd> g_foxy;

double z()
{
	if (VUMPS)
	{
		if      (Ly==1) return 2.;
		else if (Ly==2) return 3.;
		else            return 4.;
	}
	else
	{
		if      (Ly==1) return 2.-2./L;
		else if (Ly==2) return 3.-2./L;
		else            return 4.-2./L;
	}
}

double e_empty()
{
	return 0.125*z()*V+0.5*U;
}

struct Obs
{
	Eigen::MatrixXd nh;
	Eigen::MatrixXd ns;
	Eigen::MatrixXd tz;
	Eigen::MatrixXd tx;
	Eigen::MatrixXd ity;
	Eigen::MatrixXd nhvar;
	Eigen::MatrixXd nsvar;
	Eigen::MatrixXd opBOW;
	Eigen::MatrixXd entropy;
	double energy;
	double dedV;
	Eigen::MatrixXd finite_entropy; // for finite systems
	double Tsq; // T^2
	double Ssq; // S^2
	
	double BOW, CDW, SDW;
	
	double energyS;
	double energyT;
	
	double S_M;
	double T_M;
	double B_M;
	double C_M;
	double S_Gamma;
	double T_Gamma;
	double B_Gamma;
	double C_Gamma;
	
	Eigen::MatrixXd SdagS;
	Eigen::MatrixXd TdagT;
	Eigen::MatrixXd BdagB;
	Eigen::MatrixXd CdagC;
	
	vector<vector<Eigen::MatrixXd> > spectrum;
	
	void resize (size_t Lx, size_t Ly, size_t Lobs)
	{
		nh.resize(Lx,Ly); nh.setZero();
		ns.resize(Lx,Ly); ns.setZero();
		tz.resize(Lx,Ly); tz.setZero();
		tx.resize(Lx,Ly); tx.setZero();
		ity.resize(Lx,Ly); ity.setZero();
		nhvar.resize(Lx,Ly); nhvar.setZero();
		nsvar.resize(Lx,Ly); nsvar.setZero();
		opBOW.resize(Lx,Ly); opBOW.setZero();
		
		entropy.resize(Lx,Ly); entropy.setZero();
		finite_entropy.resize(Lx-1,Ly);
		
		// format:
		// n x0 y0 x1 y1 value
		SdagS.resize(Ncells*Lx*Ly*Lx*Ly,6); SdagS.setZero();
		TdagT.resize(Ncells*Lx*Ly*Lx*Ly,6); TdagT.setZero();
		BdagB.resize(Ncells*Lx*Ly*Lx*Ly,6); BdagB.setZero();
		CdagC.resize(Ncells*Lx*Ly*Lx*Ly,6); CdagC.setZero();
		
		spectrum.resize(Lx);
		for (int l=0; l<L; ++l) spectrum[l].resize(Ly);
	}
};

Obs obs;

vector<vector<vector<vector<ArrayXd> > > > SdagS;
vector<vector<vector<vector<ArrayXd> > > > TdagT;
vector<vector<vector<vector<ArrayXd> > > > BdagB;
vector<vector<vector<vector<ArrayXd> > > > CdagC;

void resize_OdagO (size_t Ncells)
{
	SdagS.resize(L);
	TdagT.resize(L);
	BdagB.resize(L);
	CdagC.resize(L);
	for (size_t x0=0; x0<L; ++x0)
	{
		SdagS[x0].resize(Ly);
		TdagT[x0].resize(Ly);
		BdagB[x0].resize(Ly);
		CdagC[x0].resize(Ly);
		for (size_t y0=0; y0<Ly; ++y0)
		{
			SdagS[x0][y0].resize(L);
			TdagT[x0][y0].resize(L);
			BdagB[x0][y0].resize(L);
			CdagC[x0][y0].resize(L);
			for (size_t x1=0; x1<L; ++x1)
			{
				SdagS[x0][y0][x1].resize(Ly);
				TdagT[x0][y0][x1].resize(Ly);
				BdagB[x0][y0][x1].resize(Ly);
				CdagC[x0][y0][x1].resize(Ly);
				for (size_t y1=0; y1<Ly; ++y1)
				{
					SdagS[x0][y0][x1][y1].resize(Ncells);
					TdagT[x0][y0][x1][y1].resize(Ncells);
					BdagB[x0][y0][x1][y1].resize(Ncells);
					CdagC[x0][y0][x1][y1].resize(Ncells);
					
					SdagS[x0][y0][x1][y1].setZero();
					TdagT[x0][y0][x1][y1].setZero();
					BdagB[x0][y0][x1][y1].setZero();
					CdagC[x0][y0][x1][y1].setZero();
				}
			}
		}
	}
}

void fill_OdagO (size_t L, size_t Ly, size_t n, const Eigenstate<MODEL::StateUd> &g, bool CALC_S=true, bool CALC_T=true, bool CALC_B=true, bool CALC_C=true)
{
	Geometry2D Geo(SNAKE,L,Ly,1.,true);
	
	VectorXd Bavg(L*Ly);
	MODEL H1cell(L*Ly+Ly,{{"OPEN_BC",false},{"CALC_SQUARE",false}});
	#pragma omp parallel for collapse(2)
	for (size_t x0=0; x0<L; ++x0)
	for (size_t y0=0; y0<Ly; ++y0)
	{
		int i0 = Geo(x0,y0);
		
		#ifdef USING_U0
		Bavg(i0) = avg(g.state, H1cell.cdagc<UP>(i0,i0+Ly), g.state)+
		           avg(g.state, H1cell.cdagc<DN>(i0,i0+Ly), g.state);
		#else
		Bavg(i0) = avg(g.state, H1cell.cdagc(i0,i0+Ly), g.state);
		#endif
	}
	
	Stopwatch<> CellTimer;
	MODEL Hncell((n+1)*L*Ly+2*Ly,{{"OPEN_BC",false},{"CALC_SQUARE",false}});
	
	cout << "n=" << n << ", length=" << Hncell.length() << endl;
	
	#pragma omp parallel for collapse(4)
	for (size_t x0=0; x0<L; ++x0)
	for (size_t x1=0; x1<L; ++x1)
	for (size_t y0=0; y0<Ly; ++y0)
	for (size_t y1=0; y1<Ly; ++y1)
	{
		int i0 = Geo(x0,y0);
		int i1 = Geo(x1,y1);
		
		#ifdef USING_U0
		if (CALC_S) SdagS[x0][y0][x1][y1](n) = avg(g.state, Hncell.SpSm(i0,L*Ly*n+i1), g.state)+
		                                       avg(g.state, Hncell.SzSz(i0,L*Ly*n+i1), g.state);
		if (CALC_T) TdagT[x0][y0][x1][y1](n) = avg(g.state, Hncell.TpTm(i0,L*Ly*n+i1), g.state)+
		                                       avg(g.state, Hncell.TzTz(i0,L*Ly*n+i1), g.state);
		#else
		if (CALC_S) SdagS[x0][y0][x1][y1](n) = avg(g.state, Hncell.SdagS(i0,L*Ly*n+i1), g.state);
		if (CALC_T) TdagT[x0][y0][x1][y1](n) = avg(g.state, Hncell.TdagT(i0,L*Ly*n+i1), g.state);
		#endif
		#ifdef USING_SO4
		if (CALC_B) BdagB[x0][y0][x1][y1](n) = avg(g.state, Hncell.B(i0,i0+Ly), Hncell.B(L*Ly*n+i1,L*Ly*n+i1+Ly), g.state)-Bavg(i0)*Bavg(i1);
		if (CALC_C) CdagC[x0][y0][x1][y1](n) = avg(g.state, Hncell.C(i0,i0+Ly), Hncell.C(L*Ly*n+i1,L*Ly*n+i1+Ly), g.state);
		#endif
//		#pragma omp critical
//		{
//			lout << "x0=" << x0 << ", x1=" << x1 << ", n=" << n << ", i0=" << i0 << ", i1=" << i1 << ", C=" << CdagC[x0][y0][x1][y1](n) << endl;
//		}
	}
	
	lout << CellTimer.info(make_string("n=",n)) << endl;
}

void save_OdagO (size_t Ncells)
{
	Geometry2D Geo(SNAKE,L,Ly,1.,true);
	
	// save to obs
	NestedLoopIterator Nelly(5,{Ncells,L,Ly,L,Ly});
	for (Nelly=Nelly.begin(); Nelly!=Nelly.end(); ++Nelly)
	{
		int n  = Nelly(0);
		int x0 = Nelly(1);
		int y0 = Nelly(2);
		int x1 = Nelly(3);
		int y1 = Nelly(4);
		if (Ly>1) lout << "n=" << n << ", x0=" << x0 << ", y0=" << y0 << ", x1=" << x1 << ", y1=" << y1 << ", vals=";
		else      lout << "n=" << n << ", x0=" << x0 << ", x1=" << x1 << ", vals=";
		lout << SdagS[x0][y0][x1][y1](n) << ", " 
		     << TdagT[x0][y0][x1][y1](n) << ", " 
		     << BdagB[x0][y0][x1][y1](n) << ", "
		     << CdagC[x0][y0][x1][y1](n) 
		     << endl;
		
		int i0 = Geo(x0,y0);
		int i1 = Geo(x1,y1);
		
		obs.SdagS(Nelly.index(),0) = n;
		obs.SdagS(Nelly.index(),1) = x0;
		obs.SdagS(Nelly.index(),2) = y0;
		obs.SdagS(Nelly.index(),3) = x1;
		obs.SdagS(Nelly.index(),4) = y1;
		obs.SdagS(Nelly.index(),5) = SdagS[x0][y0][x1][y1](n);
		
		obs.TdagT(Nelly.index(),0) = n;
		obs.TdagT(Nelly.index(),1) = x0;
		obs.TdagT(Nelly.index(),2) = y0;
		obs.TdagT(Nelly.index(),3) = x1;
		obs.TdagT(Nelly.index(),4) = y1;
		obs.TdagT(Nelly.index(),5) = TdagT[x0][y0][x1][y1](n);
		
		obs.BdagB(Nelly.index(),0) = n;
		obs.BdagB(Nelly.index(),1) = x0;
		obs.BdagB(Nelly.index(),2) = y0;
		obs.BdagB(Nelly.index(),3) = x1;
		obs.BdagB(Nelly.index(),4) = y1;
		obs.BdagB(Nelly.index(),5) = BdagB[x0][y0][x1][y1](n);
		
		obs.CdagC(Nelly.index(),0) = n;
		obs.CdagC(Nelly.index(),1) = x0;
		obs.CdagC(Nelly.index(),2) = y0;
		obs.CdagC(Nelly.index(),3) = x1;
		obs.CdagC(Nelly.index(),4) = y1;
		obs.CdagC(Nelly.index(),5) = CdagC[x0][y0][x1][y1](n);
	}
}

complex<double> calc_FT (double kx, int iky, size_t Ncells, const vector<vector<vector<vector<ArrayXd> > > > &OdagO)
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
			
			for (size_t n=1; n<Ncells; ++n)
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
		
		res += 1./L * exp(-1.i*kx*(x0d-x1d)) * FTintercell(x0,x1);
	}
	
	return res;
}

//===============================
int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",2);
	Ly = args.get<size_t>("Ly",1);
	volume = L*Ly;
	N = args.get<int>("N",volume);
	T = volume-N+1;
	Ncells = args.get<size_t>("Ncells",80); // maximal amount of unit cells for explicit contractions
	str_tol = args.get<double>("str_tol",1e-3); // SF convergence tolerance with explicit contractions
	t = args.get<double>("t",1.);
	tRung = args.get<double>("tRung",t); // tRung != t for testing only
	U = args.get<double>("U",8.);
	J = args.get<double>("J",0.);
	X = args.get<double>("X",0.);
	V = args.get<double>("V",0.);
	Vxy = args.get<double>("Vxy",V);
	Vz = args.get<double>("Vz",V);
	Vext = args.get<double>("Vext",0.);
	M = args.get<int>("M",0);
	S = abs(M)+1;
	STRUCTURE = args.get<bool>("STRUCTURE",false);
	CONTRACTIONS = args.get<bool>("CONTRACTIONS",false);
	CALC_TSQ = args.get<bool>("CALC_TSQ",false);
	CALC_BOW = args.get<bool>("CALC_BOW",false);
	VUMPS = args.get<bool>("VUMPS",true);
	fullMmax = args.get<int>("fullMmax",0);
	string INIT = args.get<string>("INIT","");
	PBC = args.get<bool>("PBC",false);
	CALC_TGAP = args.get<bool>("CALC_TGAP",false);
	CALC_SGAP = args.get<bool>("CALC_SGAP",false);
	
	VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",2));
	
	wd = args.get<string>("wd","./");
	correct_foldername(wd);
	
	#ifdef USING_SO4
//	base = make_string("U=",U,"_V=",V,"_J=",J,"_L=",L,"_Ly=",Ly);
	base = make_string("L=",L,"_Ly=",Ly,"_t=",t,"_U=",U,"_V=",V,"_J=",J);
	#else
	if (abs(Vext) > 0.)
	{
		base = make_string("L=",L,"_Ly=",Ly,"_N=",N,"_t=",t,"_U=",U,"_Vext=",Vext,"_J=",J);
	}
	else
	{
		if (Vxy==Vz)
		{
			base = make_string("L=",L,"_Ly=",Ly,"_N=",N,"_t=",t,"_U=",U,"_V=",V,"_J=",J);
		}
		else
		{
			base = make_string("L=",L,"_Ly=",Ly,"_N=",N,"_t=",t,"_U=",U,"_Vxy=",Vxy,"_Vz=",Vz,"_J=",J);
		}
	}
	#endif
	if (X != 0.)
	{
		base += make_string("_X=",X);
	}
	if (PBC)
	{
		base += make_string("_BC=PBC");
	}
	string obsfile = make_string(wd,"obs/",base,".h5");
	string statefile = make_string(wd,"state/",base);
	
	DMRG::CONTROL::GLOB GlobParam_fix;
	DMRG::CONTROL::DYN  DynParam_fix;
	VUMPS::CONTROL::GLOB GlobParam_foxy;
	VUMPS::CONTROL::DYN  DynParam_foxy;
	
	size_t min_Nsv = args.get<size_t>("min_Nsv",0ul);
	DynParam_fix.min_Nsv = [min_Nsv] (size_t i) {return min_Nsv;};
	
	size_t lim_alpha = args.get<size_t>("lim_alpha",11);
	double alpha = args.get<double>("alpha",100.);
	DynParam_fix.max_alpha_rsvd = [lim_alpha, alpha] (size_t i) {return (i<lim_alpha)? alpha:0.;};
	
	int max_Nrich = args.get<int>("max_Nrich",-1);
	DynParam_fix.max_Nrich = [max_Nrich] (size_t i) {return max_Nrich;};
	
	GlobParam_fix.Dinit  = args.get<size_t>("Dinit",2ul);
	GlobParam_fix.Dlimit = args.get<size_t>("Dlimit",200ul);
	GlobParam_fix.Qinit = args.get<size_t>("Qinit",10ul);
	GlobParam_fix.min_halfsweeps = args.get<size_t>("Imin",1);
	GlobParam_fix.max_halfsweeps = args.get<size_t>("Imax",22);
	GlobParam_fix.tol_eigval = args.get<double>("tol_eigval",1e-6);
	GlobParam_fix.tol_state = args.get<double>("tol_state",1e-5);
	GlobParam_fix.savePeriod = args.get<size_t>("savePeriod",0ul);
	GlobParam_fix.saveName = args.get<string>("saveName",statefile);
	
	GlobParam_foxy.Dinit  = args.get<size_t>("Dinit",10ul);
	GlobParam_foxy.Dlimit = args.get<size_t>("Dlimit",200ul);
	GlobParam_foxy.Qinit = args.get<size_t>("Qinit",10ul);
	GlobParam_foxy.min_iterations = args.get<size_t>("Imin",6ul);
	GlobParam_foxy.max_iterations = args.get<size_t>("Imax",1000ul);
	GlobParam_foxy.max_iter_without_expansion = args.get<size_t>("max",30ul);
	GlobParam_foxy.fullMmaxBreakoff = args.get<size_t>("Chimax",25000ul);
	
	GlobParam_foxy.tol_eigval = args.get<double>("tol_eigval",1e-6);
	GlobParam_foxy.tol_var = args.get<double>("tol_var",1e-6);
	GlobParam_foxy.tol_state = args.get<double>("tol_state",1e-5);
	GlobParam_foxy.savePeriod = args.get<size_t>("savePeriod",0ul);
	GlobParam_foxy.saveName = args.get<string>("saveName",statefile);
	
	#ifdef USING_SO4
	lout.set(base+".log",wd+"log");
	#else
	lout.set(base+".log",wd+"log");
	#endif
	
	lout << args.info() << endl;
	lout << "wd=" << wd << endl;
	
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	lout << "e_empty=" << e_empty() << endl;
	
	Stopwatch<> Watch;
	
	Geometry2D Geo1cell(SNAKE,1*L,Ly,1.,true); // periodic BC in y = true
	Geometry2D Geo2cell(SNAKE,2*L,Ly,1.,true);
	
	// save to temporary, otherwise std::bad_any_cast
	ArrayXXd tArray, Varray, Vxyarray, Vzarray, Jarray, Xarray, ZeroArray, OneArray, VextArray;
	if (VUMPS)
	{
		tArray    = t * Geo2cell.hopping();
		Varray    = V * Geo2cell.hopping();
		Vxyarray  = Vxy * Geo2cell.hopping();
		Vzarray   = Vz * Geo2cell.hopping();
		VextArray = Vext * Geo2cell.hopping();
		Jarray    = J * Geo2cell.hopping();
		Xarray    = X * Geo2cell.hopping();
		ZeroArray = 0. * Geo2cell.hopping();
		OneArray  = 1. * Geo2cell.hopping();
		lout << "t=" << t << ", V=" << V << ", J=" << J << endl;
		if (volume <= 100) lout << "hopping 2cells=" << endl << Geo2cell.hopping() << endl << endl;
	}
	else
	{
		ArrayXXd tFull;
		tFull = (PBC)? create_1D_PBC(L):Geo1cell.hopping();
		tArray    = t * tFull;
		Varray    = V * tFull;
		Vxyarray  = Vxy * tFull;
		Vzarray   = Vz * tFull;
		VextArray = Vext * tFull;
		Jarray    = J * tFull;
		Xarray    = X * tFull;
		ZeroArray = 0. * tFull;
		OneArray  = 1. * tFull;
		if (volume <= 100) lout << "hopping=" << endl << tFull << endl << endl;
	}
	
	vector<Param> params;
	qarray<MODEL::Symmetry::Nq> Qc, Qc2, QcT, QcS;
	if constexpr (std::is_same<MODEL,VMPS::HubbardSU2xSU2>::value)
	{
		params.push_back({"tFull",tArray});
		params.push_back({"Vfull",Varray});
		params.push_back({"Jfull",Jarray});
		params.push_back({"Xfull",Xarray});
		params.push_back({"U",U});
		if (VUMPS) {params.push_back({"OPEN_BC",false});}
		Qc  = {S,T};
		Qc2 = {S,T};
		QcT = {S,T+2};
		QcS = {S+2,T};
	}
	else if constexpr (std::is_same<MODEL,VMPS::HubbardSU2xU1>::value)
	{
		params.push_back({"tFull",tArray});
		params.push_back({"Vxyfull",Vxyarray});
		params.push_back({"Vzfull",Vzarray});
		params.push_back({"VextFull",VextArray});
		params.push_back({"Jfull",Jarray});
		params.push_back({"Xfull",Xarray});
		if (abs(Vext) > 0.)
		{
			params.push_back({"U",U});
			params.push_back({"Uph",0.});
			lout << termcolor::blue << "Warning: Setting U instead of particle-hole-symmetric Uph, since Vext is specified!" << termcolor::reset << endl;
		}
		else
		{
			params.push_back({"Uph",U});
			params.push_back({"U",0.});
		}
		if (VUMPS) {params.push_back({"OPEN_BC",false});}
		Qc  = {S,N};
		Qc2 = {S,2*N}; // for 2 unit cells
		QcT = {S,N-2};
		QcS = {S+2,N};
	}
	else if (std::is_same<MODEL,VMPS::Hubbard>::value)
	{
		// only 1D implemented for U(0)
		params.push_back({"t",t});
		params.push_back({"Vxy",Vxy});
		params.push_back({"Vz",Vz});
		params.push_back({"X",X});
		params.push_back({"J",J});
		params.push_back({"Uph",U});
		if (VUMPS) {params.push_back({"OPEN_BC",false});}
		Qc  = {};
		Qc2 = {};
	}
	else if (std::is_same<MODEL,VMPS::HubbardU1xU1>::value)
	{
		// only 1D implemented for U(0)
		params.push_back({"t",t});
		params.push_back({"Vxy",Vxy});
		params.push_back({"Vz",Vz});
		params.push_back({"X",X});
		params.push_back({"J",J});
		params.push_back({"Uph",U});
		if (VUMPS) {params.push_back({"OPEN_BC",false});}
		Qc  = {0,N};
		Qc2 = {0,N};
	}
	
	MODEL H(volume,params);
	if (VUMPS) H.transform_base(Qc);
	lout << "•H for ground state:" << endl;
	lout << H.info() << endl;
	
	// To calculate de/dV for finite systems as exp. value of Hamiltonian with only V=1
	vector<Param> dHdV_params;
	dHdV_params.push_back({"tFull",ZeroArray});
	dHdV_params.push_back({"Jfull",ZeroArray});
	dHdV_params.push_back({"Xfull",ZeroArray});
	dHdV_params.push_back({"U",0.});
	dHdV_params.push_back({"Uph",0.});
	if constexpr (std::is_same<MODEL,VMPS::HubbardSU2xSU2>::value)
	{
		dHdV_params.push_back({"Vfull",OneArray});
	}
	else
	{
		if (abs(Vxy)   > 0.) dHdV_params.push_back({"Vxyfull",OneArray});
		if (abs(Vz)    > 0.) dHdV_params.push_back({"Vzfull", OneArray});
		if (abs(Vext)  > 0.) dHdV_params.push_back({"VextFull", OneArray});
	}
	
	MODEL dHdV;
	if (VUMPS)
	{
		// For VUMPS: Hamiltonian with two unit cells for contractions across the cell
		dHdV = MODEL(2*volume,{{"OPEN_BC",false},{"CALC_SQUARE",false}});
		dHdV.transform_base(Qc2,false); // PRINT=false
	}
	else
	{
		dHdV = MODEL(volume,dHdV_params);
	}
	lout << "•H to calculate de/dV (needs no params in VUMPS case, V=1 in finite case):" << endl;
	lout << dHdV.info() << endl;
	
	obs.resize(L,Ly,Ncells);
	
	//================= VUMPS =================
	if (VUMPS)
	{
		MODEL::uSolver Foxy(VERB);
		HDF5Interface target;
		cout << obsfile << endl;
		target = HDF5Interface(obsfile,WRITE);
		target.close();
		
		auto measure_and_save = [&H,&dHdV,&target,&params,&Geo1cell,&Geo2cell,&Foxy,&obsfile,&Qc](size_t j) -> void
		{
			if (Foxy.errVar() < 1e-8 or Foxy.FORCE_DO_SOMETHING == true)
			{
				std::stringstream bond;
				target = HDF5Interface(obsfile,REWRITE);
				bond << g_foxy.state.calc_fullMmax();
				cout << termcolor::red << "Measure at M=" << bond.str() << ", if possible" << termcolor::reset << endl;
				
				//----------k-points to calculate----------
				vector<pair<double,int> > kxy;
				if (Ly == 1)
				{
					kxy.push_back(make_pair(0.,0));
					kxy.push_back(make_pair(M_PI,0));
				}
				else
				{
//					kxy.push_back(make_pair(0.,0)); // Γ point (0,0)
					kxy.push_back(make_pair(M_PI,Ly/2)); // M point (π,π)
				}
				
				#ifndef USING_U0
				if (STRUCTURE)
				{
					for (int i=0; i<kxy.size(); ++i)
					{
						double kx = kxy[i].first;
						int   iky = kxy[i].second;
						
						vector<Mpo<MODEL::Symmetry,complex<double> > > Sdag_ky(L);
						vector<Mpo<MODEL::Symmetry,complex<double> > > S_ky   (L);
						vector<Mpo<MODEL::Symmetry,complex<double> > > Tdag_ky(L);
						vector<Mpo<MODEL::Symmetry,complex<double> > > T_ky   (L);
						vector<Mpo<MODEL::Symmetry,complex<double> > > B_ky   (L);
						vector<Mpo<MODEL::Symmetry,complex<double> > > Bdag_ky(L);
						
						MODEL Htmp(volume+1,{});
						
						ArrayXd Bavg(L);
						for (size_t x=0; x<L; ++x)
						{
							Bavg(x) = avg(g_foxy.state, Htmp.cdagc(x,x+1), g_foxy.state);
//							lout << "x=" << x << ", Bavg(x)=" << Bavg(x) << endl;
						}
						
						// Fourier transform of operators in y-direction
						for (size_t x=0; x<L; ++x)
						{
							vector<complex<double> > phases_p = Geo1cell.FTy_phases(x,iky,0);
							vector<complex<double> > phases_m = Geo1cell.FTy_phases(x,iky,1);
							
							Sdag_ky[x] = H.Sdag_ky(phases_m);
							S_ky[x]    = H.S_ky   (phases_p);
							
							Tdag_ky[x] = H.Tdag_ky(phases_m);
							T_ky[x]    = H.T_ky   (phases_p);
							
							#ifdef USING_SO4
							Bdag_ky[x] = VMPS::HubbardSU2xSU2BondOperator<complex<double> >(volume+1,{{"x",x},{"shift",-Bavg(x)}});
							B_ky[x]    = VMPS::HubbardSU2xSU2BondOperator<complex<double> >(volume+1,{{"x",x},{"shift",-Bavg(x)}});
							#endif
							
							Sdag_ky[x].transform_base(Qc,false); // PRINT=false
							S_ky[x].transform_base(Qc,false);
							
							Tdag_ky[x].transform_base(Qc,false);
							T_ky[x].transform_base(Qc,false);
							
							#ifdef USING_SO4
							Bdag_ky[x].transform_base(Qc,false);
							B_ky[x].transform_base(Qc,false);
							#endif
						}
						lout << "Fourier transform in y-direction done!" << endl;
						
						// Calculate expectation values within the cell
						ArrayXXcd Sij_cell(L,L); Sij_cell = 0;
						ArrayXXcd Tij_cell(L,L); Tij_cell = 0;
						ArrayXXcd Bij_cell(L,L); Bij_cell = 0;
						
						for (size_t x1=0; x1<L; ++x1)
						for (size_t x2=0; x2<L; ++x2)
						{
							auto phases_m1 = Geo1cell.FTy_phases(x1,iky,1);
							auto phases_p2 = Geo1cell.FTy_phases(x2,iky,0);
							
							for (size_t y1=0; y1<Ly; ++y1)
							for (size_t y2=0; y2<Ly; ++y2)
							{
								size_t index1 = Geo1cell(x1,y1);
								size_t index2 = Geo1cell(x2,y2);
								
								if (phases_m1[index1] * phases_p2[index2] != 0.)
								{
									Sij_cell(x1,x2) += phases_m1[index1] * phases_p2[index2] * 
									                   avg(g_foxy.state, H.SdagS(index1,index2), g_foxy.state);
									Tij_cell(x1,x2) += phases_m1[index1] * phases_p2[index2] * 
									                   avg(g_foxy.state, H.TdagT(index1,index2), g_foxy.state);
//									Bij_cell(x1,x2) += phases_m1[index1] * phases_p2[index2] * 
//									                    avg(
//									                      g_foxy.state, 
//									                       VMPS::HubbardSU2xSU2BondOperator<double>(volume+1,{{"x",index1},{"shift",-Bavg(index1)}}),
//									                       VMPS::HubbardSU2xSU2BondOperator<double>(volume+1,{{"x",index2},{"shift",-Bavg(index2)}}), 
//									                      g_foxy.state
//									                       );
									Bij_cell(x1,x2) += phases_m1[index1] * phases_p2[index2] * 
									                   (avg(g_foxy.state, Htmp.cdagc(index1,index1+1), Htmp.cdagc(index2,index2+1), g_foxy.state)
									                    -Bavg(index1)*Bavg(index2));
								}
							}
						}
						
						#ifdef USING_SO4
						cout << "Bij_cell=" << endl << Bij_cell << endl << endl;
						
						cout << avg(g_foxy.state, H.cdagc(0,1), H.cdagc(0,1), g_foxy.state) - Bavg(0)*Bavg(0) << endl;
						cout << avg(g_foxy.state, H.cdagc(0,1), H.cdagc(1,2), g_foxy.state) - Bavg(0)*Bavg(1) << endl;
						cout << avg(g_foxy.state, H.cdagc(0,1), H.cdagc(2,3), g_foxy.state) - Bavg(0)*Bavg(2) << endl;
						
						cout << sqrt(3.)*avg(g_foxy.state, H.Tdag(0), H.T(1), g_foxy.state) << "\t" << avg(g_foxy.state, H.TdagT(0,1), g_foxy.state) << endl;
						cout << sqrt(3.)*avg(g_foxy.state, H.Sdag(0), H.S(1), g_foxy.state) << "\t" << avg(g_foxy.state, H.SdagS(0,1), g_foxy.state) << endl;
						
						cout << "should give zero:" << endl;
						cout << avg(g_foxy.state, VMPS::HubbardSU2xSU2BondOperator<double>(volume+1,{{"x",1ul},{"shift",-Bavg(1)}}), g_foxy.state) << endl;
						cout << "should be same:" << endl;
						cout << Bavg(0) << "\t" 
						     << avg(g_foxy.state, VMPS::HubbardSU2xSU2BondOperator<double>(volume+1,{{"x",0ul},{"shift",1e-15}}), g_foxy.state) << endl;
						cout << Bavg(1) << "\t" 
						     << avg(g_foxy.state, VMPS::HubbardSU2xSU2BondOperator<double>(volume+1,{{"x",1ul},{"shift",1e-15}}), g_foxy.state) << endl;
						
						cout << "<O*O>:" << endl;
						cout << avg(g_foxy.state, 
						            VMPS::HubbardSU2xSU2BondOperator<double>(volume+1,{{"x",1ul},{"shift",0.}}),
						            VMPS::HubbardSU2xSU2BondOperator<double>(volume+1,{{"x",2ul},{"shift",0.}}), 
						            g_foxy.state)
						     << "\t" << avg(g_foxy.state, H.cdagc(1,2), H.cdagc(2,3), g_foxy.state) << endl;
						cout << "tests done!" << endl;
						#endif
						
//						cout << VMPS::HubbardSU2xSU2BondOperator<double>(volume+1,{{"x",1ul},{"shift",0.}}) << endl;
						
						// Calculate full structure factor
						complex<double> SF_S, SF_T, SF_B, SF_C;
						#pragma omp parallel sections
						{
//							#pragma omp section
//							{
//								SF_S = g_foxy.state.SFpoint(Sij_cell, Sdag_ky,S_ky, L, kx, DMRG::VERBOSITY::ON_EXIT);
//							}
//							#pragma omp section
//							{
//								SF_T = g_foxy.state.SFpoint(Tij_cell, Tdag_ky,T_ky, L, kx, DMRG::VERBOSITY::ON_EXIT);
//							}
//							#pragma omp section
							{
								#ifdef USING_SO4
								SF_B = g_foxy.state.SFpoint(Bij_cell, Bdag_ky,B_ky, L, kx, DMRG::VERBOSITY::STEPWISE);
								#endif
							}
						}
						// Umps::SF returns 2x2 array with rows in the format:
						// 0,SF(0)
						// π,SF(π)
						
						// Γ point
						if (kx == 0. and iky == 0)
						{
							obs.S_Gamma = isReal(SF_S);
							obs.T_Gamma = isReal(SF_T);
							obs.B_Gamma = isReal(SF_B);
							obs.C_Gamma = isReal(SF_C);
							lout << termcolor::red << "S_Γ=" << SF_S << ", T_Γ=" << SF_T << ", B_Γ=" << SF_B << ", C_Γ=" << SF_C << termcolor::reset << endl;
						}
						// M point
						else if (kx == M_PI and iky == Ly/2)
						{
							obs.S_M = isReal(SF_S);
							obs.T_M = isReal(SF_T);
							obs.B_M = isReal(SF_B);
							obs.C_M = isReal(SF_C);
							lout << termcolor::red << "S_M=" << SF_S << ", T_M=" << SF_T << ", B_M=" << SF_B << ", C_M=" << SF_C << termcolor::reset << endl;
						}
					}
				}
				#endif
				
				if (CONTRACTIONS)
				{
					resize_OdagO(Ncells);
					Stopwatch<> ContractionTimer;
					int nmax = -1;
					
					for (int i=0; i<kxy.size(); ++i)
					{
						double kx  = kxy[i].first;
						int    iky = kxy[i].second;
						
						// fill contractions within the cell and for the first neighbour cell
						if (nmax < 0)
						{
							fill_OdagO(L, Ly, 0, g_foxy); 
							nmax = 0;
						}
						if (nmax < 1)
						{
							fill_OdagO(L, Ly, 1, g_foxy);
							nmax = 1;
						}
						complex<double> SF_S = calc_FT(kx, iky, 1, SdagS);
						complex<double> SF_T = calc_FT(kx, iky, 1, TdagT);
						complex<double> SF_B = calc_FT(kx, iky, 1, BdagB);
//						complex<double> SF_C = calc_FT(kx, iky, 1, CdagC);
						
						bool CALC_S = true;
						bool CALC_T = true;
						bool CALC_B = true;
						bool CALC_C = false;
						
						for (int n=2; n<Ncells; ++n)
						{
//							lout << boolalpha << "S=" << CALC_S << ", T=" << CALC_T << ", B=" << CALC_B << endl;
							
							if (nmax < n)
							{
								fill_OdagO(L, Ly, n, g_foxy, CALC_S, CALC_T, CALC_B, CALC_C); // add contractions for n-th unit cell
								nmax = n;
							}
							
							// FT for n unit cells
							complex<double> SF_S_new = calc_FT(kx, iky, n, SdagS);
							complex<double> SF_T_new = calc_FT(kx, iky, n, TdagT);
							complex<double> SF_B_new = calc_FT(kx, iky, n, BdagB);
//							complex<double> SF_C_new = calc_FT(kx, iky, n, CdagC);
							
//							CALC_S = (min(abs(SF_S-SF_S_new)/abs(SF_S), abs(SF_S-SF_S_new)) < str_tol)? false:true;
//							CALC_T = (min(abs(SF_T-SF_T_new)/abs(SF_T), abs(SF_T-SF_T_new)) < str_tol)? false:true;
//							CALC_B = (min(abs(SF_B-SF_B_new)/abs(SF_B), abs(SF_B-SF_B_new)) < str_tol)? false:true;
							
							lout << "S: " << SF_S << ", "  << SF_S_new << ", abs=" << abs(SF_S-SF_S_new) << ", rel=" << abs(SF_S-SF_S_new)/abs(SF_S) << endl;
							lout << "T: " << SF_T << ", "  << SF_T_new << ", abs=" << abs(SF_T-SF_T_new) << ", rel=" << abs(SF_T-SF_T_new)/abs(SF_T) << endl;
							lout << "B: " << SF_B << ", "  << SF_B_new << ", abs=" << abs(SF_B-SF_B_new) << ", rel=" << abs(SF_B-SF_B_new)/abs(SF_B) << endl;
//							lout << "C: " << SF_C << ", "  << SF_C_new << ", abs=" << abs(SF_C-SF_C_new) << ", rel=" << abs(SF_C-SF_C_new)/abs(SF_C) << endl;
							
							swap(SF_S,SF_S_new);
							swap(SF_T,SF_T_new);
							swap(SF_B,SF_B_new);
//							swap(SF_C,SF_C_new);
							
							// exit if absolute or relative change smaller than str_tol
							if (min(abs(SF_S-SF_S_new)/abs(SF_S), abs(SF_S-SF_S_new)) < str_tol and 
							    min(abs(SF_T-SF_T_new)/abs(SF_T), abs(SF_T-SF_T_new)) < str_tol and 
							    min(abs(SF_B-SF_B_new)/abs(SF_B), abs(SF_B-SF_B_new)) < str_tol)
//							    min(abs(SF_C-SF_C_new)/abs(SF_C), abs(SF_C-SF_C_new)) < str_tol)
							{
								save_OdagO(n);
								lout << "SF convergence after n=" << n << " unit cells or " << (n+1)*L*Ly << " sites" << endl;
								break;
							}
						}
						
						// Γ point
						if (kx == 0. and iky == 0)
						{
							obs.S_Gamma = isReal(SF_S);
							obs.T_Gamma = isReal(SF_T);
							obs.B_Gamma = isReal(SF_B);
//							obs.C_Gamma = isReal(SF_C);
							
							lout << termcolor::red 
							     << "S(Γ)=" << SF_S 
							     << ", T(Γ)=" << SF_T 
							     << ", B(Γ)=" << SF_B 
//							     << ", C(Γ)=" << SF_C 
							     << termcolor::reset << endl;
						}
						// M point
						else if (kx == M_PI and iky == Ly/2)
						{
							obs.S_M = isReal(SF_S);
							obs.T_M = isReal(SF_T);
							obs.B_M = isReal(SF_B);
//							obs.C_M = isReal(SF_C);
							
							lout << termcolor::red 
							     << "S(M)=" << SF_S 
							     << ", T(M)=" << SF_T 
							     << ", B(M)=" << SF_B 
//							     << ", C(M)=" << SF_C 
							     << termcolor::reset << endl;
						}
					}
					lout << ContractionTimer.info("CONTRACTIONS") << endl;
				}
				
				if (target.HAS_GROUP(bond.str())) {return;}
				
				Stopwatch<> SaveAndMeasure;
				for (size_t x=0; x<L; ++x)
				for (size_t y=0; y<Ly; ++y)
				{
					lout << "x=" << x << ", y=" << y << endl;
					
					obs.nh(x,y)        = avg(g_foxy.state, H.nh(Geo1cell(x,y)), g_foxy.state);
					obs.ns(x,y)        = avg(g_foxy.state, H.ns(Geo1cell(x,y)), g_foxy.state);
					#ifdef USING_SU2
					obs.tz(x,y)        = avg(g_foxy.state, H.Tz(Geo1cell(x,y)), g_foxy.state);
					obs.tx(x,y)        = avg(g_foxy.state, H.Tx(Geo1cell(x,y)), g_foxy.state);
					obs.ity(x,y)        = avg(g_foxy.state, H.iTy(Geo1cell(x,y)), g_foxy.state);
					#endif
//					obs.nhvar(x,y)     = avg(g_foxy.state, H.nhsq(Geo1cell(x,y)), g_foxy.state) - pow(obs.nh(x,y),2);
//					obs.nsvar(x,y)     = avg(g_foxy.state, H.nssq(Geo1cell(x,y)), g_foxy.state) - pow(obs.ns(x,y),2);
//					lout << "nh=" << obs.nh(x,y) << ", var=" << obs.nhvar(x,y) << ", ns=" << obs.ns(x,y) << ", var=" << obs.nsvar(x,y) << endl;
					
					#ifdef USING_U0
					obs.opBOW(x,y)     = +avg(g_foxy.state, dHdV.cdagc<UP>(Geo2cell(x,y),  Geo2cell(x+1,y)), g_foxy.state)
					                     +avg(g_foxy.state, dHdV.cdagc<DN>(Geo2cell(x,y),  Geo2cell(x+1,y)), g_foxy.state)
					                     -avg(g_foxy.state, dHdV.cdagc<UP>(Geo2cell(x+1,y),Geo2cell(x+2,y)), g_foxy.state)
					                     -avg(g_foxy.state, dHdV.cdagc<DN>(Geo2cell(x+1,y),Geo2cell(x+2,y)), g_foxy.state);
					#else
					obs.opBOW(x,y)     = +avg(g_foxy.state, dHdV.cdagc(Geo2cell(x,y),  Geo2cell(x+1,y)), g_foxy.state)
					                     -avg(g_foxy.state, dHdV.cdagc(Geo2cell(x+1,y),Geo2cell(x+2,y)), g_foxy.state);
					#endif
					lout << "opBOW=" << obs.opBOW(x,y) << endl;
					
					obs.entropy(x,y)   = g_foxy.state.entropy()(Geo1cell(x,y));
					lout << "S=" << obs.entropy(x,y) << endl;
					
					obs.spectrum[x][y] = g_foxy.state.entanglementSpectrumLoc(Geo1cell(x,y));
					lout << "spec=" << obs.spectrum[x][y].transpose() << endl;
				}
				
				obs.energy = g_foxy.energy;
				
				//----------de/dV----------
				obs.dedV = 0;
				for (size_t x=0; x<L;  ++x)
				for (size_t y=0; y<Ly; ++y)
				{
					// horizontal
					#ifdef USING_SO4
					{
						obs.dedV += avg(g_foxy.state, dHdV.TdagT(Geo2cell(x,y),Geo2cell(x+1,y)), g_foxy.state);
					}
					#else
					{
						obs.dedV += 0.5 * avg(g_foxy.state, dHdV.TpTm(Geo2cell(x,y),Geo2cell(x+1,y)), g_foxy.state);
						obs.dedV += 0.5 * avg(g_foxy.state, dHdV.TmTp(Geo2cell(x,y),Geo2cell(x+1,y)), g_foxy.state);
						obs.dedV +=       avg(g_foxy.state, dHdV.TzTz(Geo2cell(x,y),Geo2cell(x+1,y)), g_foxy.state);
					}
					#endif
					
					// vertical
					if (Ly > 1)
					{
						double edge_correction = (Ly==2)? 0.5:1.;
						#ifdef USING_SO4
						{
							obs.dedV += edge_correction * avg(g_foxy.state, dHdV.TdagT(Geo2cell(x,y),Geo2cell(x,(y+1)%L)), g_foxy.state);
						}
						#else
						{
							obs.dedV += edge_correction * 0.5 * avg(g_foxy.state, dHdV.TpTm(Geo2cell(x,y),Geo2cell(x,(y+1)%L)), g_foxy.state);
							obs.dedV += edge_correction * 0.5 * avg(g_foxy.state, dHdV.TmTp(Geo2cell(x,y),Geo2cell(x,(y+1)%L)), g_foxy.state);
							obs.dedV += edge_correction *       avg(g_foxy.state, dHdV.TzTz(Geo2cell(x,y),Geo2cell(x,(y+1)%L)), g_foxy.state);
						}
						#endif
					}
				}
				obs.dedV /= (L*Ly);
				//----------de/dV----------
				
				target.create_group(bond.str());
				std::stringstream Dmax;
				Dmax << g_foxy.state.calc_Dmax();
				std::stringstream Mmax;
				Mmax << g_foxy.state.calc_Mmax();
				
				target.save_scalar(g_foxy.state.calc_Dmax(),"Dmax",bond.str());
				target.save_scalar(g_foxy.state.calc_Mmax(),"Mmax",bond.str());
				target.save_scalar(g_foxy.state.calc_fullMmax(),"full Mmax",bond.str());
				target.save_scalar(Foxy.errEigval(),"err_eigval",bond.str());
				target.save_scalar(Foxy.errState(),"err_state",bond.str());
				target.save_scalar(Foxy.errVar(),"err_var",bond.str());
				target.save_scalar(obs.energy,"energy",bond.str());
				target.save_scalar(obs.dedV,"dedV",bond.str());
				
				target.save_matrix(obs.nh,"nh",bond.str());
				target.save_matrix(obs.ns,"ns",bond.str());
				#ifdef USING_SU2
				target.save_matrix(obs.tz,"Tz",bond.str());
				target.save_matrix(obs.tx,"Tx",bond.str());
				target.save_matrix(obs.ity,"iTy",bond.str());
				#endif
//				target.save_matrix(obs.nhvar,"nhvar",bond.str());
//				target.save_matrix(obs.nsvar,"nsvar",bond.str());
				target.save_matrix(obs.opBOW,"opBOW",bond.str());
				target.save_matrix(obs.entropy,"Entropy",bond.str());
				
				for (size_t x=0; x<L; ++x)
				for (size_t y=0; y<Ly; ++y)
				{
					target.save_matrix(obs.spectrum[x][y],make_string("spectrum_x=",x,"_y=",y),bond.str());
				}
				
				if (STRUCTURE or CONTRACTIONS)
				{
					target.save_scalar(obs.S_Gamma,"S_Gamma",bond.str());
					target.save_scalar(obs.T_Gamma,"T_Gamma",bond.str());
					target.save_scalar(obs.B_Gamma,"B_Gamma",bond.str());
					
					target.save_scalar(obs.S_M,"S_M",bond.str());
					target.save_scalar(obs.T_M,"T_M",bond.str());
					target.save_scalar(obs.B_M,"B_M",bond.str());
					
					target.save_matrix(obs.SdagS,"SiSj",bond.str());
					target.save_matrix(obs.TdagT,"TiTj",bond.str());
					target.save_matrix(obs.BdagB,"BiBj",bond.str());
					target.save_matrix(obs.CdagC,"CiCj",bond.str());
				}
				target.close();
				stringstream ss;
				ss << "Calcuated and saved observables for M=" << g_foxy.state.calc_fullMmax() << " to " << obsfile;
				
				if (Foxy.get_verbosity() >= DMRG::VERBOSITY::HALFSWEEPWISE) {lout << SaveAndMeasure.info(ss.str()) << endl << endl;}
			}
		};
		
		if (fullMmax > 0)
		{
			g_foxy.state.load(make_string(statefile,"_fullMmax=",fullMmax));
			measure_and_save(0);
		}
		else
		{
			bool USE_STATE = false;
			if (INIT != "")
			{
				g_foxy.state.load(wd+"init/"+INIT);
				USE_STATE = true;
			}
			Foxy.userSetGlobParam();
			Foxy.userSetDynParam();
			Foxy.GlobParam = GlobParam_foxy;
			Foxy.DynParam = DynParam_foxy;
			Foxy.DynParam.doSomething = measure_and_save;
			Foxy.DynParam.iteration = [](size_t i) -> UMPS_ALG::OPTION {return UMPS_ALG::PARALLEL;};
			Foxy.set_log(1, wd+"log/e0_"+base+".log",
			                wd+"log/err-eigval_"+base+".log",
			                wd+"log/err-var_"+base+".log",
			                wd+"log/err-state_"+base+".log");
			Foxy.edgeState(H, g_foxy, Qc, LANCZOS::EDGE::GROUND, USE_STATE);
			measure_and_save(0);
		}
		
		emin = g_foxy.energy;
	}
	//================= Fix =================
	else
	{
		MODEL::Solver Fix(VERB);
		
		Fix.userSetGlobParam();
		Fix.userSetDynParam();
		Fix.GlobParam = GlobParam_fix;
		Fix.DynParam = DynParam_fix;
		
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				Fix.edgeState(H, g_fix, Qc, LANCZOS::EDGE::GROUND);
			}
			#pragma omp section
			{
				if (CALC_TGAP)
				{
					MODEL::Solver FixT(DMRG::VERBOSITY::ON_EXIT);
					auto g_fixT = g_fix;
					FixT.edgeState(H, g_fixT, QcT, LANCZOS::EDGE::GROUND);
					obs.energyT = g_fixT.energy/volume;
				}
			}
			#pragma omp section
			{
				if (CALC_SGAP)
				{
					MODEL::Solver FixS(DMRG::VERBOSITY::ON_EXIT);
					auto g_fixS = g_fix;
					FixT.edgeState(H, g_fixS, QcS, LANCZOS::EDGE::GROUND);
					obs.energyS = g_fixS.energy/volume;
				}
			}
		}
		
		Stopwatch<> ObsWatch;
		
		obs.energy = g_fix.energy/volume;
		obs.dedV = avg(g_fix.state, dHdV, g_fix.state)/volume;
		
		lout << "Tgap=" << L*(obs.energyT-obs.energy) << endl;
		lout << "Sgap=" << L*(obs.energyS-obs.energy) << endl;
		
		#pragma omp parallel for collapse(2)
		for (size_t x=0; x<L-1; ++x)
		for (size_t y=0; y<Ly; ++y)
		{
			obs.nh(x,y) = avg(g_fix.state, H.nh(Geo1cell(x,y)), g_fix.state);
			obs.ns(x,y) = avg(g_fix.state, H.ns(Geo1cell(x,y)), g_fix.state);
//			obs.nhvar(x,y) = avg(g_fix.state, H.nhsq(Geo1cell(x,y)), g_fix.state) - pow(obs.nh(x,y),2);
//			obs.nsvar(x,y) = avg(g_fix.state, H.nssq(Geo1cell(x,y)), g_fix.state) - pow(obs.ns(x,y),2);
			obs.spectrum[x][y] = g_fix.state.entanglementSpectrumLoc(Geo1cell(x,y));
		}
		
		for (size_t x=0; x<L-1; ++x)
		for (size_t y=0; y<Ly; ++y)
		{
			obs.finite_entropy(x,y) = g_fix.state.entropy()(Geo1cell(x,y%Ly));
		}
		lout << "bipartition entropy=" << obs.finite_entropy(L/2,Ly/2) << endl;
		
		if (CALC_TSQ)
		{
			double Tsq = 0.;
			double Ssq = 0.;
			#pragma omp parallel for reduction(+:Ssq) reduction(+:Tsq)
			for (int i=0; i<volume; ++i)
			for (int j=0; j<=i; ++j)
			{
				double symfactor = (i==j)? 1.:2.;
				#ifdef USING_SO4
				{
					Tsq += symfactor * avg(g_fix.state, H.TdagT(i,j), g_fix.state);
				}
				#else
				{
					Tsq += symfactor * 0.5 * avg(g_fix.state, H.TpTm(i,j), g_fix.state);
					Tsq += symfactor * 0.5 * avg(g_fix.state, H.TmTp(i,j), g_fix.state);
					Tsq += symfactor *       avg(g_fix.state, H.TzTz(i,j), g_fix.state);
				}
				#endif
				
				#ifdef USING_U0
				Ssq += symfactor * (avg(g_fix.state, H.SpSm(i,j), g_fix.state) + avg(g_fix.state, H.SzSz(i,j), g_fix.state));
				#else
				Ssq += symfactor * avg(g_fix.state, H.SdagS(i,j), g_fix.state);
				#endif
			}
			obs.Tsq = Tsq;
			obs.Ssq = Ssq;
			lout << "Tsq=" << obs.Tsq << ", Ssq=" << obs.Ssq << endl;
		}
		if (CALC_BOW)
		{
			double BOW = 0.;
			double CDW = 0.;
			double SDW = 0.;
			
			int Nbonds = (PBC)? volume:volume-1;
			VectorXd BOWloc(Nbonds);
			#pragma omp parallel for
			for (int i=0; i<Nbonds; ++i)
			{
				#ifdef USING_SO4
				{
					BOWloc(i) = avg(g_fix.state, H.cdagc(i,(i+1)%L), g_fix.state);
				}
				#elif defined(USING_U0)
					BOWloc(i) = avg(g_fix.state, H.cdagc<UP>(i,(i+1)%L), g_fix.state)+
					            avg(g_fix.state, H.cdagc<DN>(i,(i+1)%L), g_fix.state)+
					            avg(g_fix.state, H.cdagc<UP>((i+1)%L,i), g_fix.state)+
					            avg(g_fix.state, H.cdagc<DN>((i+1)%L,i), g_fix.state);
				#else
				{
					BOWloc(i) = avg(g_fix.state, H.cdagc(i,(i+1)%L), g_fix.state) 
					           +avg(g_fix.state, H.cdagc((i+1)%L,i), g_fix.state);
				}
				#endif
			}
			for (int i=0; i<Nbonds-1; ++i)
			{
				obs.opBOW(i,0) = BOWloc(i)-BOWloc((i+1)%L);
			}
			lout << "BOWloc=" << endl << BOWloc << endl;
			lout << "obs.opBOW=" << endl << obs.opBOW << endl;
			
			#pragma omp parallel for reduction(+:BOW)
			for (int i=0; i<volume-1; ++i)
			for (int j=0; j<=i; ++j)
			{
				double symfactor = (i==j)? 1.:2.;
				#ifdef USING_SO4
				{
					BOW += symfactor * pow(-1.,i+j) * (
					                         avg(g_fix.state, H.cdagc(i,i+1), H.cdagc(j,j+1), g_fix.state)
					                       - pow(BOWloc(i),2)
					                      );
					CDW += symfactor * pow(-1.,i+j) * avg(g_fix.state, H.TdagT(i,j), g_fix.state);
					SDW += symfactor * pow(-1.,i+j) * avg(g_fix.state, H.SdagS(i,j), g_fix.state);
				}
				#elif defined(USING_U0)
				{
					BOW += 0; // not implemented
				}
				#else
				{
					BOW += symfactor * pow(-1.,i+j) * (
					                         avg(g_fix.state, H.cdagc(i,i+1), H.cdagc(j,j+1), g_fix.state)
					                       + avg(g_fix.state, H.cdagc(i,i+1), H.cdagc(j+1,j), g_fix.state)
					                       + avg(g_fix.state, H.cdagc(i+1,i), H.cdagc(j,j+1), g_fix.state)
					                       + avg(g_fix.state, H.cdagc(i+1,i), H.cdagc(j+1,j), g_fix.state)
					                       - pow(BOWloc(i),2)
					                      );
				}
				#endif
			}
			obs.BOW = BOW/volume;
			obs.CDW = CDW/volume;
			obs.SDW = SDW/volume;
			lout << "BOW=" << obs.BOW << endl;
			lout << "CDW=" << obs.CDW << endl;
			lout << "SDW=" << obs.SDW << endl;
		}
		
		lout << "ns=" << obs.ns.sum()/volume << endl;
		lout << "nh=" << obs.nh.sum()/volume << endl;
		
		HDF5Interface target;
		std::stringstream bond;
		target = HDF5Interface(obsfile,WRITE);
		bond << g_fix.state.calc_fullMmax();
		target.create_group(bond.str());
		
		target.save_scalar(obs.energy,"energy",bond.str());
		target.save_scalar(obs.energyT,"energyT",bond.str());
		target.save_scalar(obs.dedV,"dedV",bond.str());
		target.save_scalar(g_fix.state.calc_Dmax(),"Dmax",bond.str());
		target.save_scalar(g_fix.state.calc_Mmax(),"Mmax",bond.str());
		target.save_scalar(g_fix.state.calc_fullMmax(),"full Mmax",bond.str());
		target.save_scalar(Fix.get_errEigval(),"err_eigval",bond.str());
		target.save_scalar(Fix.get_errState(),"err_state",bond.str());
		
		target.save_matrix(obs.nh,"nh",bond.str());
		target.save_matrix(obs.ns,"ns",bond.str());
		for (size_t x=0; x<L; ++x)
		for (size_t y=0; y<Ly; ++y)
		{
			target.save_matrix(obs.spectrum[x][y],make_string("spectrum_x=",x,"_y=",y),bond.str());
		}
		target.save_matrix(obs.opBOW,"opBOW",bond.str());
		target.save_matrix(obs.finite_entropy,"finite_entropy",bond.str());
		target.save_scalar(obs.Tsq,"Tsq",bond.str());
		target.save_scalar(obs.Ssq,"Ssq",bond.str());
		target.save_scalar(obs.BOW,"BOW",bond.str());
		target.save_scalar(obs.CDW,"CDW",bond.str());
		target.save_scalar(obs.SDW,"SDW",bond.str());
		target.close();
		
		lout << ObsWatch.info("observables") << endl;
		lout << "saved to: " << obsfile << endl;
	}
	
	lout << Watch.info("total time") << endl;
	lout << "emin=" << obs.energy << ", e_empty=" << e_empty() << endl;
	
//	size_t Nmax = (VUMPS)? Ncells:1;
//	Geometry2D GeoNcell(SNAKE,Nmax*L,Ly,1.,true);
//	for (size_t l=0; l<L*Ly*Nmax; ++l)
//	{
//		MODEL Htmp(L*Ly*Nmax+1,{});
//		
//		double SdagS = avg(g_foxy.state, Htmp.SdagS(0,l), g_foxy.state);
//		double TdagT = avg(g_foxy.state, Htmp.TdagT(0,l), g_foxy.state);
//		
//		cout << "x=" << GeoNcell(l).first << ", y=" << GeoNcell(l).second << ", <S†S>=" << SdagS << ", <T†T>=" << TdagT << endl;
//	}
}

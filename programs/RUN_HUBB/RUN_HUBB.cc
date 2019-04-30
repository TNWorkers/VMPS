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

#include "Geometry2D.h" // from TOOLS
#include "NestedLoopIterator.h" // from TOOLS

size_t L, N_cell, Ly;
int volume;
double t, tRung, U, J, V, dV;
int M, N, S, T;
double alpha;
DMRG::VERBOSITY::OPTION VERB;
double Emin = 0.;
double emin = 0.;
bool UMPS_STRUCTURE_FACTOR, VUMPS;
string wd; // working directory

Eigenstate<MODEL::StateXd> g_fix;
Eigenstate<MODEL::StateUd> g_foxy;

struct Obs
{
	Eigen::MatrixXd nh;
	Eigen::MatrixXd ns;
	Eigen::MatrixXd entropy;
	double energy;
	double dedV;
	double entropy_bipart; // for finite systems
	double Tsq; // T^2
	double Ssq; // S^2
	
	double S_M;
	double T_M;
	double S_Gamma;
	double T_Gamma;
	
	Eigen::MatrixXd SdagS;
	Eigen::MatrixXd TdagT;
	
	void resize (size_t Lx, size_t Ly, size_t Lobs)
	{
		nh.resize(Lx,Ly); nh.setZero();
		ns.resize(Lx,Ly); ns.setZero();
		
		entropy.resize(Lx,Ly); entropy.setZero();
		
		// format:
		// n x0 y0 x1 y1 value
		SdagS.resize(N_cell*Lx*Ly*Lx*Ly,6); SdagS.setZero();
		TdagT.resize(N_cell*Lx*Ly*Lx*Ly,6); TdagT.setZero();
	}
};

Obs obs;

vector<vector<vector<vector<ArrayXd> > > > SdagS;
vector<vector<vector<vector<ArrayXd> > > > TdagT;

void fill_OdagO (size_t L, size_t Ly, size_t N_cell, const Eigenstate<MODEL::StateUd> &g)
{
	MODEL Htmp(2*N_cell*L*Ly+4,{{"OPEN_BC",false},{"CALC_SQUARE",false}});
	Geometry2D Geo(SNAKE,L,Ly,1.,true);
	
	SdagS.resize(L);
	TdagT.resize(L);
	for (size_t x0=0; x0<L; ++x0)
	{
		SdagS[x0].resize(Ly);
		TdagT[x0].resize(Ly);
		for (size_t y0=0; y0<Ly; ++y0)
		{
			SdagS[x0][y0].resize(L);
			TdagT[x0][y0].resize(L);
			for (size_t x1=0; x1<L; ++x1)
			{
				SdagS[x0][y0][x1].resize(Ly);
				TdagT[x0][y0][x1].resize(Ly);
				for (size_t y1=0; y1<Ly; ++y1)
				{
					SdagS[x0][y0][x1][y1].resize(N_cell);
					TdagT[x0][y0][x1][y1].resize(N_cell);
				}
			}
		}
	}
	
	for (size_t n=0; n<N_cell; ++n)
	#pragma omp parallel for collapse(4)
	for (size_t x0=0; x0<L; ++x0)
	for (size_t x1=0; x1<L; ++x1)
	for (size_t y0=0; y0<Ly; ++y0)
	for (size_t y1=0; y1<Ly; ++y1)
	{
		int i0 = Geo(x0,y0);
		int i1 = Geo(x1,y1);
		
		SdagS[x0][y0][x1][y1](n) = avg(g.state, Htmp.SdagS(i0,L*Ly*n+i1), g.state);
		TdagT[x0][y0][x1][y1](n) = avg(g.state, Htmp.TdagT(i0,L*Ly*n+i1), g.state);
	}
	
	// save to obs
	NestedLoopIterator Nelly(5,{N_cell,L,Ly,L,Ly});
	for (Nelly=Nelly.begin(); Nelly!=Nelly.end(); ++Nelly)
	{
		int n  = Nelly(0);
		int x0 = Nelly(1);
		int y0 = Nelly(2);
		int x1 = Nelly(3);
		int y1 = Nelly(4);
		
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
	}
}

complex<double> calc_FT (double kx, int iky, const vector<vector<vector<vector<ArrayXd> > > > &OdagO)
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
	N_cell = args.get<size_t>("Lobs",40); // Amount of unit cells for explicit contraction
	t = args.get<double>("t",1.);
	tRung = args.get<double>("tRung",t); // tRung != t for testing only
	U = args.get<double>("U",8.);
	J = args.get<double>("J",0.);
	V = args.get<double>("V",0.);
	dV = args.get<double>("dV",0.01);
	M = args.get<int>("M",0);
	S = abs(M)+1;
	UMPS_STRUCTURE_FACTOR = args.get<bool>("STRUCTURE",true);
	VUMPS = args.get<bool>("VUMPS",true);
	
	DMRG::CONTROL::GLOB GlobParam_fix;
	DMRG::CONTROL::DYN  DynParam_fix;
	VUMPS::CONTROL::GLOB GlobParam_foxy;
	VUMPS::CONTROL::DYN  DynParam_foxy;
	
	size_t min_Nsv = args.get<size_t>("min_Nsv",0ul);
	DynParam_fix.min_Nsv = [min_Nsv] (size_t i) {return min_Nsv;};
	alpha = args.get<double>("alpha",100.);
	
	VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",2));
	
	string base = make_string("U=",U,"_V=",V,"_J=",J,"_Ly=",Ly);
	string obsfile = make_string(wd,"obs/",base,".h5");
	string statefile = make_string(wd,"state/",base);
	
	GlobParam_fix.Dinit  = args.get<size_t>("Dinit",2ul);
	GlobParam_fix.Dlimit = args.get<size_t>("Dlimit",200ul);
	GlobParam_fix.Qinit = args.get<size_t>("Qinit",10ul);
	GlobParam_fix.min_halfsweeps = args.get<size_t>("Imin",1);
	GlobParam_fix.max_halfsweeps = args.get<size_t>("Imax",22);
	GlobParam_fix.tol_eigval = args.get<double>("tol_eigval",1e-6);
	GlobParam_fix.tol_state = args.get<double>("tol_state",1e-5);
	GlobParam_fix.savePeriod = args.get<size_t>("savePeriod",4ul);
	GlobParam_fix.saveName = args.get<string>("saveName",statefile);
	
	GlobParam_foxy.Dinit  = args.get<size_t>("Dinit",10ul);
	GlobParam_foxy.Dlimit = args.get<size_t>("Dlimit",200ul);
	GlobParam_foxy.Qinit = args.get<size_t>("Qinit",10ul);
	GlobParam_foxy.min_iterations = args.get<size_t>("Imin",6);
	GlobParam_foxy.max_iterations = args.get<size_t>("Imax",1000);
	GlobParam_foxy.max_iter_without_expansion = args.get<size_t>("max",30);
	
	GlobParam_foxy.tol_eigval = args.get<double>("tol_eigval",1e-6);
	GlobParam_foxy.tol_var = args.get<double>("tol_var",1e-6);
	GlobParam_foxy.tol_state = args.get<double>("tol_state",1e-5);
	GlobParam_foxy.savePeriod = args.get<size_t>("savePeriod",4ul);
	
	wd = args.get<string>("wd","./");
	correct_foldername(wd);
	
	lout << args.info() << endl;
	lout << "wd=" << wd << endl;
	
	lout.set(make_string("L=",L,"_Ly=",Ly,"_t=",t,"_V=",V,"_U=",U,"_J=",J,".log"),wd+"log");
	
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	Stopwatch<> Watch;
	
	Geometry2D Geo1cell(SNAKE,1*L,Ly,1.,true); // periodic BC in y = true
	Geometry2D Geo2cell(SNAKE,2*L,Ly,1.,true);
	
	// save to temporary, otherwise std::bad_any_cast
	ArrayXXd tArray, Varray, dVarray, VxyArray, VzArray, dVxyArray, dVzArray, Jarray, ZeroArray, OneArray;
	if (VUMPS)
	{
		tArray    = t * Geo2cell.hopping();
		Varray    = V * Geo2cell.hopping();
		dVarray   = (V+dV) * Geo2cell.hopping();
		Jarray    = J * Geo2cell.hopping();
		ZeroArray = 0. * Geo2cell.hopping();
		OneArray  = 1. * Geo2cell.hopping();
		lout << "t=" << t << ", V=" << V << ", dV=" << dV << ", J=" << J << endl;
		lout << "hopping=" << endl << Geo2cell.hopping() << endl << endl;
	}
	else
	{
		ArrayXXd tFull = Geo1cell.hopping();
		tArray    = t * tFull;
		Varray    = V * tFull;
		dVarray   = (V+dV) * tFull;
		Jarray    = J * tFull;
		ZeroArray = 0. * tFull;
		OneArray  = 1. * tFull;
		lout << "hopping=" << endl << tFull << endl << endl;
	}
	
	vector<Param> params;
	qarray<2> Qc;
	if constexpr (std::is_same<MODEL,VMPS::HubbardSU2xSU2>::value)
	{
		params.push_back({"tFull",tArray});
		params.push_back({"Vfull",Varray});
		params.push_back({"Jfull",Jarray});
		params.push_back({"U",U});
		if (VUMPS) {params.push_back({"OPEN_BC",false});}
		Qc = {S,T};
	}
	else
	{
		params.push_back({"tFull",tArray});
		params.push_back({"Vzfull",Varray});
		params.push_back({"Vxyfull",Varray});
		params.push_back({"Jfull",Jarray});
		params.push_back({"Uph",U});
		params.push_back({"U",0.});
		if (VUMPS) {params.push_back({"OPEN_BC",false});}
		Qc = {S,N};
	}
	
	MODEL H(volume,params);
	if (VUMPS) H.transform_base(Qc);
	lout << H.info() << endl;
	
	// To calculate de/dV for finite systems as exp. value of Hamiltonian with only V=1
	vector<Param> dHdV_params;
	dHdV_params.push_back({"tFull",ZeroArray});
	dHdV_params.push_back({"Jfull",ZeroArray});
	dHdV_params.push_back({"U",0.});
	dHdV_params.push_back({"Uph",0.});
	if constexpr (std::is_same<MODEL,VMPS::HubbardSU2xSU2>::value)
	{
		dHdV_params.push_back({"Vfull",OneArray});
	}
	else
	{
		dHdV_params.push_back({"Vxyfull",OneArray});
		dHdV_params.push_back({"Vzfull",OneArray});
	}
	
	MODEL dHdV;
	if (VUMPS)
	{
		// For VUMPS: Hamiltonian with two unit cells for contractions across the cell
		dHdV = MODEL(2*volume,{{"OPEN_BC",false},{"CALC_SQUARE",false}});
		dHdV.transform_base(Qc);
	}
	else
	{
		dHdV = MODEL(volume,dHdV_params);
	}
	lout << dHdV.info() << endl;
	
	obs.resize(L,Ly,N_cell);
	
	if (VUMPS)
	{
		MODEL::uSolver Foxy(VERB);
		HDF5Interface target;
		cout << obsfile << endl;
		target = HDF5Interface(obsfile,WRITE);
		target.close();
		
		auto measure_and_save = [&H,&dHdV,&target,&params,&Geo1cell,&Geo2cell,&Foxy,&obsfile](size_t j) -> void
		{
			if (Foxy.errVar() < 1e-8 or Foxy.FORCE_DO_SOMETHING == true)
			{
				std::stringstream bond;
				target = HDF5Interface(obsfile,REWRITE);
				bond << g_foxy.state.calc_fullMmax();
				cout << termcolor::red << "Measure at M=" << bond.str() << ", if possible" << termcolor::reset << endl;
				
				if (target.HAS_GROUP(bond.str())) {return;}
				
				Stopwatch<> SaveAndMeasure;
				for (size_t x=0; x<L; ++x)
				for (size_t y=0; y<Ly; ++y)
				{
					obs.nh(x,y)      = avg(g_foxy.state, H.nh(Geo1cell(x,y)), g_foxy.state);
					obs.ns(x,y)      = avg(g_foxy.state, H.ns(Geo1cell(x,y)), g_foxy.state);
					obs.entropy(x,y) = g_foxy.state.entropy()(Geo1cell(x,y));
				}
				
				obs.energy = g_foxy.energy;
				
				//----------de/dV----------
				obs.dedV = 0;
				for (size_t x=0; x<L;  ++x)
				for (size_t y=0; y<Ly; ++y)
				{
					// horizontal
					//if constexpr (std::is_same<MODEL,VMPS::HubbardSU2xSU2>::value)
					#ifdef USING_SO4
					{
						obs.dedV += avg(g_foxy.state, dHdV.TdagT(Geo2cell(x,y),Geo2cell(x+1,y)), g_foxy.state);
					}
					//else
					#else
					{
						obs.dedV += 0.5 * avg(g_foxy.state, dHdV.TpTm(Geo2cell(x,y),Geo2cell(x+1,y)), g_foxy.state);
						obs.dedV += 0.5 * avg(g_foxy.state, dHdV.TmTp(Geo2cell(x,y),Geo2cell(x+1,y)), g_foxy.state);
						obs.dedV += avg(g_foxy.state, dHdV.TzTz(Geo2cell(x,y),Geo2cell(x+1,y)), g_foxy.state);
					}
					#endif
					
					// vertical
					if (Ly > 1)
					{
						double edge_corr = (Ly==2)? 0.5:1.;
//						if constexpr (std::is_same<MODEL,VMPS::HubbardSU2xSU2>::value)
						#ifdef USING_SO4
						{
							obs.dedV += edge_corr * avg(g_foxy.state, dHdV.TdagT(Geo2cell(x,y),Geo2cell(x,(y+1)%L)), g_foxy.state);
						}
//						else
						#else
						{
							obs.dedV += edge_corr * 0.5 * avg(g_foxy.state, dHdV.TpTm(Geo2cell(x,y),Geo2cell(x,(y+1)%L)), g_foxy.state);
							obs.dedV += edge_corr * 0.5 * avg(g_foxy.state, dHdV.TmTp(Geo2cell(x,y),Geo2cell(x,(y+1)%L)), g_foxy.state);
							obs.dedV += edge_corr * avg(g_foxy.state, dHdV.TzTz(Geo2cell(x,y),Geo2cell(x,(y+1)%L)), g_foxy.state);
						}
						#endif
					}
				}
				lout << "done!" << endl;
				obs.dedV /= (L*Ly);
				//----------de/dV----------
				
				//----------k-points to calculate----------
				vector<pair<double,int> > kxy;
				if (Ly == 1)
				{
					kxy.push_back(make_pair(0.,0));
					kxy.push_back(make_pair(M_PI,0));
				}
				else
				{
					kxy.push_back(make_pair(0.,0)); // Γ point (0,0)
					kxy.push_back(make_pair(M_PI,Ly/2)); // M point (π,π)
				}
				
				if (UMPS_STRUCTURE_FACTOR)
				{
					for (int i=0; i<kxy.size(); ++i)
					{
						double kx = kxy[i].first;
						int   iky = kxy[i].second;
						
						vector<Mpo<MODEL::Symmetry,complex<double> > > Sdag_ky(L);
						vector<Mpo<MODEL::Symmetry,complex<double> > > S_ky   (L);
						vector<Mpo<MODEL::Symmetry,complex<double> > > Tdag_ky(L);
						vector<Mpo<MODEL::Symmetry,complex<double> > > T_ky   (L);
						
						// Fourier transform of operators in y-direction
						for (size_t x=0; x<L; ++x)
						{
							vector<complex<double> > phases_p = Geo1cell.FTy_phases(x,iky,0);
							vector<complex<double> > phases_m = Geo1cell.FTy_phases(x,iky,1);
							
							Sdag_ky[x] = H.Sdag_ky(phases_m);
							S_ky[x]    = H.S_ky   (phases_p);
							Tdag_ky[x] = H.Tdag_ky(phases_m);
							T_ky[x]    = H.T_ky   (phases_p);
						}
						
						
						// Calculate expectation values within the cell
						ArrayXXcd Sij_cell(L,L); Sij_cell = 0;
						ArrayXXcd Tij_cell(L,L); Tij_cell = 0;
						
						for (size_t x1=0; x1<L; ++x1)
						for (size_t x2=0; x2<L; ++x2)
						{
							auto phases_m1 = Geo1cell.FTy_phases(x1,iky,1);
							auto phases_p2 = Geo1cell.FTy_phases(x2,iky,0);
							
							for (size_t y1=0; y1<Ly; ++y1)
							for (size_t y2=0; y2<Ly; ++y2)
							{
								int index1 = Geo1cell(x1,y1);
								int index2 = Geo1cell(x2,y2);
								
								if (phases_m1[index1] * phases_p2[index2] != 0.)
								{
									Sij_cell(x1,x2) += phases_m1[index1] * phases_p2[index2] * 
									                   avg(g_foxy.state, H.SdagS(index1,index2), g_foxy.state);
									Tij_cell(x1,x2) += phases_m1[index1] * phases_p2[index2] * 
									                   avg(g_foxy.state, H.TdagT(index1,index2), g_foxy.state);
								}
							}
						}
						
						// Calculate full structure factor
						complex<double> SF_S, SF_T;
						#pragma omp parallel sections
						{
							#pragma omp section
							{
								SF_S = g_foxy.state.SFpoint(Sij_cell, Sdag_ky,S_ky, L, kx, DMRG::VERBOSITY::ON_EXIT);
							}
							#pragma omp section
							{
								SF_T = g_foxy.state.SFpoint(Tij_cell, Tdag_ky,T_ky, L, kx, DMRG::VERBOSITY::ON_EXIT);
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
							lout << termcolor::red << "S_Γ=" << SF_S << ", T_Γ=" << SF_T << termcolor::reset << endl;
						}
						// M point
						else if (kx == M_PI and iky == Ly/2)
						{
							obs.S_M = isReal(SF_S);
							obs.T_M = isReal(SF_T);
							lout << termcolor::red << "S_M=" << SF_S << ", T_M=" << SF_T << termcolor::reset << endl;
						}
					}
				}
				else
				{
					fill_OdagO(L, Ly, N_cell, g_foxy);
					
					for (int i=0; i<kxy.size(); ++i)
					{
						double kx = kxy[i].first;
						int iky   = kxy[i].second;
						
						complex<double> SF_S = calc_FT(kx,iky,SdagS);
						complex<double> SF_T = calc_FT(kx,iky,TdagT);
						
						// Γ point
						if (kx == 0. and iky == 0)
						{
							obs.S_Gamma = isReal(SF_S);
							obs.T_Gamma = isReal(SF_T);
							
							lout << termcolor::red << "S_Γ=" << SF_S << ", T_Γ=" << SF_T << termcolor::reset << endl;
						}
						// M point
						else if (kx == M_PI and iky == Ly/2)
						{
							obs.S_M = isReal(SF_S);
							obs.T_M = isReal(SF_T);
							
							lout << termcolor::red << "S_M=" << SF_S << ", T_M=" << SF_T << termcolor::reset << endl;
						}
					}
				}
				
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
				target.save_matrix(obs.entropy,"Entropy",bond.str());
				
				if (UMPS_STRUCTURE_FACTOR)
				{
					target.save_scalar(obs.S_Gamma,"S_Gamma",bond.str());
					target.save_scalar(obs.T_Gamma,"T_Gamma",bond.str());
					target.save_scalar(obs.S_M,"S_M",bond.str());
					target.save_scalar(obs.T_M,"T_M",bond.str());
				}
				else
				{
					target.save_matrix(obs.SdagS,"SiSj",bond.str());
					target.save_matrix(obs.TdagT,"TiTj",bond.str());
				}
				target.close();
				stringstream ss;
				ss << "Calcuated and saved observables for M=" << g_foxy.state.calc_fullMmax();
				
				if (Foxy.get_verbosity() >= DMRG::VERBOSITY::HALFSWEEPWISE) {lout << SaveAndMeasure.info(ss.str()) << endl << endl;}
			}
		};
		
		Foxy.userSetGlobParam();
		Foxy.userSetDynParam();
		Foxy.GlobParam = GlobParam_foxy;
		Foxy.DynParam = DynParam_foxy;
		Foxy.DynParam.doSomething = measure_and_save;
		Foxy.DynParam.iteration = [](size_t i) -> UMPS_ALG::OPTION {return UMPS_ALG::PARALLEL;};
		Foxy.set_log(2,"e0.dat","err_eigval.dat","err_var.dat","err_state.dat");
		Foxy.edgeState(H, g_foxy, Qc);
		
		emin = g_foxy.energy;
	}
	else
	{
		MODEL::Solver Fix(VERB);
		Fix.userSetGlobParam();
		Fix.userSetDynParam();
		Fix.GlobParam = GlobParam_fix;
		Fix.DynParam = DynParam_fix;
		Fix.edgeState(H, g_fix, Qc, LANCZOS::EDGE::GROUND); 
		
		// U-correction of energy: unnecessery with particle-hole symmetric Uph
//		if constexpr (std::is_same<MODEL,VMPS::HubbardSU2xSU2>::value)
//		{
//			Emin = g_fix.energy-0.5*U*(volume-N);
//		}
//		else
//		{
//			Emin = g_fix.energy;
//		}
		Emin = g_fix.energy;
		emin = Emin/volume;
		
		obs.energy = emin;
		obs.dedV = avg(g_fix.state, dHdV, g_fix.state)/volume;
		
		#pragma omp parallel for collapse(2)
		for (size_t x=0; x<L; ++x)
		for (size_t y=0; y<Ly; ++y)
		{
			obs.nh(x,y) = avg(g_fix.state, H.nh(Geo1cell(x,y)), g_fix.state);
			obs.ns(x,y) = avg(g_fix.state, H.ns(Geo1cell(x,y)), g_fix.state);
		}
		obs.entropy_bipart = g_fix.state.entropy()(volume/2);
		lout << "bipartition entropy=" << obs.entropy_bipart << endl;
		
		double Tsq = 0.;
		double Ssq = 0.;
		#pragma omp parallel for collapse(2) reduction(+:Ssq) reduction(+:Tsq)
		for (int i=0; i<volume; ++i)
		for (int j=0; j<volume; ++j)
		{
			//if constexpr (std::is_same<MODEL,VMPS::HubbardSU2xSU2>::value)
			#ifdef USING_SO4
			{
				Tsq += avg(g_fix.state, H.TdagT(i,j), g_fix.state);
			}
//			else
			#else
			{
				Tsq += 0.5 * avg(g_fix.state, H.TpTm(i,j), g_fix.state);
				Tsq += 0.5 * avg(g_fix.state, H.TmTp(i,j), g_fix.state);
				Tsq +=       avg(g_fix.state, H.TzTz(i,j), g_fix.state);
			}
			#endif
			
			Ssq += avg(g_fix.state, H.SdagS(i,j), g_fix.state);
		}
		obs.Tsq = Tsq;
		obs.Ssq = Ssq;
		lout << "Tsq=" << obs.Tsq  << ", Ssq=" << obs.Ssq << endl;
		lout << "ns=" << obs.ns.sum()/volume << endl;
		lout << "nh=" << obs.nh.sum()/volume << endl;
		
		HDF5Interface target;
		std::stringstream bond;
		target = HDF5Interface(obsfile,WRITE);
		bond << g_fix.state.calc_fullMmax();
		target.create_group(bond.str());
		
		target.save_scalar(obs.energy,"energy",bond.str());
		target.save_scalar(obs.dedV,"dedV",bond.str());
		target.save_scalar(g_fix.state.calc_Dmax(),"Dmax",bond.str());
		target.save_scalar(g_fix.state.calc_Mmax(),"Mmax",bond.str());
		target.save_scalar(g_fix.state.calc_fullMmax(),"full Mmax",bond.str());
		target.save_scalar(Fix.get_errEigval(),"err_eigval",bond.str());
		target.save_scalar(Fix.get_errState(),"err_state",bond.str());
		
		target.save_matrix(obs.nh,"nh",bond.str());
		target.save_matrix(obs.ns,"ns",bond.str());
		target.save_scalar(obs.entropy_bipart,"entropy_bipart",bond.str());
		target.save_scalar(obs.Tsq,"Tsq",bond.str());
		target.save_scalar(obs.Ssq,"Ssq",bond.str());
		target.close();
	}
	
	lout << Watch.info("total time") << endl;
	lout << "emin=" << emin << endl;
	
//	size_t Nmax = (VUMPS)? N_cell:1;
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

#define LANCZOS_MAX_ITERATIONS 1e2

#define USE_HDF5_STORAGE
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
#include <unsupported/Eigen/FFT>

#include "VUMPS/VumpsSolver.h"
#include "VUMPS/VumpsLinearAlgebra.h"
#include "DmrgLinearAlgebra.h"
#include "solvers/TDVPPropagator.h"
#include "solvers/EntropyObserver.h"
#include "models/SpinlessFermionsU1.h"
#include "models/SpinlessFermionsZ2.h"
#include "models/SpinlessFermions.h"
#include "models/HubbardSU2xU1.h"
#include "models/HubbardU1xU1.h"

#include "IntervalIterator.h"
#include "Quadrator.h"
#define CHEBTRANS_DONT_USE_FFTWOMP
#include "SuperQuadrator.h"
#define GREENPROPAGATOR_USE_HDF5
#include "solvers/GreenPropagator.h"
#include "RootFinder.h" // from ALGS

size_t L, Ncells, Lhetero, x0;
int M, Dtot, N;
double VA, VB;
double tp, tA, tB, tx;
double tmax, dt, tol_compr;
int Nt;
size_t Chi, max_iter, min_iter, Qinit, D;
double tol_eigval, tol_var, tol_state;
bool GAUSSINT, RELOAD;
double wmin, wmax;
string wd;

// set model here:
////1.
//typedef VMPS::SpinlessFermionsU1 MODEL; double spinfac=1.;
//typedef VMPS::SpinlessFermionsZ2 MODEL; double spinfac=1.;
//typedef VMPS::SpinlessFermions MODEL; double spinfac=1.;
//#define USE_SPINLESS
////2.
typedef VMPS::HubbardSU2xU1 MODEL; double spinfac=1.;
#define USE_SU2XU1
////3.
//typedef VMPS::HubbardU1xU1 MODEL; double spinfac=2.;
//#define USE_U1XU1

typedef MODEL::Symmetry Symmetry;

std::array<GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double>>,2> Green;
GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double>> Gfull;

ArrayXd ncell;

double n_mu (double mu, void*)
{
	return spinfac * Gfull.integrate_Glocw_cell(mu) - ncell.sum();
}

int main (int argc, char* argv[])
{
	omp_set_nested(1);
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",2);
	Ncells = args.get<size_t>("Ncells",10);
	
	wd = args.get<string>("wd","./");
	if (wd.back() != '/') {wd += "/";}
	
	Lhetero = L*Ncells;
	x0 = Lhetero/2;
	#ifdef USE_SPINLESS
	{
		N = args.get<int>("N",L/2);
	}
	#else
	{
		N = args.get<int>("N",L);
	}
	#endif
	GAUSSINT = static_cast<bool>(args.get<int>("GAUSSINT",true));
	string INIT = args.get<string>("INIT","");
	
	RELOAD = static_cast<bool>(args.get<int>("RELOAD",false));
	wmin = args.get<double>("wmin",-10.);
	wmax = args.get<double>("wmax",+10.);
	double wshift = args.get<double>("wshift",0.);
	
	dt = args.get<double>("dt",0.1);
	tmax = args.get<double>("tmax",4.);
	Nt = static_cast<int>(tmax/dt);
	tol_compr =  args.get<double>("tol_compr",1e-5);
	
	Chi = args.get<size_t>("Chi",2);
	tol_eigval = args.get<double>("tol_eigval",1e-5);
	tol_var = args.get<double>("tol_var",1e-5);
	tol_state = args.get<double>("tol_state",1e-4);
	
	max_iter = args.get<size_t>("max_iter",50ul);
	min_iter = args.get<size_t>("min_iter",100ul);
	Qinit = args.get<size_t>("Qinit",6ul);
	
	VUMPS::CONTROL::GLOB GlobParams;
	GlobParams.min_iterations = min_iter;
	GlobParams.max_iterations = max_iter;
	GlobParams.Dinit = Chi;
	GlobParams.Dlimit = Chi;
	GlobParams.Qinit = Qinit;
	GlobParams.tol_eigval = tol_eigval;
	GlobParams.tol_var = tol_var;
	GlobParams.tol_state = tol_state;
	GlobParams.max_iter_without_expansion = 30ul;
	
	lout << args.info() << endl;
	string base = make_string("Lcell=",L,"_t=1_U=0");
	if (!RELOAD) lout.set(base+".log",wd+"log");
	
	// reload:
	if (RELOAD)
	{
		vector<vector<MatrixXcd>> GinPES(L); for (int i=0; i<L; ++i) {GinPES[i].resize(L);}
		vector<vector<MatrixXcd>> GinIPE(L); for (int i=0; i<L; ++i) {GinIPE[i].resize(L);}
		vector<vector<MatrixXcd>> GinA1P(L); for (int i=0; i<L; ++i) {GinA1P[i].resize(L);}
		for (int i=0; i<L; ++i) 
		for (int j=0; j<L; ++j)
		{
			GinPES[i][j].resize(Nt,Ncells);
			GinPES[i][j].setZero();
			GinIPE[i][j].resize(Nt,Ncells);
			GinIPE[i][j].setZero();
			GinA1P[i][j].resize(Nt,Ncells);
			GinA1P[i][j].setZero();
		}
		
		for (int i=0; i<L; ++i)
		for (int j=0; j<L; ++j)
		{
			MatrixXd MtmpRe(Nt,Ncells);
			MatrixXd MtmpIm(Nt,Ncells);
			
			HDF5Interface ReaderPES(wd+"PES_"+base+make_string("_L=",Lhetero,"_tmax=",tmax)+".h5",READ);
			ReaderPES.load_matrix(MtmpRe,"txRe",make_string("G",i,j));
			ReaderPES.load_matrix(MtmpIm,"txIm",make_string("G",i,j));
			ReaderPES.close();
			
			GinPES[i][j] += MtmpRe+1.i*MtmpIm;
			
			HDF5Interface ReaderIPE(wd+"IPE_"+base+make_string("_L=",Lhetero,"_tmax=",tmax)+".h5",READ);
			ReaderIPE.load_matrix(MtmpRe,"txRe",make_string("G",i,j));
			ReaderIPE.load_matrix(MtmpIm,"txIm",make_string("G",i,j));
			ReaderIPE.close();
			
			GinIPE[i][j] += MtmpRe+1.i*MtmpIm;
			
			GinA1P[i][j] += GinPES[i][j]+GinIPE[i][j];
		}
		
		GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double>> 
		GrecalcPES = GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> >(wd+"PESr_"+base,tmax,GinPES,ZERO_2PI,500,true); // GAUSSINT=true
		GrecalcPES.recalc_FTwCell(wmin,wmax,501,wshift);
		GrecalcPES.FT_allSites(wshift);
		
		GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double>>
		GrecalcIPE = GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> >(wd+"IPEr_"+base,tmax,GinIPE,ZERO_2PI,500,true); // GAUSSINT=true
		GrecalcIPE.recalc_FTwCell(wmin,wmax,501,wshift);
		GrecalcIPE.FT_allSites(-wshift);
		
		HDF5Interface ReaderPES(wd+"PES_"+base+make_string("_L=",L,"_tmax=",tmax)+".h5",READ);
		MatrixXd Mtmp;
		ReaderPES.load_matrix(Mtmp,"ncell");
		ncell = Mtmp;
		
		IntervalIterator mu(wmin,wmax,101);
		for (mu=mu.begin(); mu!=mu.end(); ++mu)
		{
			double res = spinfac * Gfull.integrate_Glocw_cell(*mu);
			mu << res;
		}
		mu.save(wd+"n(μ)_"+base+".dat");
		RootFinder R(n_mu,wmin,wmax);
		lout << "μ=" << R.root() << endl;
		
		Gfull.ncell = ncell;
		Gfull.mu = R.root();
		Gfull.save();
	}
	else
	{
		qarray<MODEL::Symmetry::Nq> Q = MODEL::singlet(N);
		
		vector<Param> params;
		params.push_back({"CALC_SQUARE",false});
		params.push_back({"OPEN_BC",false});
		params.push_back({"U",0.});
		
		MODEL H(L,params);
		H.transform_base(Q,true); // PRINT=true
		lout << H.info() << endl;
		
		MODEL::uSolver uDMRG(DMRG::VERBOSITY::ON_EXIT);
		Eigenstate<MODEL::StateUd> g;
		uDMRG.userSetGlobParam();
		uDMRG.GlobParam = GlobParams;
		if (INIT != "")
		{
			g.state.load(wd+"init/"+INIT);
			lout << g.state.info() << endl;
			
			HDF5Interface Reader(wd+"init/"+base+"_Sym="+Symmetry::name()+".h5",READ);
			Reader.load_scalar(g.energy,"val","energy");
			Reader.close();
			lout << termcolor::blue << "ground state loaded!" << termcolor::reset << endl;
		}
		else
		{
			uDMRG.edgeState(H,g,Q);
			
			g.state.save(wd+"init/"+base+"_Sym="+Symmetry::name());
			HDF5Interface Writer(wd+"init/"+base+"_Sym="+Symmetry::name()+".h5",REWRITE);
			Writer.create_group("energy");
			Writer.save_scalar(g.energy,"val","energy");
			Writer.close();
		}
		
		ncell.resize(L); ncell.setZero();
		for (int l=0; l<L; ++l)
		{
			ncell(l) = avg(g.state, H.n(l), g.state);
		}
		lout << "ncell=" << ncell.transpose() << ", avg=" << ncell.sum()/L << endl;
		#ifndef USE_SPINLESS
		{
			ArrayXd dcell(L);
			for (int l=0; l<L; ++l)
			{
				dcell(l) = avg(g.state, H.d(l), g.state);
			}
			lout << "dcell=" << dcell.transpose() << ", avg=" << dcell.sum()/L << endl;
		}
		#endif
		
		MODEL H_hetero(Lhetero,params);
		lout << H_hetero.info() << endl;
		H_hetero.transform_base(Q,false,L); // PRINT=false
		H_hetero.precalc_TwoSiteData(true); // FORCE=true
		
		// create vector of O
		std::array<vector<Mpo<MODEL::Symmetry,double>>,2> O;
		O[0].resize(L);
		O[1].resize(L);
		for (int l=0; l<L; ++l)
		{
			#ifdef USE_SPINLESS
			{
				O[0][l] = H_hetero.c(Lhetero/2+l);
				O[1][l] = H_hetero.cdag(Lhetero/2+l);
			}
			#elif defined(USE_U1XU1)
			{
				O[0][l] = H_hetero.c<UP>(Lhetero/2+l);
				O[1][l] = H_hetero.cdag<UP>(Lhetero/2+l);
//				O[0][l] = H_hetero.Scomp(SP,Lhetero/2+l);
//				O[1][l] = H_hetero.Sz(Lhetero/2+l);
			}
			#elif defined(USE_SU2XU1)
			{
				O[0][l] = H_hetero.c(Lhetero/2+l,0);
				O[1][l] = H_hetero.cdag(Lhetero/2+l,0,1.);
//				O[0][l] = H_hetero.S(Lhetero/2+l,0);
//				O[1][l] = H_hetero.Sdag(Lhetero/2+l,0,1.);
			}
			#endif
			O[0][l].transform_base(Q,false,L); // PRINT=false
			O[1][l].transform_base(Q,false,L);
		}
		// Ofull
		std::array<vector<Mpo<MODEL::Symmetry,double>>,2> Ofull;
		Ofull[0].resize(Lhetero);
		Ofull[1].resize(Lhetero);
		for (int l=0; l<Lhetero; ++l)
		{
			#ifdef USE_SPINLESS
			{
				Ofull[0][l] = H_hetero.c(l);
				Ofull[1][l] = H_hetero.cdag(l);
			}
			#elif defined(USE_U1XU1)
			{
				Ofull[0][l] = H_hetero.c<UP>(l);
				Ofull[1][l] = H_hetero.cdag<UP>(l);
//				Ofull[0][l] = H_hetero.Scomp(SP,l);
//				Ofull[1][l] = H_hetero.Sz(l);
			}
			#elif defined(USE_SU2XU1)
			{
				Ofull[0][l] = H_hetero.c(l,0);
				Ofull[1][l] = H_hetero.cdag(l,0,1.);
//				Ofull[0][l] = H_hetero.S(l,0);
//				Ofull[1][l] = H_hetero.Sdag(l,0,1.);
			}
			#endif
			Ofull[0][l].transform_base(Q,false,L); // PRINT=false
			Ofull[1][l].transform_base(Q,false,L);
		}
		
		// OxV in cell
		Stopwatch<> OxVTimer;
		Mps<MODEL::Symmetry,double> Phi = uDMRG.create_Mps(Ncells, g, H, x0, false); // ground state as heterogenic MPS, ADD_ODD_SITE=false
		//---
		std::array<vector<Mps<MODEL::Symmetry,complex<double>>>,2> OxPhiCell;
		OxPhiCell[0].resize(L);
		OxPhiCell[1].resize(L);
		std::array<vector<Mps<MODEL::Symmetry,double>>,2> OxPhiCellReal;
		OxPhiCellReal[0] = uDMRG.create_Mps(Ncells, g, H, O[0][0], O[0], false);
		OxPhiCellReal[1] = uDMRG.create_Mps(Ncells, g, H, O[1][0], O[1], false);
		for (int l=0; l<L; ++l)
		{
			OxPhiCell[0][l] = OxPhiCellReal[0][l].template cast<complex<double>>();
			OxPhiCell[1][l] = OxPhiCellReal[1][l].template cast<complex<double>>();
		}
		// OxV for all sites
//		std::array<vector<Mps<MODEL::Symmetry,complex<double>>>,2> OxPhiFull;
//		OxPhiFull[0].resize(Lhetero);
//		OxPhiFull[1].resize(Lhetero);
//		std::array<vector<Mps<MODEL::Symmetry,double>>,2> OxPhiFullReal;
//		OxPhiFullReal[0] = uDMRG.create_Mps(Ncells, g, H, Ofull[0][x0], Ofull[0], false);
//		OxPhiFullReal[1] = uDMRG.create_Mps(Ncells, g, H, Ofull[1][x0], Ofull[1], false);
//		for (int l=0; l<Lhetero; ++l)
//		{
//			OxPhiFull[0][l] = OxPhiFullReal[0][l].template cast<complex<double>>();
//			OxPhiFull[1][l] = OxPhiFullReal[1][l].template cast<complex<double>>();
//		}
		//-----------
		lout << OxVTimer.info("OxV for all sites") << endl;
		
		// GreenPropagator
		Green[0] = GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> >(wd+"PES_"+base,tmax,Nt,wmin,wmax,501,ZERO_2PI,500,GAUSSINT);
		Green[1] = GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> >(wd+"IPE_"+base,tmax,Nt,wmin,wmax,501,ZERO_2PI,500,GAUSSINT);
//		Green[0] = GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> >(wd+"SF1_"+base,tmax,Nt,wmin,wmax,501,ZERO_2PI,500,GAUSSINT);
//		Green[1] = GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> >(wd+"SF2_"+base,tmax,Nt,wmin,wmax,501,ZERO_2PI,500,GAUSSINT);
		
		Green[1].set_verbosity(DMRG::VERBOSITY::ON_EXIT);
		
		// set operator to measure
		vector<Mpo<MODEL::Symmetry,double>> Measure;
		Measure.resize(Lhetero);
		for (int l=0; l<Measure.size(); ++l)
		{
			Measure[l] = H_hetero.n(l);
			Measure[l].transform_base(Q,false,L); // PRINT=false
		}
		Green[0].set_measurement(Measure,4,"n","measure");
		Green[1].set_measurement(Measure,4,"n","measure");
		
		double Eg = avg_hetero(Phi, H_hetero, Phi, true); // USE_BOUNDARY=true
		double Eg_ = Lhetero * g.energy;
		lout << setprecision(14) << "Eg=" << Eg << ", " << Eg_ << ", diff=" << abs(Eg-Eg_) << ", eg=" << g.energy << endl;
		
		std::array<bool,2> TIME_DIR;
		TIME_DIR[0] = false;
		TIME_DIR[1] = true;
		
//		///////////////////////////////////////
//		MODEL::StateXcd PhiProp = Phi.cast<complex<double>>();
//		MODEL::StateXcd PhiProp0 = Phi.cast<complex<double>>();
//		cout << PhiProp.info() << endl;
//		TDVPPropagator<MODEL,MODEL::Symmetry,double,complex<double>,Mps<MODEL::Symmetry,complex<double>>> TDVP(H_hetero,PhiProp);
//		double tval = 0.;
//		for (int j=0; j<12; ++j)
//		{
//			TDVP.t_step0(H_hetero, PhiProp, -1.i*0.1);
//			cout << TDVP.info() << endl;
//			tval += 0.1;
//		}
//		complex<double> phase = PhiProp0.dot(PhiProp);
//		complex<double> exact = exp(-1.i*Eg*tval);
//		cout << "phase=" << phase << ", exact=" << exact << ", diff=" << abs(phase-exact) << endl;
//		///////////////////////////////////////
		
		#pragma omp parallel for
		for (int z=0; z<2; ++z)
		{
//			Green[z].set_OxPhiFull(OxPhiFull[z]);
//			Green[z].compute(H_hetero, OxPhiCell[z], OxPhiFull[z][Lhetero/2], Eg, TIME_DIR[z]); // TIME_FORWARDS=false
			
			Green[z].compute_cell(H_hetero, OxPhiCell[z], Eg, TIME_DIR[z], true); // COUNTERPROPAGATE=true
			Green[z].FT_allSites();
			Green[z].ncell = ncell;
		}
		
		for (int z=0; z<2; ++z) Green[z].save();
		
		vector<vector<MatrixXcd>> Gin(L); for (int i=0; i<L; ++i) {Gin[i].resize(L);}
		for (int i=0; i<L; ++i) 
		for (int j=0; j<L; ++j)
		{
			Gin[i][j].resize(Nt,Ncells);
			Gin[i][j].setZero();
		}
		
		for (int i=0; i<L; ++i)
		for (int j=0; j<L; ++j)
		{
			Gin[i][j] += Green[0].get_GtxCell()[i][j] + Green[1].get_GtxCell()[i][j];
		}
		
//		Gfull = GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> >(wd+"A1P_"+base,tmax,Gin,ZERO_2PI,500,GAUSSINT);
//		Gfull.recalc_FTwCell(wmin,wmax,501);
//		Gfull.FT_allSites();
//		
//		IntervalIterator mu(wmin,wmax,101);
//		for (mu=mu.begin(); mu!=mu.end(); ++mu)
//		{
//			double res = spinfac * Gfull.integrate_Glocw_cell(*mu);
//			mu << res;
//		}
//		mu.save(wd+"n(μ)_"+base+".dat");
//		RootFinder R(n_mu,wmin,wmax);
//		lout << "μ=" << R.root() << endl;
//		
//		Gfull.mu = R.root();
//		Gfull.ncell = ncell;
//		Gfull.save();
	}
}

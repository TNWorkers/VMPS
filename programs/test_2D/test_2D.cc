#include <iostream>
#include <fstream>
#include <complex>
#include <variant>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "ArgParser.h"

#include "Geometry2D.h"
#include "Lattice2D.h"

#include "solvers/DmrgSolver.h"
#include "solvers/TDVPPropagator.h"
#include "solvers/MpsCompressor.h"

#include "HDF5Interface.h"

int M, Dtot;
double Stot;
size_t D;
size_t Lx, Ly;
double J, Jx, Jy, Jz, Jprime, Jrung, Jloc, Jtri, R, Bz;
double alpha;
double t;
size_t Dinit, Dlimit, Qinit, Imin, Imax, limalpha, lim_two_site_iter;
int max_Nrich;
double tol_eigval, tol_state, eps_svd;
double dt, tmax;
DMRG::VERBOSITY::OPTION VERB;

Eigenstate<MODEL::StateXd> g;

MatrixXd SpinCorr;
MatrixXcd FTSpinCorr;

int main (int argc, char* argv[])
{
	Sym::initialize(100,"cgc_hash/table_50.3j","cgc_hash/table_40.6j","cgc_hash/table_24.9j");

	ArgParser args(argc,argv);
	Lx = args.get<size_t>("Lx",6);
	Ly = args.get<size_t>("Ly",6);

	J = args.get<double>("J",1.);
	Jx = args.get<double>("Jx",J);
	Jy = args.get<double>("Jy",J);
	Jz = args.get<double>("Jz",J);
	Bz = args.get<double>("Bz",0.);
	R = args.get<double>("R",0.);
	Jrung = args.get<double>("Jrung",J);
	Jprime = args.get<double>("Jprime",0.);
	Jloc = args.get<double>("Jloc",0.);
	Jtri = args.get<double>("Jtri",0.);

	M = args.get<int>("M",0);
	D = args.get<size_t>("D",2);
	Dtot = abs(M)+1;
	Stot = (Dtot-1.)/2.;
	size_t min_Nsv = args.get<size_t>("min_Nsv",0ul);
	VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",2));
	
	LatticeType lattice = static_cast<LatticeType>(args.get<int>("lattice",0)); //0 SQUARE, 1 TRIANG
	
	bool PERIODIC_Y = args.get<bool>("PER_Y",false);
	bool PERIODIC_X = args.get<bool>("PER_X",false);

	bool CORR = args.get<bool>("CORR",true);
	
	size_t maxPower = args.get<size_t>("maxPower",2ul);
	
	eps_svd = args.get<double>("eps_svd",1e-7);
	alpha = args.get<double>("alpha",1e2);
	limalpha = args.get<size_t>("limalpha",30);
	lim_two_site_iter = args.get<size_t>("lim2site",10);
		
	Dinit  = args.get<size_t>("Dinit",2ul);
	Dlimit = args.get<size_t>("Dmax",200ul);
	Qinit  = args.get<size_t>("Qinit",7ul);
	Imin   = args.get<size_t>("Imin",2ul);
	Imax   = args.get<size_t>("Imax",50ul);
	tol_eigval = args.get<double>("tol_eigval",1e-7);
	tol_state  = args.get<double>("tol_state",1e-7);
	max_Nrich = args.get<int>("max_Nrich",-1);
	
	vector<Param> SweepParams;
	SweepParams.push_back({"max_alpha",alpha});
	SweepParams.push_back({"lim_alpha",limalpha});
	SweepParams.push_back({"eps_svd",eps_svd});
	SweepParams.push_back({"max_halfsweeps",Imax});
	SweepParams.push_back({"min_halfsweeps",Imin});
	SweepParams.push_back({"Dinit",Dinit});
	SweepParams.push_back({"Qinit",Qinit});
	SweepParams.push_back({"min_Nsv",min_Nsv});
	SweepParams.push_back({"Dlimit",Dlimit});
	SweepParams.push_back({"tol_eigval",tol_eigval});
	SweepParams.push_back({"tol_state",tol_state});
	SweepParams.push_back({"max_Nrich",max_Nrich});
	SweepParams.push_back({"CONVTEST",DMRG::CONVTEST::VAR_2SITE});
		
	lout << args.info() << endl;
	lout.set(make_string("Lx=",Lx,"_Ly=",Ly,"_M=",M,"_D=",D,"_J=",J,".log"),"log");

	std::string obsfile = make_string("obs/Lx=",Lx,"_Ly=",Ly,"_M=",M,"_D=",D,"_J=",J,"_",lattice,".h5");
	HDF5Interface target(obsfile,WRITE);
	target.close();
	
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif

	Lattice2D lat({Lx,Ly},{PERIODIC_X,PERIODIC_Y},lattice, 2ul);
	vector<double> coupl;
	coupl.push_back(1.); //nn coupling
	if (Jprime != 0.) {coupl.push_back(1.);} //nnn coupling
	Geometry2D Geo(lat,SNAKE,coupl); 
	
	Stopwatch<> Watch;
	MODEL H;
	qarray<MODEL::Symmetry::Nq> Qc;

	if constexpr( MODEL::FAMILY == HEISENBERG)
	{
		ArrayXXd Jarray = J * Geo.hopping(1);
		if (Jprime != 0.) {Jarray += Jprime * Geo.hopping(2);}
		cout << Jarray << endl;
		H = MODEL(Lx*Ly,{{"Jfull",Jarray},{"maxPower",maxPower}});
        #if defined(USING_U0)
		Qc = {};
        #elif defined(USING_U1)
		Qc = {M};
        #elif defined(USING_SU2)
		Qc = {Dtot};
		#endif
	}
	
	lout << H.info() << endl;
	
	MODEL::Solver DMRG(VERB);
	DMRG.userSetGlobParam();
	DMRG.userSetDynParam();
	DMRG.GlobParam = H.get_DmrgGlobParam(SweepParams);
	DMRG.DynParam = H.get_DmrgDynParam(SweepParams);
	DMRG.DynParam.iteration = [](size_t i) {return (i<lim_two_site_iter) ? DMRG::ITERATION::TWO_SITE : DMRG::ITERATION::ONE_SITE; };
	DMRG.edgeState(H, g, Qc, LANCZOS::EDGE::GROUND);
	
	t = Watch.time();
	cout << endl << endl << termcolor::bold << "E_gs/bond=" << g.energy/Geo.numberOfBonds() << termcolor::reset << endl;

	if (CORR)
	{
		SpinCorr.resize(Lx*Ly, Lx*Ly);
		for (int x0=0; x0<Lx; ++x0)
		for (int y0=0; y0<Ly; ++y0)
		for (int x1=0; x1<Lx; ++x1)
		for (int y1=0; y1<Ly; ++y1)
		{
			int i0 = Geo(x0,y0);
			int i1 = Geo(x1,y1);
			SpinCorr(i0,i1) = avg(g.state, H.SdagS(i0,i1), g.state);
		}
		FTSpinCorr.resize(Lx,Ly); FTSpinCorr.setZero();
		for (int ikx=0; ikx<Lx; ikx++)
		for (int iky=0; iky<Ly; iky++)
		{
			for (int x0=0; x0<Lx; ++x0)
			for (int x1=0; x1<Lx; ++x1)
			for (int y0=0; y0<Ly; ++y0)
			for (int y1=0; y1<Ly; ++y1)
			{
				int i0 = Geo(x0,y0);
				int i1 = Geo(x1,y1);
				auto R0 = x0*lat.a[0] + y0*lat.a[1];
				auto R1 = x1*lat.a[0] + y1*lat.a[1];
				auto k = ikx*lat.b[0] + iky*lat.b[1]; 
				FTSpinCorr(ikx,iky) += 1./(Lx*Ly) * exp(2.*M_PI*1.i * static_cast<double>(ikx/static_cast<double>(Lx)*(x0-x1) + iky/static_cast<double>(Ly)*(y0-y1))) * SpinCorr(i0,i1);
			}
		}
		target = HDF5Interface(obsfile,REWRITE);
		target.save_matrix(SpinCorr,"SiSj");
		Eigen::MatrixXd FTSpinCorrreal = FTSpinCorr.real();
		target.save_matrix(FTSpinCorrreal,"Sk");
		target.close();
	}	

	Sym::finalize(true);
}

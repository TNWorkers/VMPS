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

//-------- Test of the ArnoldiSolver:
//size_t dim (const MatrixXcd &A) {return A.rows();}
//#include "LanczosWrappers.h"
//#include "HxV.h"

#include "VUMPS/VumpsSolver.h"
#include "VUMPS/VumpsLinearAlgebra.h"
#include "VUMPS/UmpsCompressor.h"
#include "models/Heisenberg.h"
#include "models/HeisenbergU1.h"
#include "models/HeisenbergSU2.h"
#include "DmrgLinearAlgebra.h"
#include "solvers/TDVPPropagator.h"
#include "solvers/EntropyObserver.h"

#include "IntervalIterator.h"

size_t L, Ncells;
int M, Dtot, N;
double J, dt;
size_t Chi, max_iter, min_iter, Qinit, D;
double tol_eigval, tol_var, tol_state;

int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",2);
	Ncells = args.get<size_t>("Ncells",30);
	J = args.get<double>("J",1.);
	M = args.get<int>("M",0);
	Dtot = abs(M)+1;
	N = args.get<int>("N",L/2);
	D = args.get<size_t>("D",3ul);
	
	dt = args.get<double>("dt",0.1);
	double tol_compr = 1e-4;
	
	Chi = args.get<size_t>("Chi",20);
	tol_eigval = args.get<double>("tol_eigval",1e-7);
	tol_var = args.get<double>("tol_var",1e-6);
	tol_state = args.get<double>("tol_state",1e-2);
	
	max_iter = args.get<size_t>("max_iter",100ul);
	min_iter = args.get<size_t>("min_iter",1ul);
	Qinit = args.get<size_t>("Qinit",6ul);
	
	VUMPS::CONTROL::GLOB GlobParams;
	GlobParams.min_iterations = min_iter;
	GlobParams.max_iterations = max_iter;
	GlobParams.Dinit  = Chi;
	GlobParams.Dlimit = Chi;
	GlobParams.Qinit = Qinit;
	GlobParams.tol_eigval = tol_eigval;
	GlobParams.tol_var = tol_var;
	GlobParams.tol_state = tol_state;
	GlobParams.max_iter_without_expansion = 20ul;
	
//	typedef VMPS::HeisenbergSU2 MODEL;
//	qarray<MODEL::Symmetry::Nq> Qc = {Dtot};
//	cout << "Dtot=" << Dtot << endl;
//	MODEL H(L,{{"J",J},{"OPEN_BC",false},{"D",D},{"CALC_SQUARE",false}});
//	lout << H.info() << endl;
//	H.transform_base(Qc);
//	
//	MODEL::uSolver uDMRG(DMRG::VERBOSITY::HALFSWEEPWISE);
//	Eigenstate<MODEL::StateUd> g;
//	uDMRG.userSetGlobParam();
//	uDMRG.userSetLanczosParam();
//	uDMRG.GlobParam = GlobParams;
//	uDMRG.edgeState(H, g, Qc);
	
	
	
	
	typedef VMPS::HeisenbergU1 MODEL;
	qarray<MODEL::Symmetry::Nq> Qg = MODEL::Symmetry::qvacuum();
	
	MODEL H(L,{{"J",J},{"D",D},{"OPEN_BC",false}});
	lout << H.info() << endl;
//	H.transform_base(Qg);
	
	MODEL::uSolver uDMRG(DMRG::VERBOSITY::HALFSWEEPWISE);
	Eigenstate<MODEL::StateUd> g;
	uDMRG.userSetGlobParam();
	uDMRG.userSetLanczosParam();
	uDMRG.GlobParam = GlobParams;
	uDMRG.edgeState(H, g, Qg);
	
	Mps<MODEL::Symmetry,double> Psi = uDMRG.create_Mps(Ncells, H, g, true);
	lout << Psi.info() << endl;
	
	MODEL::StateXd PsiTmp;
	
	size_t Lhetero = L*Ncells+1;
	
	MODEL H_hetero(Lhetero,{{"J",1.},{"D",D},{"OPEN_BC",false}});
	Psi.graph("before");
	OxV_exact(H_hetero.Scomp(SP,Ncells), Psi, PsiTmp, 2., DMRG::VERBOSITY::HALFSWEEPWISE);
//	OxV_exact(H_hetero.S(Ncells), Psi, PsiTmp, 2., DMRG::VERBOSITY::HALFSWEEPWISE);
	PsiTmp.graph("after");
	PsiTmp.sweep(0,DMRG::BROOM::QR);
	lout << PsiTmp.info() << endl << endl;
	Psi = PsiTmp;
	
	IntervalIterator x(-0.5*Lhetero, 0.5*Lhetero, Lhetero);
	for (x=x.begin(); x!=x.end(); ++x)
	{
		double res = avg(Psi, H_hetero.Scomp(SZ,x.index()), Psi, false, DMRG::DIRECTION::RIGHT);
		lout << "x=" << x.index() << ", <Sz>=" << res << endl;
		x << res;
	}
	lout << endl;
	x.save(make_string("Sz_L=",Lhetero,"_t=0.dat"));
	
	Mps<MODEL::Symmetry,complex<double> > Psit;
	Psit = Psi.cast<complex<double> >();
	Psit.eps_svd = tol_compr;
	Psit.max_Nsv = Psi.calc_Dmax();
	
	IntervalIterator t(0,1,20);
	TDVPPropagator<MODEL,MODEL::Symmetry,double,complex<double>,MODEL::StateXcd> TDVP(H_hetero, Psit);
	EntropyObserver<MODEL::StateXcd> Sobs(Lhetero,20*10,DMRG::VERBOSITY::HALFSWEEPWISE);
	vector<bool> TWO_SITE = Sobs.TWO_SITE(0,Psit);
	
	double tval = 0.;
	
	for (int j=0; j<10; ++j)
	{
		for (t=t.begin(); t!=t.end(); ++t)
		{
			TDVP.t_step_adaptive(H_hetero, Psit, -1.i*dt, TWO_SITE, 1,1e-8);
			tval += dt;
			
			if (Psit.get_truncWeight().sum() > 0.5*tol_compr)
			{
				lout << "Psit.get_truncWeight().sum()=" << Psit.get_truncWeight().sum() << endl;
				Psit.max_Nsv = min(static_cast<size_t>(max(Psit.max_Nsv*1.1, Psit.max_Nsv+1.)),200ul);
				lout << termcolor::yellow << "Setting Psi.max_Nsv to " << Psit.max_Nsv << termcolor::reset << endl;
			}
			
			lout << "t=" << tval << endl;
			lout << TDVP.info() << endl;
			lout << Psit.info() << endl;
			lout << endl;
			
			auto PsiTmp = Psit;
			PsiTmp.eps_svd = 1e-15;
			PsiTmp.skim(DMRG::BROOM::SVD);
			TWO_SITE = Sobs.TWO_SITE(t.index()+10*j,PsiTmp);
		}
		
		for (x=x.begin(); x!=x.end(); ++x)
		{
			double res = isReal(avg(Psit, H_hetero.Sz(x.index()), Psit, false, DMRG::DIRECTION::RIGHT));
			lout << "x=" << x.index() << ", <Sz>=" << res << endl;
			x << res;
		}
		x.save(make_string("Sz_L=",Lhetero,"_t=",tval,".dat"));
	}
	
	
	
}

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
#include "models/HubbardU1xU1.h"
#include "models/HubbardSU2xU1.h"
#include "DmrgLinearAlgebra.h"
#include "solvers/TDVPPropagator.h"

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
	Ncells = args.get<size_t>("Ncells",20);
	J = args.get<double>("J",1.);
	M = args.get<int>("M",0);
	Dtot = abs(M)+1;
	N = args.get<int>("N",L/2);
	
	dt = args.get<double>("dt",0.1);
	double tol_compr = 1e-4;
	
	Chi = args.get<size_t>("Chi",20);
	tol_eigval = args.get<double>("tol_eigval",1e-7);
	tol_var = args.get<double>("tol_var",1e-6);
	tol_state = args.get<double>("tol_state",1e-2);
	
	max_iter = args.get<size_t>("max_iter",100ul);
	min_iter = args.get<size_t>("min_iter",1ul);
	Qinit = args.get<size_t>("Qinit",6ul);
	D = args.get<size_t>("D",3ul);
	
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
	qarray<MODEL::Symmetry::Nq> Qg = {0};
	qarray<MODEL::Symmetry::Nq> Qc = {1};
	
	MODEL H(L,{{"J",1.},{"D",D},{"OPEN_BC",false}});
	H.transform_base(Qg);
	lout << H.info() << endl;
	
	MODEL::uSolver uDMRG(DMRG::VERBOSITY::HALFSWEEPWISE);
	Eigenstate<MODEL::StateUd> g;
	uDMRG.userSetGlobParam();
	uDMRG.userSetLanczosParam();
	uDMRG.GlobParam = GlobParams;
	uDMRG.edgeState(H, g, Qg);
	
	Mps<MODEL::Symmetry,double> Psi = uDMRG.create_Mps(Ncells, H, g);
	lout << Psi.info() << endl;
//	lout << Psi.BoundaryL.print() << endl;
	
	MODEL::StateXd PsiTmp;
	
	MODEL H_hetero(L*Ncells,{{"J",1.},{"D",D},{"OPEN_BC",false}});
	OxV_exact(H_hetero.Scomp(SP,Ncells-1), Psi, PsiTmp, 2., DMRG::VERBOSITY::HALFSWEEPWISE);
	PsiTmp.graph("after_prod");
	PsiTmp.sweep(0,DMRG::BROOM::QR);
	lout << PsiTmp.info() << endl << endl;
	Psi = PsiTmp;
	
	IntervalIterator x(-double(L*Ncells)/2.+1., L*Ncells/2., L*Ncells);
	for (x=x.begin(); x!=x.end(); ++x)
	{
		double res = avg(Psi, H_hetero.Scomp(SZ,x.index()), Psi, false, DMRG::DIRECTION::RIGHT);
		lout << "x=" << x.index() << ", <Sz>=" << res << endl;
		x << res;
	}
	lout << endl;
	x.save(make_string("Sz_L=",L*Ncells,"_t=0.dat"));
	
	Mps<MODEL::Symmetry,complex<double> > Psit;
	Psit = Psi.cast<complex<double> >();
	Psit.eps_svd = tol_compr;
	Psit.max_Nsv = Psi.calc_Dmax();
	
//	Psit.transform_base(Qc);
//	H_hetero.transform_base(Qc);
//	Psit.graph("after_transform");
	
//	lout << Psit.BoundaryL.print() << endl;
	
	TDVPPropagator<MODEL,MODEL::Symmetry,double,complex<double>,MODEL::StateXcd> TDVP(H_hetero, Psit);
	
	IntervalIterator t(0,1,20);
	double tval = 0.;
	
	for (int j=0; j<10; ++j)
	{
		for (t=t.begin(); t!=t.end(); ++t)
		{
			TDVP.t_step0(H_hetero, Psit, -1.i*dt, 1,1e-8);
			Psit.graph("Psit");
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
		}
		
		for (x=x.begin(); x!=x.end(); ++x)
		{
			double res = isReal(avg(Psit, H_hetero.Sz(x.index()), Psit, false, DMRG::DIRECTION::RIGHT));
			lout << "x=" << x.index() << ", <Sz>=" << res << endl;
			x << res;
		}
		x.save(make_string("Sz_L=",L*Ncells,"_t=",tval,".dat"));
	}
	
	
	
}

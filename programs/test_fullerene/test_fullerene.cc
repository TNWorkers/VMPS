#define LANCZOS_MAX_ITERATIONS 1e2

#define USE_HDF5_STORAGE
#define DMRG_DONT_USE_OPENMP
#define MPSQCOMPRESSOR_DONT_USE_OPENMP

#include <iostream>
#include <fstream>
#include <complex>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

#include "Logger.h"
Logger lout;
#include "ArgParser.h"

#include "StringStuff.h"
#include "Stopwatch.h"

#include <Eigen/Core>
using namespace Eigen;
#include <unsupported/Eigen/FFT>

#include "solvers/DmrgSolver.h"
#include "DmrgLinearAlgebra.h"
#include "models/ParamCollection.h"

//#include "models/HubbardSU2xU1.h"
//typedef VMPS::HubbardSU2xU1 MODEL;

#include "models/HeisenbergSU2.h"
typedef VMPS::HeisenbergSU2 MODEL;
// L=12 exact E0/L=-0.51566, SdagS=-0.20626
// L=20 exact E0/L=-0.48611, SdagS=-0.32407

#include "solvers/TDVPPropagator.h"

ArrayXXd permute_random (const ArrayXXd &A)
{
	PermutationMatrix<Dynamic,Dynamic> P(A.rows());
	P.setIdentity();
	srand(time(0));
	std::random_shuffle(P.indices().data(), P.indices().data()+P.indices().size());
	MatrixXd A_p = P.transpose() * A.matrix() * P;
//	cout << "(A-A_p).norm()=" << (A.matrix()-A_p).norm() << endl;
	return A_p.array();
}

/////////////////////////////////
int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	int L = args.get<int>("L",60);
	assert(L==12 or L==20 or L==60);
	double t = args.get<double>("t",1.);
	double U = args.get<double>("U",0.);
	int N = args.get<int>("N",L);
	
	bool BETAPROP = static_cast<bool>(args.get<int>("BETAPROP",0));
	double dbeta = args.get<double>("dbeta",0.1);
	double betamax = args.get<double>("betamax",20.);
	int Nbeta = static_cast<int>(betamax/dbeta);
	
	size_t maxPower = args.get<size_t>("maxPower",1ul);
	double tol_compr = args.get<double>("tol_compr",1e-15);
	
	string wd = args.get<string>("wd","./");
	if (wd.back() != '/') {wd += "/";}
	
	string base;
	if constexpr (MODEL::FAMILY == HUBBARD)
	{
		base = make_string("L=",L,"_N=",N,"_U=",U);
	}
	else
	{
		base = make_string("L=",L);
	}
	if (BETAPROP)
	{
		base += make_string("_dbeta=",dbeta);
	}
	lout.set(base+".log",wd+"log");
	
	DMRG::VERBOSITY::OPTION VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",DMRG::VERBOSITY::HALFSWEEPWISE));
	
	// glob. params
	DMRG::CONTROL::GLOB GlobParam;
	GlobParam.min_halfsweeps = args.get<size_t>("min_halfsweeps",1ul);
	GlobParam.max_halfsweeps = args.get<size_t>("max_halfsweeps",100ul);
	GlobParam.Dinit = args.get<size_t>("Dinit",2ul);
	GlobParam.Qinit = args.get<size_t>("Qinit",6ul);
	GlobParam.CONVTEST = DMRG::CONVTEST::VAR_HSQ; // DMRG::CONVTEST::VAR_2SITE
	
	// dyn. params
	DMRG::CONTROL::DYN  DynParam;
	int max_Nrich = args.get<int>("max_Nrich",-1);
	DynParam.max_Nrich = [max_Nrich] (size_t i) {return max_Nrich;};
	
	size_t Dincr_per = args.get<size_t>("Dincr_per",4ul);
	DynParam.Dincr_per = [Dincr_per] (size_t i) {return Dincr_per;};
	
	size_t Dincr_abs = args.get<size_t>("Dincr_abs",6ul);
	DynParam.Dincr_abs = [Dincr_abs] (size_t i) {return Dincr_abs;};
	
	size_t lim_alpha = args.get<size_t>("lim_alpha",0.8*GlobParam.max_halfsweeps);
	double alpha = args.get<double>("alpha",100.);
	DynParam.max_alpha_rsvd = [lim_alpha, alpha] (size_t i) {return (i<lim_alpha)? alpha:0.;};
	
	GlobParam.savePeriod = Dincr_per;
	GlobParam.saveName = make_string(wd,MODEL::FAMILY,"_",base);
	
	lout << args.info() << endl;
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	ArrayXXd hopping = hopping_fullerene(L);
	if (L<60) lout << endl << hopping << endl << endl;
	
	// free fermions
	SelfAdjointEigenSolver<MatrixXd> Eugen(-1.*hopping.matrix());
	VectorXd occ = Eugen.eigenvalues().head(N/2);
	VectorXd unocc = Eugen.eigenvalues().tail(L-N/2);
	lout << "orbital energies occupied:" << endl << occ  << endl;
	lout << "orbital energies unoccupied:" << endl << unocc  << endl << endl;
	double E0 = 2.*occ.sum();
	lout << setprecision(16) << "non-interacting: E0=" << E0 << ", E0/L=" << E0/L << endl << endl;
	
	if (!BETAPROP)
	{
		vector<Param> params;
		if constexpr (MODEL::FAMILY == HUBBARD)
		{
			params.push_back({"U",U});
			params.push_back({"tFull",hopping});
		}
		else
		{
			params.push_back({"Jfull",hopping});
		}
		params.push_back({"maxPower",maxPower});
		
		MODEL H(size_t(L),params);
		lout << H.info() << endl;
		
		Eigenstate<MODEL::StateXd> g;
		MODEL::Solver DMRG(VERB);
		DMRG.userSetGlobParam();
		DMRG.userSetDynParam();
		DMRG.GlobParam = GlobParam;
		DMRG.DynParam = DynParam;
		DMRG.edgeState(H, g, MODEL::singlet(), LANCZOS::EDGE::GROUND);
		
		if constexpr (MODEL::FAMILY == HEISENBERG)
		{
			lout << "SdagS=" << avg(g.state, H.SdagS(0,1), g.state) << endl;
			lout << "SdagS=" << avg(g.state, H.SdagS(1,2), g.state) << endl;
		}
	}
	else
	{
		assert(MODEL::FAMILY == HEISENBERG);
		
	//	vector<Param> beta0_params;
	////	params.push_back({"tFull",hopping});
	//	beta0_params.push_back({"J",1.,0});
	//	beta0_params.push_back({"J",0.,1});
	//	beta0_params.push_back({"maxPower",2ul});
	//	MODEL H0(size_t(2*L),beta0_params);
	//	lout << H0.info() << endl;
		
		vector<Param> beta0_params;
		beta0_params.push_back({"Ly",2ul});
		beta0_params.push_back({"Jrung",1.});
		beta0_params.push_back({"J",0.});
		beta0_params.push_back({"maxPower",1ul});
		MODEL H0(size_t(L),beta0_params);
		lout << H0.info() << endl;
		
		Eigenstate<MODEL::StateXd> g;
		MODEL::Solver DMRG(VERB);
		DMRG.edgeState(H0, g, MODEL::singlet(), LANCZOS::EDGE::GROUND, false);
		lout << endl;
		
	//	vector<Param> beta_params;
	//	ArrayXXd hopping_ext(2*L,2*L); hopping_ext=0.;
	//	for (int i=0; i<L; ++i)
	//	for (int j=0; j<L; ++j)
	//	{
	//		hopping_ext(2*i,2*j) = hopping(i,j);
	//	}
	//	hopping_ext += create_1D_OBC(2*L,1e-7); // to avoid decoupling bug
	//	beta_params.push_back({"Jfull",hopping_ext});
	//	beta_params.push_back({"maxPower",2ul});
	//	MODEL H(size_t(2*L),beta_params);
	//	lout << H.info() << endl;
		
		vector<Param> beta_params;
		beta_params.push_back({"Jfull",hopping});
		beta_params.push_back({"Ly",2ul});
		beta_params.push_back({"maxPower",1ul});
		beta_params.push_back({"J",0.});
		beta_params.push_back({"Jrung",0.});
		MODEL H(size_t(L),beta_params);
		lout << H.info() << endl;
		
		auto PsiT = g.state;
		PsiT.eps_svd = 1e-5;
		PsiT.min_Nsv = 0ul;
		PsiT.max_Nsv = 100ul;
		TDVPPropagator<MODEL,MODEL::Symmetry,double,double,MODEL::StateXd> TDVPT(H,PsiT);
		
		vector<MODEL::Operator> Sdagtot(L);
		vector<MODEL::Operator> Stot(L);
		for (int i=0; i<L; ++i)
		{
			Sdagtot[i] = H.Sdag(i);
			Stot[i] = H.S(i);
		}
		
		ofstream Filer(make_string(wd,"thermodyn_L=",L,"_dβ=",dbeta,".dat"));
		Filer << "#C\te\tChi" << endl;
		double beta = 0.;
		for (int i=0; i<Nbeta; ++i)
		{
			Stopwatch<> betaStepper;
			if (PsiT.calc_Dmax() == 100ul)
			{
				TDVPT.t_step0(H, PsiT, -0.5*dbeta, 1, 1e-8);
			}
			else
			{
				TDVPT.t_step(H, PsiT, -0.5*dbeta, 1, 1e-8);
			}
			PsiT /= sqrt(dot(PsiT,PsiT));
			beta = (i+1)*dbeta;
			lout << TDVPT.info() << endl;
			lout << setprecision(16) << PsiT.info() << setprecision(6) << endl;
			double C = beta*beta*(avg(PsiT,H,H,PsiT)-pow(avg(PsiT,H,PsiT),2))/L; // specific heat
			double e = avg(PsiT,H,PsiT)/L; // inner energy density
			double chi = beta*(avg(PsiT,Sdagtot,Stot,PsiT))/L; // pow(avg(PsiT,Stot,PsiT),2)=0 // uniform magnetic susceptibility
			
			auto PsiTtmp = PsiT; PsiTtmp.entropy_skim();
			lout << "S=" << PsiTtmp.entropy().transpose() << endl;
			
			lout << "β=" << beta << ", T=" << 1./beta << ", e=" << e << ", C=" << C << ", chi=" << chi << endl;
			Filer << 1./beta << "\t" << C << "\t" << e << "\t" << chi << endl;
			lout << betaStepper.info("βstep") << endl;
			lout << endl;
		}
		Filer.close();
	}
}

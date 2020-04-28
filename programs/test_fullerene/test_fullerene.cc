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

#include "models/HeisenbergSU2.h"
typedef VMPS::HeisenbergSU2 MODEL;
// L=12 exact E0/L=-0.51566, SdagS=-0.20626
// L=20 exact E0/L=-0.48611, SdagS=-0.32407

#include "models/DoubleHeisenbergSU2.h"
typedef VMPS::DoubleHeisenbergSU2 MODELC;

//#include "models/DoubleHeisenbergU1.h"
//typedef VMPS::DoubleHeisenbergU1 MODELC;

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
	assert(L==6 or L==12 or L==20 or L==60);
	double t = args.get<double>("t",1.);
	double U = args.get<double>("U",0.);
	int N = args.get<int>("N",L);
	int S = args.get<int>("S",0);
	size_t D = args.get<size_t>("D",2ul);
	size_t maxPower = args.get<size_t>("maxPower",1ul);
	
	bool BETAPROP = static_cast<bool>(args.get<int>("BETAPROP",0));
	bool CANONICAL = static_cast<bool>(args.get<int>("CANONICAL",0));
	bool CALC_C = static_cast<bool>(args.get<int>("CALC_C",1));
	double dbeta = args.get<double>("dbeta",0.1);
	double betamax = args.get<double>("betamax",20.);
	int Nbeta = static_cast<int>(betamax/dbeta);
	double tol_beta_compr = args.get<double>("tol_beta_compr",1e-5);
	size_t Dbetalimit = args.get<size_t>("Dbetalimit",100ul);
	bool CALC_CHI = static_cast<bool>(args.get<int>("CALC_CHI",1));
	size_t Ly = args.get<size_t>("Ly",2ul);
	int dLphys = (Ly==2ul)? 1:2;
	int N_stages = args.get<int>("N_stages",1);
	string LOAD = args.get<string>("LOAD","");
	
	string wd = args.get<string>("wd","./");
	if (wd.back() != '/') {wd += "/";}
	
	string base;
	if constexpr (MODEL::FAMILY == HUBBARD)
	{
		base = make_string("L=",L,"_N=",N,"_S=",S,"_U=",U);
	}
	else
	{
		base = make_string("L=",L,"_D=",D,"_S=",S);
	}
	if (BETAPROP)
	{
		base += make_string("_Ly=",Ly,"_dbeta=",dbeta,"_tol=",tol_beta_compr,"_Dlim=",Dbetalimit);
	}
	lout.set(base+".log",wd+"log");
	
	DMRG::VERBOSITY::OPTION VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",DMRG::VERBOSITY::HALFSWEEPWISE));
	
	// glob. params
	DMRG::CONTROL::GLOB GlobParam;
	GlobParam.min_halfsweeps = args.get<size_t>("min_halfsweeps",1ul);
	GlobParam.max_halfsweeps = args.get<size_t>("max_halfsweeps",100ul);
	GlobParam.Dinit = args.get<size_t>("Dinit",2ul);
	GlobParam.Qinit = args.get<size_t>("Qinit",6ul);
	GlobParam.CONVTEST = DMRG::CONVTEST::VAR_2SITE; // DMRG::CONVTEST::VAR_HSQ
	
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
	lout << "adjacency:" << endl << hopping << endl << endl;
	lout << "distances:" << endl << calc_distanceMatrix(hopping) << endl << endl;
	
	// free fermions
	SelfAdjointEigenSolver<MatrixXd> Eugen(-1.*hopping.matrix());
	VectorXd occ = Eugen.eigenvalues().head(N/2);
	VectorXd unocc = Eugen.eigenvalues().tail(L-N/2);
	lout << "orbital energies occupied:" << endl << occ.transpose()  << endl;
	lout << "orbital energies unoccupied:" << endl << unocc.transpose()  << endl << endl;
	double E0 = 2.*occ.sum();
	lout << setprecision(16) << "non-interacting: E0=" << E0 << ", E0/L=" << E0/L << setprecision(6) << endl << endl;
	
	//---------groundstate---------
	if (!BETAPROP)
	{
		vector<Param> params;
		qarray<MODEL::Symmetry::Nq> Q;
		if constexpr (MODEL::FAMILY == HUBBARD)
		{
			params.push_back({"U",U});
			params.push_back({"tFull",hopping});
			Q = {int(2*S+1),N};
		}
		else
		{
			params.push_back({"Jfull",hopping});
			params.push_back({"D",D});
			Q = {int(2*S+1)};
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
//		if (LOAD!="")
//		{
//			g.state.load(LOAD);
//			DMRG.edgeState(H, g, Q, LANCZOS::EDGE::GROUND, true);
//		}
//		else
//		{
//			DMRG.edgeState(H, g, Q, LANCZOS::EDGE::GROUND);
//		}
		
		lout << "varE=" << abs(avg(g.state,H,H,g.state)-pow(g.energy,2))/L << endl;
		
		if constexpr (MODEL::FAMILY == HEISENBERG)
		{
			g.state.load(LOAD);
			ofstream CorrFiler(make_string(wd,"SdagS_",base,".dat"));
			CorrFiler << "#d\tSdagS\tstdev" << endl;
			auto distanceMatrix = calc_distanceMatrix(hopping);
			lout << "nearest-neighbour correlations:" << endl;
			for (int d=1; d<=distanceMatrix.maxCoeff(); ++d)
			{
				vector<double> SdagS_bonds_;
				for (int j=0; j<L; ++j)
				for (int i=0; i<j; ++i)
				{
					if (distanceMatrix(i,j) == d)
					{
						double res = avg(g.state, H.SdagS(i,j), g.state);
						lout << "i=" << i << ", j=" << j << ", d=" << d << ", SdagS=" << res << endl;
						SdagS_bonds_.push_back(res);
					}
				}
				ArrayXd SdagS_bonds = Map<ArrayXd,Unaligned>(SdagS_bonds_.data(), SdagS_bonds_.size());
				double mean = SdagS_bonds.sum()/SdagS_bonds.rows();
				double var = (SdagS_bonds-mean).pow(2).sum()/SdagS_bonds.rows();
				double stdev = sqrt(var);
				lout << "d=" << d << ", SdagS mean=" << mean << ", var=" << var << ", stdev=" << stdev << endl;
				CorrFiler << d << "\t" << mean << "\t" << stdev << endl;
			}
		}
	}
	//-------canonical-------
	else if (BETAPROP and CANONICAL)
	{
		assert(MODEL::FAMILY == HEISENBERG);
		
		vector<Param> beta0_params;
		MODELC H0;
		ArrayXXd AllConnected(L,L); AllConnected=-1.; AllConnected.matrix().diagonal().setZero();
		beta0_params.push_back({"Kfull",AllConnected});
		beta0_params.push_back({"maxPower",2ul});
		H0 = MODELC(L,beta0_params);
		lout << H0.info() << endl;
		
		Eigenstate<MODELC::StateXd> g;
		MODELC::Solver DMRG(VERB);
		DMRG.edgeState(H0, g, MODELC::singlet(), LANCZOS::EDGE::GROUND, false);
		
		// test entanglement
		MatrixXd SdagSphys_Tinf(L,L); SdagSphys_Tinf.setZero();
		MatrixXd SdagSancl_Tinf(L,L); SdagSancl_Tinf.setZero();
		#pragma omp parallel for collapse(2)
		for (int j=0; j<L; ++j)
		for (int i=0; i<L; ++i)
		{
			SdagSphys_Tinf(i,j) = avg(g.state, H0.SdagS<0>(i,j), g.state);
			SdagSancl_Tinf(i,j) = avg(g.state, H0.SdagS<1>(i,j), g.state);
		}
		lout << "entanglement test physical spins:" << endl;
		lout << SdagSphys_Tinf << endl << endl;
		lout << "entanglement test ancillary spins:" << endl;
		lout << SdagSancl_Tinf << endl << endl;
		
		vector<Param> beta_params;
		if (CALC_C)
		{
			beta_params.push_back({"maxPower",(L==60)?1ul:2ul});
		}
		else
		{
			beta_params.push_back({"maxPower",1ul});
		}
		MODELC H;
		if (L==6)
		{
			beta_params.push_back({"J",1.});
		}
		else
		{
			beta_params.push_back({"Jfull",hopping});
		}
		H = MODELC(size_t(L),beta_params);
		lout << H.info() << endl;
		lout << endl;
		
		auto PsiT = g.state;
		PsiT.max_Nsv = Dbetalimit;
		TDVPPropagator<MODELC,MODELC::Symmetry,double,double,MODELC::StateXd> TDVPT(H,PsiT);
		
		ofstream Filer(make_string(wd,"thermodynC_",base,".dat"));
		Filer << "#T\tc\te\tchi" << endl;
		double beta = 0.;
		vector<double> cvec;
		vector<double> evec;
		vector<double> chivec;
		auto PsiTprev = PsiT;
		
		for (int i=0; i<Nbeta+1; ++i)
		{
			Stopwatch<> FullTimer;
			
			#pragma omp parallel sections
			{
				#pragma omp section
				{
					if (i!=Nbeta)
					{
						Stopwatch<> Stepper;
//						if (i==0)
//						{
//							PsiT.eps_svd = 0.;
//							PsiT.min_Nsv = 1ul;
//						}
//						else
//						{
							PsiT.eps_svd = tol_beta_compr;
							PsiT.min_Nsv = 0ul;
//						}
						
						if (PsiT.calc_Dmax() == Dbetalimit and i*dbeta>1.)
						{
							PsiT.eps_svd = 0.1*tol_beta_compr;
							TDVPT.t_step0(H, PsiT, -0.5*dbeta, N_stages, 1e-8);
						}
						else
						{
							TDVPT.t_step(H, PsiT, -0.5*dbeta, N_stages, 1e-8);
						}
						PsiT /= sqrt(dot(PsiT,PsiT));
						lout << "propagated to: β=" << (i+1)*dbeta << endl;
						lout << TDVPT.info() << endl;
						lout << setprecision(16) << PsiT.info() << setprecision(6) << endl;
						lout << Stepper.info("βstep") << endl;
					}
				}
				#pragma omp section
				{
					if (i>0)
					{
						double beta = i*dbeta;
						Stopwatch<> Stepper;
						//---------inner energy density---------
						double E = avg(PsiTprev,H,PsiTprev);
						double e = E/L;
						evec.push_back(e);
						lout << Stepper.info("e") << endl;
						//---------specific heat---------
						double c = std::nan("c");
						if (CALC_C)
						{
							c = (L==60)? beta*beta*(avg(PsiTprev,H,H,PsiTprev  )-pow(E,2))/L:
						                 beta*beta*(avg(PsiTprev,H,PsiTprev,2ul)-pow(E,2))/L;
						}
						cvec.push_back(c);
						lout << Stepper.info("c") << endl;
						//---------uniform magnetic susceptibility---------
						double chi = beta*avg(PsiTprev,H.Sdagtot<0>(0,sqrt(3.)),H.Stot<0>(0,1.),PsiTprev)/L;
						chivec.push_back(chi);
						//---------
						lout << Stepper.info("chi") << endl;
					}
				}
			}
			
			PsiTprev = PsiT;
			
			// entropy
			auto PsiTtmp = PsiT; PsiTtmp.entropy_skim(); lout << "S=" << PsiTtmp.entropy().transpose() << endl;
			
			if (i>0)
			{
				lout << termcolor::blue 
				     << "β=" << i*dbeta << ", T=" << 1./(i*dbeta) 
				     << ", e=" << evec[i-1] 
				     << ", c=" << cvec[i-1] 
				     << ", χ=" << chivec[i-1] 
				     << termcolor::reset
				     << endl;
				Filer << 1./(i*dbeta) << "\t" << cvec[i-1] << "\t" << evec[i-1] << "\t" << chivec[i-1] << endl;
			}
			
			lout << FullTimer.info("total") << endl;
			lout << endl;
		}
		
		Filer.close();
	}
	//-------grand canonical-------
	else
	{
		assert(MODEL::FAMILY == HEISENBERG);
		
		vector<Param> beta0_params;
		beta0_params.push_back({"Ly",Ly});
		beta0_params.push_back({"maxPower",1ul});
		MODEL H0;
		if (Ly==1)
		{
			beta0_params.push_back({"J",1.,0});
			beta0_params.push_back({"J",0.,1});
			H0 = MODEL(size_t(2*L),beta0_params);
		}
		else
		{
			beta0_params.push_back({"Jrung",1.});
			beta0_params.push_back({"J",0.});
			H0 = MODEL(size_t(L),beta0_params);
		}
		lout << H0.info() << endl;
		
		Eigenstate<MODEL::StateXd> g;
		MODEL::Solver DMRG(DMRG::VERBOSITY::ON_EXIT);
		DMRG.edgeState(H0, g, MODEL::singlet(), LANCZOS::EDGE::GROUND, false);
		
		// Zero hopping may cause problems. Restart until the correct product state is reached.
		if (Ly==1)
		{
			vector<bool> ENTROPY_CHECK;
			for (int l=1; l<2*L-1; l+=2) ENTROPY_CHECK.push_back(abs(g.state.entropy()(l))<1e-10);
			bool ALL = all_of(ENTROPY_CHECK.begin(), ENTROPY_CHECK.end(), [](const bool v){return v;});
			
			while (ALL == false)
			{
				lout << termcolor::yellow << "restarting..." << termcolor::reset << endl;
				DMRG.edgeState(H0, g, MODEL::singlet(), LANCZOS::EDGE::GROUND, false);
				ENTROPY_CHECK.clear();
				for (int l=1; l<2*L-1; l+=2) ENTROPY_CHECK.push_back(abs(g.state.entropy()(l))<1e-10);
				for (int l=1; l<2*L-1; l+=2)
				{
					bool TEST = abs(g.state.entropy()(l))<1e-10;
//					lout << "l=" << l << ", S=" << abs(g.state.entropy()(l)) << "\t" << boolalpha << TEST << endl;
				}
				ALL = all_of(ENTROPY_CHECK.begin(), ENTROPY_CHECK.end(), [](const bool v){return v;});
//				lout << boolalpha << "ALL=" << ALL << endl;
			}
		}
		
		lout << endl;
		
		vector<Param> beta_params;
		beta_params.push_back({"Ly",Ly});
		if (CALC_C)
		{
			beta_params.push_back({"maxPower",(L==60)?1ul:2ul});
		}
		else
		{
			beta_params.push_back({"maxPower",1ul});
		}
		beta_params.push_back({"J",0.});
		beta_params.push_back({"Jrung",0.});
		MODEL H;
		if (Ly==1)
		{
			ArrayXXd hopping_ext(2*L,2*L); hopping_ext=0.;
			for (int i=0; i<L; ++i)
			for (int j=0; j<L; ++j)
			{
				hopping_ext(2*i,2*j) = hopping(i,j);
			}
			if (L==60) hopping_ext += create_1D_OBC(2*L,1e-6); // to avoid decoupling bug
			beta_params.push_back({"Jfull",hopping_ext});
			H = MODEL(size_t(2*L),beta_params);
		}
		else
		{
			beta_params.push_back({"Jfull",hopping});
			H = MODEL(size_t(L),beta_params);
		}
		lout << H.info() << endl;
		lout << endl;
		
		auto PsiT = g.state;
		PsiT.max_Nsv = Dbetalimit;
		TDVPPropagator<MODEL,MODEL::Symmetry,double,double,MODEL::StateXd> TDVPT(H,PsiT);
		
		ofstream Filer(make_string(wd,"thermodynGC_",base,".dat"));
		Filer << "#T\tc\te\tchi" << endl;
		double beta = 0.;
		vector<double> cvec;
		vector<double> evec;
		vector<double> chivec;
		auto PsiTprev = PsiT;
		
		for (int i=0; i<Nbeta+1; ++i)
		{
			Stopwatch<> FullTimer;
			
			#pragma omp parallel sections
			{
				#pragma omp section
				{
					if (i!=Nbeta)
					{
						Stopwatch<> Stepper;
						if (i==0)
						{
							PsiT.eps_svd = 0.;
							PsiT.min_Nsv = 1ul;
						}
						else
						{
							PsiT.eps_svd = tol_beta_compr;
							PsiT.min_Nsv = 0ul;
						}
						
						if (PsiT.calc_Dmax() == Dbetalimit and i*dbeta>1.)
						{
							PsiT.eps_svd = 0.1*tol_beta_compr;
							TDVPT.t_step0(H, PsiT, -0.5*dbeta, N_stages, 1e-8);
						}
						else
						{
							TDVPT.t_step(H, PsiT, -0.5*dbeta, N_stages, 1e-8);
						}
						PsiT /= sqrt(dot(PsiT,PsiT));
						lout << "propagated to: β=" << (i+1)*dbeta << endl;
						lout << TDVPT.info() << endl;
						lout << setprecision(16) << PsiT.info() << setprecision(6) << endl;
						lout << Stepper.info("βstep") << endl;
					}
				}
				#pragma omp section
				{
					if (i>0)
					{
						double beta = i*dbeta;
						Stopwatch<> Stepper;
						//---------inner energy density---------
						double E = avg(PsiTprev,H,PsiTprev);
						double e = E/L;
						evec.push_back(e);
						lout << Stepper.info("e") << endl;
						//---------specific heat---------
						double c = std::nan("c");
						if (CALC_C)
						{
							c = (L==60)? beta*beta*(avg(PsiTprev,H,H,PsiTprev  )-pow(E,2))/L:
						                    beta*beta*(avg(PsiTprev,H,PsiTprev,2ul)-pow(E,2))/L;
						}
						cvec.push_back(c);
						lout << Stepper.info("c") << endl;
						//---------uniform magnetic susceptibility---------
						double chi;
						// slow way:
			//			double chi_ = 0.;
			//			for (int i=0; i<L; ++i)
			//			for (int j=0; j<L; ++j)
			//			{
			//				double res = avg(PsiTprev,H.SdagS(i,j),PsiTprev);
			//				chi_ += res; // note: pow(avg(PsiTprev,S(i),PsiTprev),2)=0
			//			}
			//			chi_ *= beta/L;
						// fast way:
			//			if (CALC_CHI)
			//			{
			//				chi = beta*(2.*avg(PsiTprev,Hchi,PsiTprev)/L+0.75); // S(S+1)=0.75: diagonal contribution
			//			}
						// best way:
						chi = beta*avg(PsiTprev,H.Sdagtot(0,sqrt(3.),dLphys),H.Stot(0,1.,dLphys),PsiTprev)/L;
						chivec.push_back(chi);
						//---------
						lout << Stepper.info("chi") << endl;
					}
				}
			}
			
			PsiTprev = PsiT;
			
			// entropy
			auto PsiTtmp = PsiT; PsiTtmp.entropy_skim(); lout << "S=" << PsiTtmp.entropy().transpose() << endl;
			
			if (i>0)
			{
				lout << termcolor::blue 
				     << "β=" << i*dbeta << ", T=" << 1./(i*dbeta) 
				     << ", e=" << evec[i-1] 
				     << ", c=" << cvec[i-1] 
				     << ", χ=" << chivec[i-1] 
				     << termcolor::reset
				     << endl;
				Filer << 1./(i*dbeta) << "\t" << cvec[i-1] << "\t" << evec[i-1] << "\t" << chivec[i-1] << endl;
			}
			
			lout << FullTimer.info("total") << endl;
			lout << endl;
		}
		
		Filer.close();
	}
}

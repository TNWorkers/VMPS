#ifdef BLAS
#include "util/LapackManager.h"
#pragma message("LapackManager")
#endif

#define USE_HDF5_STORAGE
#define DMRG_DONT_USE_OPENMP

//#define DMRG_CONTRACTLANDR_PARALLELIZE
//#define DMRG_PARALLELIZE_GRALF

//#define EIGEN_DONT_PARALLELIZE
//#define DMRG_PIVOT2_PARALLELIZE
//#define DMRG_PIVOT1_PARALLELIZE
//#define DMRG_SPLITAA_PARALLELIZE
//#define DMRG_CONTRACTAA_PARALLELIZE
////#define DMRG_PIVOTVECTOR_PARALLELIZE // problem with omp and complex scalar
//#define DMRG_PRECALCBLOCKTSD_PARALLELIZE

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

#include "solvers/DmrgSolver.h"
#include "solvers/MpsCompressor.h"
#include "DmrgLinearAlgebra.h"
#include "models/ParamCollection.h"
#include "EigenFiles.h"
#include "solvers/TDVPPropagator.h"
// for DOS:
//#include "solvers/GreenPropagator.h"
//using boost::math::quadrature::ooura_fourier_sin;
//using boost::math::quadrature::ooura_fourier_cos;
//#include <boost/math/quadrature/ooura_fourier_integrals.hpp>
//#include "InterpolGSL.h"
//#include "IntervalIterator.h"

#include "models/HeisenbergSU2.h"
typedef VMPS::HeisenbergSU2 MODEL;
#define USING_SU2
//#include "models/HeisenbergU1.h"
//typedef VMPS::HeisenbergU1 MODEL;
//#define USING_U1

/* 
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
 */

// returns i,j coordinates of bond indices at distance d
vector<pair<int,int>> bond_indices (int d, const ArrayXXi &distanceMatrix)
{
	vector<pair<int,int>> res;
	int L = distanceMatrix.rows();
	
	for (int j=0; j<L; ++j)
	for (int i=0; i<j; ++i)
	{
		if (distanceMatrix(i,j) == d)
		{
			res.push_back(make_pair(i,j));
		}
	}
	return res;
}

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

void calc_corr (const MODEL &H, const MODEL::StateXd &Psi, int S, string base, string wd, int L, int dmin, int dmax, const ArrayXXi &distanceMatrix)
{
	lout << "correlations for following state:" << endl;
	lout << Psi.info() << endl;
	
	if (S>0)
	{
		ofstream SFiler(make_string(wd,"S_",base,".dat"));
		SFiler << "#" << Psi.info() << endl;
		#pragma omp parallel for
		for (int l=0; l<L; ++l)
		{
			double res = 0;
			#ifdef USING_SU2
			{
				res = avg(Psi, H.S(l), Psi) / sqrt(S*(S+1.));
			}
			#elif defined(USING_U1)
			{
				res = avg(Psi, H.Sz(l), Psi) / sqrt(S*(S+1.));
			}
			#endif
			#pragma omp critical
			{
				SFiler << setprecision(16) << l << "\t" << res << setprecision(6) << endl;
				lout << setprecision(16) << l << "\t" << res << setprecision(6) << endl;
			}
		}
		SFiler.close();
	}
	
	ofstream CorrFilerAll(make_string(wd,"SdagS_",base,"_d=all",".dat"));
	CorrFilerAll << "#" << Psi.info() << endl;
	CorrFilerAll << "#i\tj\tSdagS" << endl;
	
	Stopwatch<> Timer;
	for (int d=dmin; d<=min(distanceMatrix.maxCoeff(),dmax); ++d)
	{
		lout << "correlations at distance d=" << d << endl;
		ofstream CorrFiler(make_string(wd,"SdagS_",base,"_d=",d,".dat"));
		CorrFiler << "#" << Psi.info() << endl;
		CorrFiler << "#i\tj\td\tSdagS" << endl;
		
		vector<pair<int,int>> indices = bond_indices(d,distanceMatrix);
		
		#pragma omp parallel for
		for (int k=0; k<indices.size(); ++k)
		{
			int i = indices[k].first;
			int j = indices[k].second;
			double val = avg(Psi, H.SdagS(i,j), Psi);
			
			#pragma omp critical
			{
				lout << setprecision(16) << "i=" << i << ", j=" << j << ", d=" << d << ", SdagS=" << val << setprecision(6) << endl;
				CorrFiler << setprecision(16) << i << "\t" << j << "\t" << val << setprecision(6) << endl;
				CorrFiler.flush();
				CorrFilerAll << setprecision(16) << i << "\t" << j << "\t" << d << "\t" << val << setprecision(6) << endl;
				CorrFilerAll.flush();
			}
		}
		
		CorrFiler.close();
		lout << Timer.info(make_string("d=",d)) << endl;
	}
	CorrFilerAll.close();
}

void calc_var (const MODEL &H, const Eigenstate<MODEL::StateXd> &Psi, string LOAD, size_t maxPower, int L, string base, string wd, string label)
{
	ofstream VarFiler(make_string(wd,"var_",base,".dat"));
	lout << label << ":" << endl;
	VarFiler << "#" << label << endl;
	VarFiler << "#" << Psi.state.info() << endl;
	
	Stopwatch<> Timer;
	double E = (LOAD!="")? avg(Psi.state,H,Psi.state):Psi.energy;
	lout << setprecision(16) << "E=" << E << setprecision(6) << endl;
	VarFiler << setprecision(16) << "E=" << E << setprecision(6) << endl;
	lout << Timer.info("E") << endl;
	VarFiler << Timer.info("E") << endl;
	
	if (maxPower == 1)
	{
		double var = abs(avg(Psi.state,H,H,Psi.state)-pow(E,2))/L;
		lout << setprecision(16) << "varE=" << var << setprecision(6) << endl;
		VarFiler << setprecision(16) << "varE=" << var << setprecision(6) << endl;
	}
	else
	{
		double var = abs(avg(Psi.state,H,Psi.state,2)-pow(E,2))/L;
		lout << setprecision(16) << "varE=" << var << setprecision(6) << endl;
		VarFiler << setprecision(16) << "varE=" << var << setprecision(6) << endl;
	}
	lout << Timer.info("varE") << endl;
	lout << endl;
	VarFiler.close();
	
//	auto HmE = H;
//	double factor = args.get<double>("factor",1.);
//	double offset = args.get<double>("offset",0.);
//	HmE.scale(factor,offset);
//	lout << "scale test: " << avg(Psi.state,HmE,Psi.state) << "\t" << factor*E+offset << "\t" << pow(factor,L)*E+pow(offset,L) << endl;
}

map<string,int> make_Lmap()
{
	map<string,int> m;
	//
	m["CHAIN"] = 0; // chain
	m["RING"] = 0; // ring
	// Platonic solids:
	m["P04"] = 4; // tetrahedron
	m["P06"] = 6; // octahedron
	m["P08"] = 8; // cube
	m["P12"] = 12; // icosahedron
	m["P20"] = 20; // dodecahedron
	// Traingles:
	m["T03"] = 3;
	m["T05"] = 5;
	m["T06"] = 6;
	// Fullerenes:
	m["C12"] = 12; // =truncated tetrahedron ATT
	m["C20"] = 20; // =dodecahedron P20
	m["C24"] = 24;
	m["C26"] = 26;
	m["C30"] = 30;
	m["C40"] = 40;
	m["C60"] = 60;
	// Archimedean solids:
	m["ATT"] = 12; // truncated tetrahedron
	m["ACO"] = 12; // cuboctahedron
	m["ATO"] = 24; // truncated octahedron
	m["AID"] = 30; // icosidodecahedron
	m["ASD"] = 60; // snub dodecahedron
	// sodalite cages:
	m["SOD15"] = 15; // NOT IMPLEMENTED
	m["SOD20"] = 20; // cuboctahedron decorated with P04
	m["SOD32"] = 32; // NOT IMPLEMENTED
	m["SOD50"] = 50; // icosidodecahedron decorated with P04
	m["SOD60"] = 60; // rectified truncated octahedron decorated with P04
	return m;
}

map<string,string> make_vertexMap()
{
	map<string,string> m;
	
	m["ATT"] = "3.6^2";
	m["ACO"] = "3.4.3.4";
	m["ATO"] = "4.6^2";
	m["AID"] = "3.5.3.5";
	m["ASD"] = "3^4.5";
	
	return m;
}

/////////////////////////////////
int main (int argc, char* argv[])
{
	Eigen::initParallel();
	
	ArgParser args(argc,argv);
	int L = args.get<int>("L",60);
	double t = args.get<double>("t",1.);
	double U = args.get<double>("U",0.);
	double J = args.get<double>("J",1.);
	double Jprime = args.get<double>("Jprime",0.);
	double Bz = args.get<double>("Bz",0.);
	int S = args.get<int>("S",0);
	int M = args.get<int>("M",0);
	size_t D = args.get<size_t>("D",2ul);
	size_t maxPower = args.get<size_t>("maxPower",2ul);
	
	bool PRINT_HOPPING = args.get<bool>("PRINT_HOPPING",false);
	bool PRINT_FREE = args.get<bool>("PRINT_FREE",false);
	if constexpr (MODEL::FAMILY == HUBBARD) PRINT_FREE = true;
	string MOL = args.get<string>("MOL","C60");
	int VARIANT = args.get<int>("VARIANT",0); // to try different enumeration variants
	map<string,int> Lmap = make_Lmap();
	map<string,string> Vmap = make_vertexMap();
	if (MOL!="CHAIN" and MOL!="RING") L = Lmap[MOL]; // for linear chain, include chain length using -L
	int N = args.get<int>("N",L);
	
	bool BETAPROP = args.get<bool>("BETAPROP",false);
	bool BETA1STEP = args.get<bool>("BETA1STEP",false);
//	bool CANONICAL = args.get<bool>("CANONICAL",false);
	bool CALC_C = args.get<bool>("CALC_C",false);
	bool CALC_CHI = args.get<bool>("CALC_CHI",true);
	double dbeta = args.get<double>("dbeta",0.1);
	double betamax = args.get<double>("betamax",50.);
	double betainit = 0.;
	double betaswitch = args.get<double>("betaswitch",10.);
	double s_betainit = args.get<double>("s_betainit",log(2));
	double tol_beta_compr = args.get<double>("tol_beta_compr",1e-7);
	size_t Mlim = args.get<size_t>("Mlim",1000ul);
	size_t Ly = args.get<size_t>("Ly",2ul);
	int dLphys = (Ly==2ul)? 1:2;
	int N_stages = args.get<int>("N_stages",1);
	
	string LOAD = args.get<string>("LOAD","");
	int Nexc = args.get<int>("Nexc",0);
	int ninit = args.get<int>("ninit",0); // ninit=0: start with 1st excited state, ninit=1: start with 2nd excited state etc.
	vector<string> LOAD_EXCITED = args.get_list<string>("LOAD_EXCITED",{});
	double Epenalty = args.get<double>("Epenalty",1e4);
	bool CALC_CORR = args.get<bool>("CALC_CORR",true);
	bool CALC_CORR_EXCITED = args.get<bool>("CALC_CORR_EXCITED",false);
	bool CALC_GS = args.get<int>("CALC_GS",true);
	bool CALC_VAR = args.get<bool>("CALC_VAR",true);
	bool CALC_VAR_EXCITED = args.get<bool>("CALC_VAR_EXCITED",true);
	bool INIT_EXCITED_RANDOM = args.get<bool>("INIT_EXCITED_RANDOM",true);
	
	int dmax = args.get<int>("dmax",9);
	int dmin = args.get<int>("dmin",1);
	
	bool CALC_DOS = args.get<int>("CALC_DOS",false);
	double dt = args.get<double>("dt",0.05);
	double tmax = args.get<double>("tmax",5.);
	int Nt = static_cast<int>(tmax/dt);
	size_t Dtlimit = args.get<size_t>("Dtlimit",150ul);
	double tol_t_compr = args.get<double>("tol_t_compr",1e-4);
	int x0 = args.get<int>("x0",0);
	
	string wd = args.get<string>("wd","./");
	if (wd.back() != '/') {wd += "/";}
	
	size_t Mlimit_default = 500ul; // Extracted from LOAD, but can be overwritten by -Mlimit option.
	
	// overwrite beta params in case of LOAD
	if (LOAD!="" and BETAPROP==true)
	{
		vector<string> parsed_params;
		boost::split(parsed_params, LOAD, [](char c){return c == '_';});
		for (int i=0; i<parsed_params.size(); ++i)
		{
			vector<string> parsed_vals;
			boost::split(parsed_vals, parsed_params[i], [](char c){return c == '=';});
			for (int j=0; j<parsed_vals.size(); ++j)
			{
				if (parsed_vals[j] == "β")
				{
					betainit = boost::lexical_cast<double>(parsed_vals[j+1]);
					lout << "extracted: betainit=" << betainit << endl;
				}
				else if (parsed_vals[j] == "dβ")
				{
					dbeta = boost::lexical_cast<double>(parsed_vals[j+1]);
					lout << "extracted: dbeta=" << dbeta << endl;
				}
				else if (parsed_vals[j] == "Mlim")
				{
					Mlim = boost::lexical_cast<int>(parsed_vals[j+1]);
					lout << "extracted: Mlim=" << Mlim << endl;
				}
				else if (parsed_vals[j] == "tol")
				{
					tol_beta_compr = boost::lexical_cast<double>(parsed_vals[j+1]);
					lout << "extracted: tol_beta_compr=" << tol_beta_compr << endl;
				}
			}
		}
	}
	else if (LOAD!="" and BETAPROP==false)
	{
		vector<string> parsed_params;
		boost::split(parsed_params, LOAD, [](char c){return c == '_';});
		for (int i=0; i<parsed_params.size(); ++i)
		{
			vector<string> parsed_vals;
			boost::split(parsed_vals, parsed_params[i], [](char c){return c == '=';});
			for (int j=0; j<parsed_vals.size(); ++j)
			{
				if (parsed_vals[j] == "L")
				{
					L = boost::lexical_cast<int>(parsed_vals[j+1]);
					lout << "extracted: L=" << L << endl;
				}
				else if (parsed_vals[j] == "S")
				{
					S = boost::lexical_cast<int>(parsed_vals[j+1]);
					lout << "extracted: S=" << S << endl;
				}
				else if (parsed_vals[j] == "M")
				{
					M = boost::lexical_cast<int>(parsed_vals[j+1]);
					lout << "extracted: M=" << M << endl;
				}
				else if (parsed_vals[j] == "U")
				{
					U = boost::lexical_cast<int>(parsed_vals[j+1]);
					lout << "extracted: U=" << U << endl;
				}
				if (parsed_vals[j] == "Mlimit")
				{
					Mlimit_default = boost::lexical_cast<int>(parsed_vals[j+1]);
					lout << "extracted: Mlimit_default=" << Mlimit_default << endl;
				}
			}
		}
	}
	
	string base;
	if constexpr (MODEL::FAMILY == HUBBARD)
	{
		base = make_string("L=",L,"_N=",N,"_S=",S,"_U=",U);
	}
	else
	{
		base = make_string("L=",L,"_D=",D);
		#ifdef USING_SU2
		{
			base += make_string("_S=",S);
		}
		#elif defined(#ifdef USING_U1)
		{
			base += make_string("_M=",M);
		}
		#endif
	}
	if (MOL=="C60" and VARIANT==0)
	{
		base += make_string("_MOL=","C60CMK");
	}
	else
	{
		base += make_string("_MOL=",MOL);
	}
//	if (CALC_DOS)
//	{
//		base += make_string("_dt=",dt,"_tmax=",tmax,"_tol=",tol_t_compr);
//	}
	if (BETAPROP)
	{
		base += make_string("_Ly=",Ly,"_dbeta=",dbeta,"_tol=",tol_beta_compr,"_Mlim=",Mlim);
	}
//	string base_excited = base;
//	if (CALC_NEUTRAL_GAP) base_excited += make_string("_Epenalty=",Epenalty);
	
	DMRG::VERBOSITY::OPTION VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",DMRG::VERBOSITY::HALFSWEEPWISE));
	
	// dyn. params
	DMRG::CONTROL::DYN  DynParam;
	int max_Nrich = args.get<int>("max_Nrich",-1);
	DynParam.max_Nrich = [max_Nrich] (size_t i) {return max_Nrich;};
	
	size_t Mincr_per = args.get<size_t>("Mincr_per",4ul);
	DynParam.Mincr_per = [Mincr_per,LOAD] (size_t i) {return (i==0 and LOAD!="")? 0:Mincr_per;}; // if LOAD, resize before first step
	
	size_t Mincr_abs = args.get<size_t>("Mincr_abs",300ul);
	DynParam.Mincr_abs = [Mincr_abs] (size_t i) {return Mincr_abs;};
	
	size_t start_2site = args.get<size_t>("start_2site",0ul);
	size_t end_2site = args.get<size_t>("end_2site",20ul);
	DynParam.iteration = [start_2site,end_2site] (size_t i) {return (i>=start_2site and i<=end_2site and i%2==0)? DMRG::ITERATION::TWO_SITE : DMRG::ITERATION::ONE_SITE;};
	
	double eps_svd = args.get<double>("eps_svd",1e-10);
	DynParam.eps_svd = [eps_svd] (size_t i) {return eps_svd;};
	
	// glob. params
	DMRG::CONTROL::GLOB GlobParam;
	GlobParam.Mlimit = args.get<size_t>("Mlimit",Mlimit_default); // for groundstate
	GlobParam.min_halfsweeps = args.get<size_t>("min_halfsweeps",Mincr_per*GlobParam.Mlimit/(Mincr_abs)+Mincr_per);
	GlobParam.max_halfsweeps = args.get<size_t>("max_halfsweeps",GlobParam.min_halfsweeps);
	GlobParam.Minit = args.get<size_t>("Minit",2ul);
	GlobParam.Qinit = args.get<size_t>("Qinit",2ul);
	GlobParam.CONVTEST = DMRG::CONVTEST::VAR_2SITE; // DMRG::CONVTEST::VAR_HSQ
	GlobParam.CALC_S_ON_EXIT = false;
	if (!BETAPROP) base += make_string("_Mlimit=",GlobParam.Mlimit);
	GlobParam.savePeriod = args.get<size_t>("savePeriod",2);
	GlobParam.saveName = make_string(wd,MODEL::FAMILY,"_",base);
	
	// alpha
	size_t start_alpha = args.get<size_t>("start_alpha",0);
	size_t end_alpha = args.get<size_t>("end_alpha",GlobParam.max_halfsweeps-Mincr_per+1ul);
	double alpha = args.get<double>("alpha",100.);
	DynParam.max_alpha_rsvd = [start_alpha, end_alpha, alpha] (size_t i) {return (i>=start_alpha and i<end_alpha)? alpha:0.;};
	
	lout.set(make_string(base,"_Mlimit=",GlobParam.Mlimit,".log"), wd+"log", true);
	
	lout << args.info() << endl;
	#ifdef _OPENMP
	omp_set_nested(1);
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	ArrayXXd hopping;
	if (MOL=="RING")
	{
		bool COMPRESSED = args.get<bool>("COMPRESSED",false);
		hopping = create_1D_PBC(L,J,Jprime,COMPRESSED); // Heisenberg ring for testing
	}
	else if (MOL=="CHAIN")
	{
		hopping = create_1D_OBC(L,J,Jprime); // Heisenberg chain for testing
	}
	else if (MOL.at(0) == 'P')
	{
		hopping = J*hopping_Platonic(L,VARIANT);
	}
	else if (MOL.at(0) == 'A')
	{
		hopping = J*hopping_Archimedean(Vmap[MOL],VARIANT);
	}
	else if (MOL.substr(0,3) == "SOD")
	{
		hopping = J*hopping_sodaliteCage(L,VARIANT);
	}
	else if (MOL.at(0)=='C')
	{
		hopping = J*hopping_fullerene(L,VARIANT);
	}
	else if (MOL.at(0)=='T')
	{
		hopping = J*hopping_triangular(L,VARIANT);
	}
	else
	{
		lout << "Unknown molecule!" << endl;
		throw;
	}
	
	// change hopping along the ring for C60
	bool RING = args.get<bool>("RING",false);
	double J2 = args.get<double>("J2",0.9);
	if (RING and VARIANT==0 and MOL=="C60")
	{
		vector<int> ringsites = {16,17,27,30,20, 21,31,38,28,29, 39,42,32,33,43, 35,25,24,34,26};
		for (int i=0; i<ringsites.size(); ++i)
		{
			hopping(ringsites[i],ringsites[(i+1)%20]) = J2;
			hopping(ringsites[(i+1)%20],ringsites[i]) = J2;
		}
	}
	
	/*bool PERMUTE = args.get<int>("PERMUTE",false);
	if (PERMUTE)
	{
		hopping = permute_random(hopping);
	}*/
	
	auto distanceMatrix = calc_distanceMatrix(hopping);
	if (PRINT_HOPPING)
	{
		lout << "adjacency:" << endl << hopping << endl << endl;
		lout << "distances:" << endl << distanceMatrix << endl << endl;
		for (int d=1; d<=distanceMatrix.maxCoeff(); ++d)
		{
			lout << "d=" << d << ", #bonds=" << bond_indices(d,distanceMatrix).size() << endl;
		}
		lout << endl;
	}
	
	// free fermions
	if (PRINT_FREE)
	{
		SelfAdjointEigenSolver<MatrixXd> Eugen(-1.*hopping.matrix());
		lout << Eugen.eigenvalues().transpose() << endl;
		VectorXd occ = Eugen.eigenvalues().head(N/2);
		VectorXd unocc = Eugen.eigenvalues().tail(L-N/2);
		lout << "orbital energies occupied:" << endl << occ.transpose()  << endl;
		lout << "orbital energies unoccupied:" << endl << unocc.transpose()  << endl << endl;
		double E0 = 2.*occ.sum();
		lout << setprecision(16) << "non-interacting fermions: E0=" << E0 << ", E0/L=" << E0/L << setprecision(6) << endl << endl;
	}
	
	//---------groundstate---------
	if (!BETAPROP)
	{
		vector<Param> params;
		qarray<MODEL::Symmetry::Nq> Q;
		if constexpr (MODEL::FAMILY == HUBBARD)
		{
			params.push_back({"U",U});
			params.push_back({"tFull",hopping});
			Q = {2*S+1,N};
		}
		else
		{
			params.push_back({"Jfull",hopping});
			params.push_back({"D",D});
			#ifdef USING_SU2
			{
				Q = {2*S+1};
			}
			#elif defined(USING_U1)
			{
				Q = {M};
				params.push_bacl({"Bz",Bz})
			}
			#endif
		}
		lout << "Q=" << Q << endl;
		params.push_back({"maxPower",maxPower});
		
		MODEL H(size_t(L),params);
		lout << H.info() << endl;
		
		Eigenstate<MODEL::StateXd> g;
		vector<Eigenstate<MODEL::StateXd>> excited(Nexc);
		MODEL::Solver DMRG(VERB);
		DMRG.userSetGlobParam();
		DMRG.userSetDynParam();
		DMRG.GlobParam = GlobParam;
		DMRG.DynParam = DynParam;
		
		if (LOAD!="")
		{
			g.state.load(LOAD,g.energy);
			lout << termcolor::blue << "loaded: " << g.state.info() << termcolor::reset << endl;
		}
		
		if (CALC_GS) DMRG.edgeState(H, g, Q, LANCZOS::EDGE::GROUND, (LOAD!="")?true:false);
		
		if (LOAD_EXCITED.size()>0)
		{
			excited.resize(LOAD_EXCITED.size());
			for (int n=0; n<LOAD_EXCITED.size(); ++n)
			{
				excited[n].state.load(LOAD_EXCITED[n],excited[n].energy);
				lout << "loaded excited: " << excited[n].state.info() << endl;
			}
		}
		
		if (Nexc>0)
		{
			lout << termcolor::blue << "CALC_GAP" << termcolor::reset << endl;
			lout << "ninit=" << ninit << ", Nexc=" << Nexc << endl;
			for (int n=ninit; n<Nexc; ++n)
			{
				lout << "------ n=" << n << " ------" << endl;
				GlobParam.saveName = make_string(wd,MODEL::FAMILY,"_n=",n,"_",base);
				MODEL::Solver DMRGe(VERB);
				DMRGe.Epenalty = Epenalty;
				lout << "Epenalty=" << DMRGe.Epenalty << endl;
				DMRGe.userSetGlobParam();
				DMRGe.userSetDynParam();
				DMRGe.GlobParam = GlobParam;
				DMRGe.DynParam = DynParam;
				DMRGe.push_back(g.state);
				for (int m=0; m<n; ++m)
				{
					DMRGe.push_back(excited[m].state);
				}
				
				if (LOAD_EXCITED.size() <= n)
				{
					excited[n].state = g.state;
					if (INIT_EXCITED_RANDOM) excited[n].state.setRandom();
					excited[n].state.sweep(0,DMRG::BROOM::QR);
					excited[n].state /= sqrt(dot(excited[n].state,excited[n].state));
				}
				excited[n].state.eps_svd = eps_svd;
				
				VectorXd overlaps(n+1);
				
				overlaps(0) = dot(g.state,excited[n].state);
				for (int m=0; m<n; ++m) overlaps(m+1) = dot(excited[m].state,excited[n].state);
				
				DMRGe.edgeState(H, excited[n], Q, LANCZOS::EDGE::GROUND, true);
				
				lout << endl;
				lout << "excited[" << n << "].energy=" << setprecision(16) << excited[n].energy << endl;
				
				overlaps(0) = dot(g.state,excited[n].state);
				for (int m=0; m<n; ++m) overlaps(m+1) = dot(excited[m].state,excited[n].state);
				lout << "final overlap=" << overlaps.transpose() << endl;
			}
		}
		if constexpr (MODEL::FAMILY == HEISENBERG)
		{
			if (CALC_CORR)
			{
				calc_corr(H, g.state, S, base, wd, L, dmin, dmax, distanceMatrix);
			}
			
			if (CALC_CORR_EXCITED and (Nexc>0 or LOAD_EXCITED.size()>0))
			{
				for (int n=0; n<excited.size(); ++n)
				{
					calc_corr(H, excited[n].state, S, make_string(base,"_n=",n), wd, L, dmin, dmax, distanceMatrix);
				}
				// --- implement average over all degenerate manifold here ---
			}
			
			if (CALC_VAR)
			{
				calc_var(H, g, LOAD, maxPower, L, base, wd, "ground state variance");
			}
			
			if (CALC_VAR_EXCITED and (Nexc>0 or LOAD_EXCITED.size()>0))
			{
				for (int n=0; n<excited.size(); ++n)
				{
					calc_var(H, excited[n], (LOAD_EXCITED.size()>0)?LOAD_EXCITED[n]:"", maxPower, L, base, wd, make_string("excited state n=",n," variance:"));
				}
			}
			
			//----density of states----
//			if (CALC_DOS)
//			{
//				std::array<double,2> tsign = {1.,-1.};
//				std::array<string,2> tlabel = {"forwards","back"};
//				
//				MODEL::StateXd Tmp;
//				Stopwatch<> Stepper;
//				OxV_exact(H.S(x0), g.state, Tmp, 2., DMRG::VERBOSITY::ON_EXIT);
//				lout << Stepper.info("OxV") << endl;
//				std::array<MODEL::StateXcd,2> Psi;
//				for (int t=0; t<2; ++t) Psi[t] = Tmp.cast<complex<double> >();
//				auto Phi = g.state.cast<complex<double> >();
//				
//				for (int t=0; t<2; ++t)
//				{
//					Psi[t].eps_svd = tol_t_compr;
//					Psi[t].max_Nsv = Tmp.calc_Mmax();
//				}
//				
//				std::array<TDVPPropagator<MODEL,MODEL::Symmetry,double,complex<double>,MODEL::StateXcd>,2> TDVPt;
//				for (int t=0; t<2; ++t)
//				{
//					TDVPt[t] = TDVPPropagator<MODEL,MODEL::Symmetry,double,complex<double>,MODEL::StateXcd>(H,Psi[t]);
//				}
//				MatrixXd Gloct(Nt+1,3); Gloct.setZero();
//				complex<double> res = -1.i * dot(Psi[1],Psi[0]);
//				Gloct(0,0) = 0;
//				Gloct(0,1) = res.real();
//				Gloct(0,2) = res.imag();
//				string filet = wd+"Gloct_"+base+".dat";
//				string filew = wd+"DOS_"+base+".dat";
//				
//				for (int i=0; i<Nt; ++i)
//				{
//					Stopwatch<> Stepper;
//					
////					#pragma omp parallel for
//					for (int t=0; t<2; ++t)
//					{
//						TDVPt[t].t_step(H, Psi[t], -0.5*1.i*tsign[t]*dt);
//						
//						if (Psi[t].get_truncWeight().sum() > 0.5*tol_t_compr)
//						{
//							Psi[t].max_Nsv = min(static_cast<size_t>(max(Psi[t].max_Nsv*1.1, Psi[t].max_Nsv+1.)),Dtlimit);
////							#pragma omp critical
//							{
//								lout << termcolor::yellow << "Setting Psi["<<t<<"].max_Nsv to " << Psi[t].max_Nsv << termcolor::reset << endl;
//							}
//						}
//						
////						#pragma omp critical
//						{
//							lout << tlabel[t] << ":" << endl;
//							lout << "\t" << TDVPt[t].info() << endl;
//							lout << "\t" << Psi[t].info() << endl;
//						}
//					}
//					
//					lout << Stepper.info("t-step") << endl;
//					
//					double tval = (i+1)*dt;
//					complex<double> phase = -1.i*exp(1.i*g.energy*tval);
//					complex<double> res = phase * dot(Psi[1],Psi[0]);
//					Gloct(i+1,0) = tval;
//					Gloct(i+1,1) = res.real();
//					Gloct(i+1,2) = res.imag();
//					saveMatrix(Gloct,filet,false);
//					lout << "propagated to t=" << tval << endl;
//					lout << endl;
//				}
//				
//				for (int t=0; t<2; ++t)
//				{
//					lout << "t=" << t << endl;
//					string filename = make_string(wd,"state_t=",Nt*dt,"_tdir=",t,"_",base);
//					lout << termcolor::green << "saving state to: " << filename << termcolor::reset << endl;
//					Psi[t].save(filename);
//				}
//				
//				// test for save:
////				for (int t=0; t<2; ++t)
////				{
////					string filename = make_string(wd,"state_t=",Nt*dt,"_tdir=",t,"_",base,".dat");
////					cout << "loading: " << filename << endl;
////					MODEL::StateXcd test;
////					test.load(filename);
////					cout << test.info() << endl;
////				}
//				
////				GreenPropagatorGlobal::tmax = tmax;
////				GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double>> Green(filew, tmax, Nt, -5., +10., 501, MPI_PPI, 501, INTERP);
////				Green.set_verbosity(DMRG::VERBOSITY::HALFSWEEPWISE);
////				IntervalIterator w(-5.,10.,501);
////				ArrayXd wvals = w.get_abscissa();
////				ArrayXcd DOS = Green.FTloc_tw(VectorXcd(Gloct.col(1)+1.i*Gloct.col(2)), wvals);
////				for (w=w.begin(); w!=w.end(); ++w)
////				{
////					w << -M_1_PI * DOS(w.index()).imag();
////				}
////				w.save(filew);
//				
//				VectorXd tvals(Nt+1); for (int i=0; i<Nt+1; ++i) tvals(i) = i*dt;
////				cout << tvals.transpose() << endl;
//				ooura_fourier_sin<double> OouraSin = ooura_fourier_sin<double>();
//				ooura_fourier_cos<double> OouraCos = ooura_fourier_cos<double>();
//				Interpol<GSL> Gloct_interpRe(tvals);
//				Interpol<GSL> Gloct_interpIm(tvals);
//				for (int it=0; it<tvals.rows(); ++it)
//				{
//					Gloct_interpRe.insert(it,Gloct(it,1));
//					Gloct_interpIm.insert(it,Gloct(it,2));
//				}
//				Gloct_interpRe.set_splines();
//				Gloct_interpIm.set_splines();
//				auto fRe = [&Gloct_interpRe, &tmax](double t) {return (t<=tmax)? Gloct_interpRe(t)*exp(-pow(2.*t/tmax,2)):0.;};
//				auto fIm = [&Gloct_interpIm, &tmax](double t) {return (t<=tmax)? Gloct_interpIm(t)*exp(-pow(2.*t/tmax,2)):0.;};
//				
//				double resReSin, resImCos;
//				IntervalIterator w(-5.,10.,1001);
//				for (w=w.begin(); w!=w.end(); ++w)
//				{
//					double wval = *w;
//					double Glocw;
//					if (wval == 0.)
//					{
//						Glocw = Gloct_interpIm.integrate();
//					}
//					else
//					{
//						resReSin = OouraSin.integrate(fRe,wval).first;
//						resImCos = OouraCos.integrate(fIm,wval).first;
//						Glocw = -M_1_PI * (resReSin+resImCos);
//					}
//					w << Glocw;
//				}
//				w.save(filew);
//				Gloct_interpRe.kill_splines();
//				Gloct_interpIm.kill_splines();
//			}
		}
	}
	//-------canonical-------
//	else if (BETAPROP and CANONICAL)
//	{
//		assert(MODEL::FAMILY == HEISENBERG);
//		
//		vector<Param> beta0_params;
//		MODELC H0;
//		ArrayXXd AllConnected(L,L); AllConnected=-1.; AllConnected.matrix().diagonal().setZero();
//		beta0_params.push_back({"Kfull",AllConnected});
//		beta0_params.push_back({"maxPower",2ul});
//		H0 = MODELC(L,beta0_params);
//		lout << H0.info() << endl;
//		
//		Eigenstate<MODELC::StateXd> g;
//		MODELC::Solver DMRG(VERB);
//		DMRG.edgeState(H0, g, MODELC::singlet(), LANCZOS::EDGE::GROUND, false);
//		
//		// test entanglement
//		MatrixXd SdagSphys_Tinf(L,L); SdagSphys_Tinf.setZero();
//		MatrixXd SdagSancl_Tinf(L,L); SdagSancl_Tinf.setZero();
//		#pragma omp parallel for collapse(2)
//		for (int j=0; j<L; ++j)
//		for (int i=0; i<L; ++i)
//		{
//			SdagSphys_Tinf(i,j) = avg(g.state, H0.SdagS<0>(i,j), g.state);
//			SdagSancl_Tinf(i,j) = avg(g.state, H0.SdagS<1>(i,j), g.state);
//		}
//		lout << "entanglement test physical spins:" << endl;
//		lout << SdagSphys_Tinf << endl << endl;
//		lout << "entanglement test ancillary spins:" << endl;
//		lout << SdagSancl_Tinf << endl << endl;
//		
//		vector<Param> beta_params;
//		if (CALC_C)
//		{
//			beta_params.push_back({"maxPower",(L==60)?1ul:2ul});
//		}
//		else
//		{
//			beta_params.push_back({"maxPower",1ul});
//		}
//		beta_params.push_back({"J1full",hopping});
//		MODELC H(size_t(L),beta_params);
//		lout << H.info() << endl;
//		lout << endl;
//		
//		auto PsiT = g.state;
//		PsiT.max_Nsv = Mlim;
//		TDVPPropagator<MODELC,MODELC::Symmetry,double,double,MODELC::StateXd> TDVPT(H,PsiT);
//		
//		ofstream Filer(make_string(wd,"thermodynC_",base,".dat"));
//		Filer << "#T\tc\te\tchi" << endl;
//		double beta = 0.;
//		vector<double> cvec;
//		vector<double> evec;
//		vector<double> chivec;
//		auto PsiTprev = PsiT;
//		
//		for (int i=0; i<Nbeta+1; ++i)
//		{
//			Stopwatch<> FullTimer;
//			
//			#pragma omp parallel sections
//			{
//				#pragma omp section
//				{
//					if (i!=Nbeta)
//					{
//						Stopwatch<> Stepper;
////						if (i==0)
////						{
////							PsiT.eps_svd = 0.;
////							PsiT.min_Nsv = 1ul;
////						}
////						else
////						{
//							PsiT.eps_svd = tol_beta_compr;
//							PsiT.min_Nsv = 0ul;
////						}
//						
//						if (PsiT.calc_Mmax() == Mlim and i*dbeta>1.)
//						{
//							PsiT.eps_svd = 0.1*tol_beta_compr;
//							TDVPT.t_step0(H, PsiT, -0.5*dbeta, N_stages, 1e-8);
//						}
//						else
//						{
//							TDVPT.t_step(H, PsiT, -0.5*dbeta, N_stages, 1e-8);
//						}
//						PsiT /= sqrt(dot(PsiT,PsiT));
//						lout << "propagated to: β=" << (i+1)*dbeta << endl;
//						lout << TDVPT.info() << endl;
//						lout << setprecision(16) << PsiT.info() << setprecision(6) << endl;
//						lout << Stepper.info("βstep") << endl;
//					}
//				}
//				#pragma omp section
//				{
//					if (i>0)
//					{
//						double beta = i*dbeta;
//						Stopwatch<> Stepper;
//						//---------inner energy density---------
//						double E = avg(PsiTprev,H,PsiTprev);
//						double e = E/L;
//						evec.push_back(e);
//						lout << Stepper.info("e") << endl;
//						//---------specific heat---------
//						double c = std::nan("c");
//						if (CALC_C)
//						{
//							c = (L==60)? beta*beta*(avg(PsiTprev,H,H,PsiTprev  )-pow(E,2))/L:
//						                 beta*beta*(avg(PsiTprev,H,PsiTprev,2ul)-pow(E,2))/L;
//						}
//						cvec.push_back(c);
//						lout << Stepper.info("c") << endl;
//						//---------uniform magnetic susceptibility---------
//						double chi = beta*avg(PsiTprev,H.Sdagtot<0>(0,sqrt(3.)),H.Stot<0>(0,1.),PsiTprev)/L;
//						chivec.push_back(chi);
//						//---------
//						lout << Stepper.info("chi") << endl;
//					}
//				}
//			}
//			
//			PsiTprev = PsiT;
//			
//			// entropy
//			auto PsiTtmp = PsiT; PsiTtmp.entropy_skim(); lout << "S=" << PsiTtmp.entropy().transpose() << endl;
//			
//			if (i>0)
//			{
//				lout << termcolor::blue 
//				     << setprecision(16)
//				     << "β=" << i*dbeta << ", T=" << 1./(i*dbeta) 
//				     << ", e=" << evec[i-1] 
//				     << ", c=" << cvec[i-1] 
//				     << ", χ=" << chivec[i-1] 
//				     << termcolor::reset
//				     << setprecision(6)
//				     << endl;
//				Filer << setprecision(16) << 1./(i*dbeta) << "\t" << cvec[i-1] << "\t" << evec[i-1] << "\t" << chivec[i-1] << setprecision(6) << endl;
//			}
//			
//			lout << FullTimer.info("total") << endl;
//			lout << endl;
//		}
//		
//		Filer.close();
//	}
	//-------grand canonical-------
	else
	{
		assert(MODEL::FAMILY == HEISENBERG);
		
		MODEL::StateXd PsiT;
		
		if (LOAD != "")
		{
			PsiT.load(LOAD);
			lout << "loaded: " << PsiT.info() << endl;
			lout << termcolor::blue << "continuing β-propagation at β=" << betainit << " with: " 
			     << "dβ=" << dbeta << ", "
			     << "Mlim=" << Mlim << ", "
			     << "tol=" << tol_beta_compr << ", "
			     << boolalpha << "BETA1STEP=" << BETA1STEP
			     << termcolor::reset << endl;
		}
		else
		{
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
			DMRG::CONTROL::GLOB GlobParam;
			GlobParam.CALC_S_ON_EXIT = false;
			DMRG.userSetGlobParam();
			DMRG.GlobParam = GlobParam;
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
			
			PsiT = g.state;
			
			lout << endl;
		}
		
		// construct H
		vector<Param> beta_params;
		beta_params.push_back({"Ly",Ly});
		beta_params.push_back({"J",0.});
		beta_params.push_back({"Jrung",0.});
		beta_params.push_back({"maxPower",maxPower});
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
		
		PsiT.max_Nsv = Mlim;
		if (LOAD!="") lout << "preparing TDVP..." << endl;
		TDVPPropagator<MODEL,MODEL::Symmetry,double,double,MODEL::StateXd> TDVPT(H,PsiT);
		lout << PsiT.info() << endl;
		
		ofstream Filer(make_string(wd,"thermodynGC_",base,".dat"));
		Filer << "#beta\tT\tc\te\tchi\ts" << endl;
		Filer.close();
		vector<double> cvec;
		vector<double> evec;
		vector<double> chivec;
		vector<double> svec;
		vector<double> lnZvec;
		auto PsiTprev = PsiT;
		
		vector<double> betasteps;
		vector<double> betavals;
		
		if (betainit < 1e-15)
		{
			betavals.push_back(0.01);
			betasteps.push_back(0.01);
			cout << "betaval=" << betavals[betavals.size()-1] << ", betastep=" << betasteps[betasteps.size()-1] << endl;
			
			for (int i=1; i<20; ++i)
			{
				betasteps.push_back(0.01);
				double beta_last = betavals[betavals.size()-1];
				betavals.push_back(beta_last+0.01);
				cout << "betaval=" << betavals[betavals.size()-1] << ", betastep=" << betasteps[betasteps.size()-1] << endl;
			}
		}
		else
		{
			assert(betainit>=0.2);
			betavals.push_back(betainit+dbeta);
			betasteps.push_back(dbeta);
			lout << "betaval=" << betavals[betavals.size()-1] << ", betastep=" << betasteps[betasteps.size()-1] << endl;
			
			// This makes: s_betainit = log(2) + accumulated ln(Z)
			// From here additional contributions in ln(Z) are added; and beta*e is added at each step
			double e = avg(PsiT,H,PsiT)/L;
			s_betainit -= betainit*e;
		}
		
		while (betavals[betavals.size()-1] < betamax)
		{
			betasteps.push_back(dbeta);
			double beta_last = betavals[betavals.size()-1];
			betavals.push_back(beta_last+dbeta);
			lout << "betaval=" << betavals[betavals.size()-1] << ", betastep=" << betasteps[betasteps.size()-1] << endl;
		}
		lout << endl;
//		betavals.pop_back();
//		betasteps.pop_back();
		
//		for (int i=0; i<betasteps.size(); ++i)
//		{
//			cout << "betaval=" << betavals[i] << ", betastep=" << betasteps[i] << endl;
//		}
		
//		ArrayXd beta_savepoints(99);
//		for (int i=0; i<beta_savepoints.rows(); ++i)
//		{
//			beta_savepoints(i) = i+1;
//		}
		vector<double> std_beta_savepoints = {1,2,3,4,5,6,7,8,9,10,25,50};
		ArrayXd beta_savepoints = ArrayXd::Map(std_beta_savepoints.data(), std_beta_savepoints.size());
		
		Stopwatch<> FullTimer;
		
		for (int i=0; i<betasteps.size()+1; ++i)
		{
			Stopwatch<> FullStepTimer;
			
			if (i!=betasteps.size())
			{
				double beta = betavals[i];
				Stopwatch<> Stepper;
				
				if (beta < 0.2)
				{
					PsiT.eps_svd = 1e-9;
					PsiT.min_Nsv = 1ul;
				}
				else
				{
					PsiT.eps_svd = tol_beta_compr;
					PsiT.min_Nsv = 0ul;
				}
				
				if (BETA1STEP or beta>=betaswitch)
				{
					TDVPT.t_step0(H, PsiT, -0.5*betasteps[i], N_stages, 1e-8);
				}
				else
				{
					TDVPT.t_step(H, PsiT, -0.5*betasteps[i], N_stages, 1e-8);
				}
				double norm = dot(PsiT,PsiT);
				lnZvec.push_back(log(norm));
				PsiT /= sqrt(norm);
				
				lout << "propagated to: β=" << beta << endl;
				
				if (((beta_savepoints-beta).abs() < 1e-10).any())
				{
					string filename = make_string(wd,"state_β=",beta,"_",base);
					lout << termcolor::green << "saving state to: " << filename << termcolor::reset << endl;
					PsiT.save(filename);
				}
				lout << TDVPT.info() << endl;
				lout << setprecision(16) << PsiT.info() << setprecision(6) << endl;
				lout << Stepper.info("βstep") << endl;
			}
			if (i>0)
			{
				double beta = betavals[i-1];
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
					c = (maxPower==1)? beta*beta*(avg(PsiTprev,H,H,PsiTprev  )-pow(E,2))/L:
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
				#ifdef USING_SU2
				{
					chi = beta*avg(PsiTprev,H.Sdagtot(0,sqrt(3.),dLphys),H.Stot(0,1.,dLphys),PsiTprev)/L;
				}
				#elif defined(#ifdef USING_U1)
				{
					//chi = beta*avg(PsiTprev,H.Sztot(0,1.,dLphys),H.Sztot(0,1.,dLphys),PsiTprev)/L;
					// Sztot NOT IMPLEMENTED
				}
				#endif
				chivec.push_back(chi);
				lout << Stepper.info("chi") << endl;
				
				//---------thermal entropy---------
				int Nsum = (i<betasteps.size())? lnZvec.size()-1:lnZvec.size();
				VectorXd tmp = VectorXd::Map(lnZvec.data(), Nsum);
				double s = s_betainit + tmp.sum()/L + beta*e;
				cout << "s_betainit=" << s_betainit << ", tmp.sum()/L=" << tmp.sum()/L << ", beta*e=" << beta*e << ", total=" << s << endl;
				svec.push_back(s);
				//---------
				lout << Stepper.info("s") << endl;
			}
			
			PsiTprev = PsiT;
			
			// entanglement entropy
//			Stopwatch<> EntropyWatch;
//			auto PsiTtmp = PsiT; PsiTtmp.entropy_skim(); lout << "S=" << PsiTtmp.entropy().transpose() << endl;
//			lout << EntropyWatch.info("entropy") << endl;
			
			if (i>0)
			{
				double beta = betavals[i-1];
				lout << termcolor::blue 
				     << setprecision(16)
				     << "β=" << beta << ", T=" << 1./beta
				     << ", e=" << evec[i-1] 
				     << ", c=" << cvec[i-1] 
				     << ", χ=" << chivec[i-1] 
				     << ", s=" << svec[i-1] 
				     << termcolor::reset
				     << setprecision(6)
				     << endl;
				Filer.open(make_string(wd,"thermodynGC_",base,".dat"), std::ios_base::app);
				Filer << setprecision(16) 
				      << beta << "\t"
				      << 1./beta << "\t" 
				      << cvec[i-1] << "\t" 
				      << evec[i-1] << "\t" 
				      << chivec[i-1] << "\t" 
				      << svec[i-1] 
				      << setprecision(6) << endl;
				Filer.close();
			}
			
			lout << FullStepTimer.info("full step") << endl;
			lout << FullTimer.info("total",false) << endl;
			lout << endl;
		}
		
//		Filer.close();
	}
}

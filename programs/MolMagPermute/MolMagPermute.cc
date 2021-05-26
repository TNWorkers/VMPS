#ifdef BLAS
#include "util/LapackManager.h"
#pragma message("LapackManager")
#endif

#define USE_HDF5_STORAGE
#define DMRG_DONT_USE_OPENMP
#define STATE_COMPRESS_M_INCREMENT 200
#define PROD_COMPRESS_M_INCREMENT 200
#define USE_OLD_COMPRESSION
#define OXV_EXACT_INIT_M 3000

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

#include "models/HeisenbergSU2.h"
typedef VMPS::HeisenbergSU2 MODEL;
#define USING_SU2
//#include "models/HeisenbergU1.h"
//typedef VMPS::HeisenbergU1 MODEL;
//#define USING_U1

#include "Permutations.h"

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
	string MOL = args.get<string>("MOL","SOD60");
	int VARIANT = args.get<int>("VARIANT",0); // to try different enumeration variants
	map<string,int> Lmap = make_Lmap();
	map<string,string> Vmap = make_vertexMap();
	if (MOL!="CHAIN" and MOL!="RING") L = Lmap[MOL]; // for linear chain, include chain length using -L
	int N = args.get<int>("N",L);
	
	bool BETA1STEP = args.get<bool>("BETA1STEP",false);
	bool CALC_C = args.get<bool>("CALC_C",false);
	bool CALC_C_HME = args.get<bool>("CALC_C_HME",false);
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
	if (LOAD!="")
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
				if (parsed_vals[j] == "MOL")
				{
					MOL = parsed_vals[j+1];
					lout << "extracted: MOL=" << MOL << endl;
				}
			}
		}
	}
	
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
	
        size_t Mlimit = args.get<size_t>("Mlimit",Mlimit_default);
        
        string ROT = args.get<string>("rot","h");

        string base;
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
	if (Jprime != 0.)
	{
		base += make_string("_Jprime=",Jprime);
	}
	if (MOL=="C60" and VARIANT==0)
	{
		base += make_string("_MOL=","C60CMK");
	}
	else
	{
		base += make_string("_MOL=",MOL);
	}

        lout.set(make_string(base,"_Mlimit=",Mlimit,"_ROT=",ROT,".log"), wd+"log", true);

	lout << args.info() << endl;
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif

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
	
	//----------------------------------------------------
	//----------- Code to perform the rotation -----------
	//----------------------------------------------------
	
	double tol_compr = args.get<double>("tol_compr",(MOL=="SOD20")?1e-7:1e-4);
	int max_halfsweeps = args.get<int>("max_halfsweeps",24);
	int min_halfsweeps = args.get<int>("min_halfsweeps",1);
	int div = args.get<int>("div",3);

	size_t Mincr = args.get<size_t>("Mincr",400ul);
	
	Eigenstate<MODEL::StateXd> g;
	g.state.load(LOAD,g.energy);
	if (g.state.get_pivot() == 1) g.state.sweep(0,DMRG::BROOM::QR);
	else if (g.state.get_pivot() == L-2) g.state.sweep(L-1,DMRG::BROOM::QR);
	lout << "LOADED: " << g.state.info() << endl;
	
	Permutation P(make_string(MOL,"_",ROT,"_90.dat"));
	std::vector<Transposition> T = P.transpositions();
	MODEL::Operator Ptot;
	int i=0;
	
	lout << P.print() << endl;
	lout << "----transpositions:----" << endl;
	for (const auto &t:T)
	{
		lout << t.source << "\t" << t.target << endl;
	}
	lout << "-----------------------" << endl;
	
	assert(T.size()%div == 0 and "Choose div such that the number of permutations is divisible by it!");
	
	MODEL::StateXd Psi1, Psi2;
	
	int Nchunks = T.size()/div;
	vector<vector<pair<int,int>>> Psets(Nchunks);
	
	for (int k=0; k<Nchunks; ++k)
	for (int j=0; j<div; ++j)
	{
		int index = j + div*k;
		Psets[k].push_back(make_pair(T[index].source,T[index].target));
	}
	
	vector<MODEL::Operator> Pop(Nchunks);
	vector<MODEL::Operator> PopDag(Nchunks);
	vector<MODEL::Operator> PopDagP(Nchunks);
	
	for (int k=0; k<Nchunks; ++k)
	{
		for (int j=0; j<div; ++j)
		{
			int index = j + div*k;
			lout << "k=" << k << ", j=" << j << endl;
			
			MODEL::Operator Pel = H.SdagS(Psets[k][j].first, Psets[k][j].second); Pel.scale(2.,0.5);
			
			if (j==0)
			{
				Pop[k] = Pel;
				PopDag[k] = Pel;
				lout << "set Op at k=" << k << ": " << Psets[k][j].first << "→" << Psets[k][j].second << endl;
			}
			else
			{
				Pop[k] = prod(Pel,Pop[k]);
				PopDag[k] = prod(PopDag[k],Pel);
				lout << "multiply Op from left at k=" << k << ": " << Psets[k][j].first << "→" << Psets[k][j].second << endl;
			}
		}
		lout << "P=" << Pop[k].info() << endl;
		PopDagP[k] = prod(PopDag[k],Pop[k]);
		lout << "P†P=" << PopDagP[k].info() << endl;
	}
	
	//OxV_exact(Pop[0], g.state, Psi1, tol_compr, DMRG::VERBOSITY::HALFSWEEPWISE, max_halfsweeps, min_halfsweeps);
	MpsCompressor<MODEL::Symmetry,double,double> Compadre(DMRG::VERBOSITY::HALFSWEEPWISE);
	Compadre.prodCompress(Pop[0], PopDag[0], g.state, Psi1, Q, g.state.calc_Mmax(), Mincr, Mlimit, tol_compr, max_halfsweeps, min_halfsweeps, 1, make_string(base,"_Mlimit=",Mlimit,"_ROT=",ROT), &PopDagP[0]);
	for (int k=1; k<Nchunks; ++k)
	{
//		OxV_exact(Pop[k], Psi1, Psi2, tol_compr, DMRG::VERBOSITY::HALFSWEEPWISE, max_halfsweeps, min_halfsweeps);
		MpsCompressor<MODEL::Symmetry,double,double> Compadre(DMRG::VERBOSITY::HALFSWEEPWISE);
		Compadre.prodCompress(Pop[k], PopDag[k], Psi1, Psi2, Q, g.state.calc_Mmax(), Mincr, Mlimit, tol_compr, max_halfsweeps, min_halfsweeps, 1ul, make_string(base,"_Mlimit=",Mlimit,"_ROT=",ROT),  &PopDagP[k]);
		Psi1 = Psi2;
	}
	
	Psi1.save(make_string(LOAD,"_ROT=",ROT,"_div=",div));
	lout << termcolor::green << "rotated wavefunction saved to: " << make_string(LOAD,"_ROT=",ROT,"_div=",div) << termcolor::reset << endl;
	lout << Psi1.info() << endl;
	
	lout << setprecision(16) << "<PHP>=" << avg(Psi1, H, Psi1) << ", E0=" << avg(g.state, H, g.state) << ", dot=" << g.state.dot(Psi1) << setprecision(6) << endl;
}

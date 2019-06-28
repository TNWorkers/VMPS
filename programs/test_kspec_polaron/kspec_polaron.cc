#include <iostream>
#include <fstream>
#include <complex>
#include <filesystem>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

#include "Stopwatch.h"
#include "PolychromaticConsole.h"

#include "../DMRG/solvers/DmrgSolver.h"
#include "../DMRG/solvers/MpsCompressor.h"
#include "../DMRG/DmrgLinearAlgebra.h"

#include "ArgParser.h"

#include "OrthPolyGreen.h"
#include "EigenFiles.h"

size_t L;
int N, M, D;
double t, U, J, Jdir;
vector<int> i0_list;
string spec, wd;
vector<double> dE;
vector<int> Msave;
string outfile;
int Mmax;
double Emin, Emax, E0;
double d, n_sig;
bool CHEB, SHIFT;
DMRG::VERBOSITY::OPTION VERB1, VERB2, VERB3;
typedef MODEL::Symmetry Symmetry;
int max_Nrich;

int main (int argc, char* argv[]) 
{
	#ifdef USE_FAST_WIG_SU2_COEFFS
	Sym::initialize(50,"../../table_50.3j","../../table_50.6j","../../hashed_20.9j");
	#endif
	
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",16);
	N = args.get<size_t>("N",4);
	M = args.get<size_t>("M",L-N);
	D = abs(M)+1; // D = 2S+1
//	qarray<2> Qi = MODEL::polaron(L,N);
	qarray<2> Qi;
	#ifdef USING_SU2
	Qi = {D,N};
	#else
	Qi = {M,N};
	#endif
	vector<qarray<2> > Qc;
	spec = args.get<string>("spec","IPES");
	t = args.get<double>("t",1.);
	U = args.get<double>("U",0.);
	J = args.get<double>("J",12.);
	Jdir = args.get<double>("Jdir",0.);
	CHEB = args.get<bool>("CHEB",true);
	SHIFT = args.get<bool>("SHIFT",true);
	wd = args.get<string>("wd","./");
	if (wd.back() != '/') {wd += "/";}
	SPIN_INDEX sigma = static_cast<SPIN_INDEX>(args.get<bool>("sigma",1)); // UP=0=false DN=1=true
	i0_list = args.get_list<int>("i0",{});
	if (i0_list.size() == 0)
	{
		for (int i0=0; i0<L/2; ++i0) i0_list.push_back(i0);
	}
	max_Nrich = args.get<int>("max_Nrich",40);
	dE = args.get_list<double>("dE",{0.5,0.2});
	
	VERB1 = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB1",DMRG::VERBOSITY::HALFSWEEPWISE));
	VERB2 = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB2",DMRG::VERBOSITY::HALFSWEEPWISE));
	VERB3 = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB3",DMRG::VERBOSITY::ON_EXIT));
	
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	if (VERB1 < static_cast<DMRG::VERBOSITY::OPTION>(2)) VERB1 = DMRG::VERBOSITY::HALFSWEEPWISE;
	if (VERB2 < static_cast<DMRG::VERBOSITY::OPTION>(2)) VERB2 = DMRG::VERBOSITY::HALFSWEEPWISE;
	if (VERB3 < static_cast<DMRG::VERBOSITY::OPTION>(2)) VERB3 = DMRG::VERBOSITY::HALFSWEEPWISE;
	lout << "not parallelized" << endl;
	#endif
	
	outfile = make_string(spec,"_L=",L,"_M=",M,"_N=",N,"_J=",J,"_U=",U);
	if (Jdir != 0.)
	{
		outfile += make_string("_Jdir=",Jdir);
	}
	#ifdef USING_U1
	if (spec == "PES" or spec == "IPES")
	{
		outfile += make_string("_sigma=",sigma);
	}
	#endif
	string EminmaxFile = outfile+".dat";
	lout.set(outfile+"_dE="+str(dE)+".log",wd+"log");
	lout << args.info() << endl;
	lout << "CHEB=" << boolalpha << CHEB << ", SHIFT=" << boolalpha << SHIFT << endl;
	lout << "wd: " << wd << endl;
	lout << "outfile: " << outfile << endl;
	lout << "EminmaxFile: " << EminmaxFile << endl;
	
	vector<Param> SweepParams;
	SweepParams.push_back({"max_Nrich",max_Nrich});
	SweepParams.push_back({"min_halfsweeps",1ul});
	#ifdef USING_SU2
	if (N==0)
	{
		SweepParams.push_back({"Qinit",100ul});
	}
	#endif
	
	//--------------<Hamiltonian>---------------
	MODEL H(L,{{"t",t},{"J",J},{"U",U},{"Jdir",Jdir},{"CALC_SQUARE",true}});
	lout << H.info() << endl << endl;
	//--------------</Hamiltonian>---------------
	
	#include "../DMRG/programs/snippets/As.txt"
	#include "../DMRG/programs/snippets/groundstate_load.txt"
//	#include "../DMRG/programs/snippets/state_compression.txt"
	#include "../DMRG/programs/snippets/AsxInit.txt"
	#include "../DMRG/programs/snippets/KPS.txt"
	#include "../DMRG/programs/snippets/ImAB.txt"
	
	#ifdef USE_FAST_WIG_SU2_COEFFS
	Sym::finalize(true);
	#endif
}

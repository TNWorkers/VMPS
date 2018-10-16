#include <iostream>
#include <fstream>
#include <complex>
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

size_t L;
int N, M, D;
double t, tPrime, U, J;
int i0;
vector<int> Msave;
int Mmax;
string spec, wd, outfile, Efilename;
vector<double> dE;
double Emin, Emax, E0;
double d, n_sig;
bool CHEB, SHIFT;
DMRG::VERBOSITY::OPTION VERB1, VERB2, VERB3;
typedef MODEL::Symmetry Symmetry;

int main (int argc, char* argv[]) 
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",20);
	N = args.get<size_t>("N",0);
	M = args.get<size_t>("M",L);
	D = abs(M)+1; // D = 2S+1
	qarray<2> Qi = MODEL::polaron(L,N);
	vector<qarray<2> > Qc;
	spec = args.get<string>("spec","APS");
	t = args.get<double>("t",1.);
	tPrime = args.get<double>("tPrime",0.);
	U = args.get<double>("U",0.);
	J = args.get<double>("J",3.);
	CHEB = args.get<bool>("CHEB",true);
	SHIFT = args.get<bool>("SHIFT",true);
	wd = args.get<string>("wd","./");
	if (wd.back() != '/') {wd += "/";}
	SPIN_INDEX sigma = static_cast<SPIN_INDEX>(args.get<bool>("sigma",1)); // UP=0 DN=1
	i0 = args.get<int>("i0",L/2);
	
	VERB1 = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB1",0));
	VERB2 = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB2",0));
	VERB3 = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB3",2));
	
	dE = args.get_list<double>("dE",{0.2});
	outfile = make_string(spec,"_L=",L,"_M=",M,"_N=",N,"_J=",J,"_U=",U);
	if (spec == "PES" or spec == "IPES")
	{
		outfile += make_string("_sigma=",sigma);
	}
	outfile += make_string("_i0=",i0);
	Efilename = outfile;
	outfile += "_dE=";
	lout.set(outfile+str(dE)+".log",wd+"log");
	lout << args.info() << endl;
	
	lout << outfile << endl;
	
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	//--------------<Hamiltonian>---------------
	MODEL H(L,{{"t",t},{"J",J},{"U",U},{"tPrime",tPrime},{"CALC_SQUARE",false}});
	lout << H.info() << endl << endl;
	//--------------</Hamiltonian>---------------
	
	#include "../DMRG/programs/snippets/As.txt"
	#include "../DMRG/programs/snippets/groundstate.txt"
//	#include "../DMRG/programs/snippets/state_compression.txt"
	#include "../DMRG/programs/snippets/AsxInit.txt"
	#include "../DMRG/programs/snippets/KPS.txt"
	#include "../DMRG/programs/snippets/ImAB.txt"
}

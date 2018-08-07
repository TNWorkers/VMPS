#include <iostream>
#include <fstream>
#include <complex>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

#include "Stopwatch.h"
#include "PolychromaticConsole.h"

#include "solvers/DmrgSolver.h"
#include "solvers/MpsCompressor.h"
#include "DmrgLinearAlgebra.h"

#include "ArgParser.h"
#include "OrthPolyGreen.h"

size_t L;
int N, M, S;
double t, U, V, tPrime;
vector<int> Msave;
int Mmax;
string spec, wd, outfile, Efilename;
vector<double> dE;
double Emin, Emax, E0;
double d, n_sig;
bool CHEB;

int main (int argc, char* argv[]) 
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",4);
	N = args.get<size_t>("N",L);
	M = args.get<size_t>("M",0);
	S = abs(M)+1;
	#ifdef USING_SU2
	qarray<2> Qi = {S,N};
	#else
	qarray<2> Qi = {M,N};
	#endif
	qarray<2> Qc;
	spec = args.get<string>("spec","AES");
	t = args.get<double>("t",1.);
	U = args.get<double>("U",6.);
	tPrime = args.get<double>("tPrime",0);
	V = args.get<double>("V",0.);
	CHEB = args.get<bool>("CHEB",true);
	wd = args.get<string>("wd","./");
	if (wd.back() != '/') {wd += "/";}
	constexpr SPIN_INDEX sigma = UP;
	
	dE = args.get_list<double>("dE",{0.2});
	outfile = make_string(spec,"_L=",L,"_M=",M,"_N=",N,"_U=",U);
	if (V != 0.) {outfile += make_string("_V=",V);}
	if (tPrime != 0.) {outfile += make_string("_tPrime=",tPrime);}
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
	MODEL H(L,{{"t",t},{"U",U},{"tPrime",tPrime},{"V",V},{"CALC_SQUARE",false}});
	lout << H.info() << endl << endl;
	//--------------</Hamiltonian>---------------
	
	#include "programs/snippets/A.txt"
	#include "programs/snippets/groundstate.txt"
//	#include "programs/snippets/state_compression.txt"
	#include "programs/snippets/AxInit.txt"
	#include "programs/snippets/KPS.txt"
}

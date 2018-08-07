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
double t, tPrime, U, J;
vector<int> Msave;
int Mmax;
string spec, wd, outfile, Efilename;
vector<double> dE;
double Emin, Emax, E0;
double d, n_sig;
bool CHEB;
typedef MODEL::Symmetry Symmetry;

int main (int argc, char* argv[]) 
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",10);
	N = args.get<size_t>("N",0);
	M = args.get<size_t>("M",L);
	S = abs(M)+1;
	qarray<2> Qi = MODEL::polaron(L,N);
	vector<qarray<2> > Qc;
	spec = args.get<string>("spec","IPES");
	t = args.get<double>("t",1.);
	tPrime = args.get<double>("tPrime",0.);
	U = args.get<double>("U",0.);
	J = args.get<double>("J",3.);
	CHEB = args.get<bool>("CHEB",true);
	wd = args.get<string>("wd","./");
	if (wd.back() != '/') {wd += "/";}
	constexpr SPIN_INDEX sigma = DN;
	
	dE = args.get_list<double>("dE",{0.2});
	outfile = make_string(spec,"_L=",L,"_M=",M,"_N=",N,"_J=",J,"_U=",U,"_sigma=",sigma);
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
	
	#include "programs/snippets/A.txt"
	#include "programs/snippets/groundstate.txt"
//	#include "programs/snippets/state_compression.txt"
	#include "programs/snippets/AxInit.txt"
	#include "programs/snippets/KPS.txt"
}

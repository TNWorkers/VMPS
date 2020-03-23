#define DMRG_DONT_USE_OPENMP
//#define PRINT_SU2_FACTORS

#include <iostream>

#include "termcolor.hpp"

#include "ArgParser.h"

#include "Logger.h"
Logger lout;

using namespace std;
#include "symmetry/S1xS2.h"
#include "symmetry/SU2.h"
#include "symmetry/U1.h"
#include "symmetry/U0.h"
#include "symmetry/kind_dummies.h"
#include "bases/SpinBase.h"

// #include "models/HubbardU1xU1.h"
// #include "models/HubbardSU2xU1.h"

// #include "models/KondoU1xU1.h"
// #include "models/KondoSU2xU1.h"

// #include "solvers/DmrgSolver.h"
// #include "solvers/MpsCompressor.h"

size_t L;
double U, J, tPrime;
int S,Sc,M,N,Nup,Ndn,D;
int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",4);
	size_t i = args.get<int>("i",0);
	size_t j = args.get<int>("j",0);
	N = args.get<int>("N",4);
	J = args.get<double>("J",-1.);
	tPrime = args.get<double>("tPrime",0.);
	U = args.get<double>("U",0.);
	S = args.get<int>("S",1);
	M = args.get<int>("M",0);
	Sc = args.get<int>("Sc",1);
	Nup = args.get<int>("Nup",L/2);
	Ndn = args.get<int>("Ndn",L/2);
	D = args.get<int>("D",1);
	
	bool OBSERVABLES = args.get<bool>("OBSERVABLES",true);

	typedef Sym::S1xS2<Sym::U1<Sym::SpinU1>, Sym::SU2<Sym::SpinSU2> > Symmetry;
	typedef Sym::S1xS2<Sym::U1<Sym::ChargeU1>,Sym::U1<Sym::SpinU1> > Symmetry2;
	typedef SpinBase<Symmetry> Base;
	typedef Base::OperatorType Op;
	vector<Base> B(2);
	B[0] = Base(L,D);
	
	SpinBase<Symmetry2> B2(L,D);

	cout << B[0].S() << endl;
	cout << B[0].Q(0) << endl;
	cout << B[0].Qdag(0) << endl;
	cout << B2.Sp() << endl;
	cout << B2.Sm() << endl;
	
	if (OBSERVABLES)
	{
		// qarray<2> Q1 = {S,N};

		// auto H = F.HubbardHamiltonian(U);
		// EDSolver<Op> John(H,{Q1},Eigen::DecompositionOptions::ComputeEigenvectors);
		// cout << "gse=" << John.eigenvalues().data().block[0](0,0) << endl;

		// Eigen::VectorXd cdagc(L); cdagc.setZero();
		// for(size_t i=0; i<L; i++)
		// {
		// 	auto cQ1 = Op::prod(F.c(i), John.groundstate(Q1), {2,-1});
		// 	auto res = Op::prod(cQ1.adjoint(), cQ1, {1,0});
		// 	cdagc(i) = res.data().block[0](0,0);
		// }
		// cout << "sqrt(2.)*cdagc" << endl << sqrt(2.)*cdagc << endl << endl;

		// Eigen::MatrixXd cc(L,L); cc.setZero();
		// for(size_t i=0; i<L; i++)
		// for(size_t j=0; j<L; j++)
		// {
		// 	auto cQ1i = Op::prod(F.c(i), John.groundstate(Q1), {2,-1});
		// 	auto cQ1j = Op::prod(F.c(j), John.groundstate(Q1), {2,-1});

		// 	auto res = Op::prod(cQ1i.adjoint(), cQ1j, {1,0});
		// 	cc(i,j) = res.data().block[0](0,0);
		// }
		// cout << "sqrt(2.)*cc" << endl << sqrt(2.)*cc << endl << endl;
	}
}

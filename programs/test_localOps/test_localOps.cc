#define DMRG_DONT_USE_OPENMP
//#define PRINT_SU2_FACTORS
#define USING_SU2
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
int S,Sc,M,N,Nup,Ndn;
size_t D;
int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",4);
	size_t i = args.get<int>("i",0);
	size_t j = args.get<int>("j",0);
	N = args.get<int>("N",4);
	J = args.get<double>("J",1.);
	double Jk = args.get<double>("Jk",1.);
	tPrime = args.get<double>("tPrime",0.);
	U = args.get<double>("U",0.);
	S = args.get<int>("S",1);
	M = args.get<int>("M",0);
	Sc = args.get<int>("Sc",1);
	Nup = args.get<int>("Nup",L/2);
	Ndn = args.get<int>("Ndn",L/2);
	D = args.get<size_t>("D",2);
	
	bool OBSERVABLES = args.get<bool>("OBSERVABLES",true);
	bool PERIODIC = args.get<bool>("PERIODIC",false);

#if defined(USING_SU2)
	typedef Sym::SU2<Sym::SpinSU2> Symmetry;
#elif defined(USING_U1)
	typedef Sym::U1<Sym::SpinU1> Symmetry;
#else
	typedef Sym::U0 Symmetry;
#endif
	
	typedef Sym::S1xS2<Sym::U1<Sym::ChargeU1>,Sym::U1<Sym::SpinU1> > Symmetry2;
	typedef SpinBase<Symmetry> Base;
	typedef Base::OperatorType Op;
	Base B(L,D);
        cout << Op::prod(B.Sdag(), B.S(), {M}) << endl;
// 	ArrayXXd Jarr; Jarr.resize(L,L); Jarr.setZero();
// 	Jarr.matrix().template diagonal<1>().setConstant(0.5*J);
// 	Jarr.matrix().template diagonal<-1>() = Jarr.matrix().template diagonal<1>();
// 	if (PERIODIC and L > 2)
// 	{
// 		Jarr(0,L-1) = 0.5*J;
// 		Jarr(L-1,0) = 0.5*J;
// 	}
// 	auto H = B.HeisenbergHamiltonian(Jarr);
// 	EDSolver<Op> John(H,{},Eigen::DecompositionOptions::ComputeEigenvectors);
// 	cout << "gse=" << endl << John.eigenvalues().data().print(true) << endl;
	
// 	if (OBSERVABLES)
// 	{
// 		qarray<Symmetry::Nq> Q1;
// #if defined(USING_SU2)
// 		Q1 = {S};
// #elif defined(USING_U1)
// 		Q1 = {M};
// #else
// 		Q1 = {};
// #endif
// 		Eigen::ArrayXXd SdagS(L,L); SdagS.setZero();
// 		for(size_t i=0; i<L; i++)
// 		for(size_t j=0; j<L; j++)
// 		{
// #if defined(USING_SU2)
// 			auto SG = Op::prod(B.S(i), John.groundstate(Q1), {3});
// 			auto SSG = Op::prod(B.Sdag(j), SG, {1});
// 			auto res = Op::prod(John.groundstate(Q1), SSG, {1});
// 			SdagS(i,j) = sqrt(3.)*res.data().block[0](0,0);
// #elif defined(USING_U1)
// 			auto SzG = Op::prod(B.Sz(i), John.groundstate(Q1), {0});
// 			auto SzSzG = Op::prod(B.Sz(j), SzG, {0});
// 			auto resSzSz = Op::prod(John.groundstate(Q1), SzSzG, {0});
// 			auto SpG = Op::prod(B.Sp(i), John.groundstate(Q1), {2});
// 			auto SmSpG = Op::prod(B.Sm(j), SpG, {0});
// 			auto resSmSp = Op::prod(John.groundstate(Q1), SmSpG, {0});
// 			auto SmG = Op::prod(B.Sm(i), John.groundstate(Q1), {-2});
// 			auto SpSmG = Op::prod(B.Sp(j), SmG, {0});
// 			auto resSpSm = Op::prod(John.groundstate(Q1), SpSmG, {0});
// 			SdagS(i,j) = resSzSz.data().block[0](0,0) + 0.5*(resSpSm.data().block[0](0,0) + resSmSp.data().block[0](0,0));
// #else
// 			auto SzG = Op::prod(B.Sz(i), John.groundstate(Q1), {});
// 			auto SzSzG = Op::prod(B.Sz(j), SzG, {});
// 			auto resSzSz = Op::prod(John.groundstate(Q1), SzSzG, {});
			
// 			auto SpG = Op::prod(B.Sp(i), John.groundstate(Q1), {});
// 			auto SmSpG = Op::prod(B.Sm(j), SpG, {});
// 			auto resSmSp = Op::prod(John.groundstate(Q1), SmSpG, {});
			
// 			auto SmG = Op::prod(B.Sm(i), John.groundstate(Q1), {});
// 			auto SpSmG = Op::prod(B.Sp(j), SmG, {});
// 			auto resSpSm = Op::prod(John.groundstate(Q1), SpSmG, {});
// 			SdagS(i,j) = resSzSz.data().block[0](0,0) + 0.5*(resSpSm.data().block[0](0,0) + resSmSp.data().block[0](0,0));
// #endif
// 		}
// 		cout << "nearest neighbour spin correlations:" << endl << SdagS.matrix().template diagonal<1>() << endl;
// 		cout << "edge-edge spin correlations: " << SdagS(0,L-1) << endl;
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
	// }
}

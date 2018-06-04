#include <iostream>

#include "termcolor.hpp"

#include "ArgParser.h"

#include "Logger.h"
Logger lout;

using namespace std;
#include "bases/FermionBaseSU2xU1.h"

#include "models/HubbardU1xU1.h"
#include "solvers/DmrgSolver.h"

size_t L;
double U;
int S,N;
int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",4);
	U = args.get<double>("U",8.);
	S = args.get<int>("S",1);
	N = args.get<int>("N",L);
	
	bool COMMUTATORS = args.get<bool>("COMMUTATORS",false);
	bool OBSERVABLES = args.get<bool>("OBSERVABLES",true);

	typedef Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > Symmetry;
	typedef FermionBase<Symmetry> Base;
	typedef Base::Operator Op;
	Base F(L);

	if(COMMUTATORS)
	{
		cout << termcolor::red << "-------Commutators--------" << termcolor::reset << endl;
		for(size_t i=0; i<L; i++)
		for(size_t j=0; j<L; j++)
		{
			auto commutator = Op::prod(F.cdag(i),F.c(j),Symmetry::qvacuum()) - Op::prod(F.c(j),F.cdag(i),Symmetry::qvacuum());
			cout << "i=" << i << ", j=" << j << endl << commutator.data().print(true) << endl << endl;
			
			auto anticommutator = Op::prod(F.cdag(i),F.c(j),Symmetry::qvacuum()) + Op::prod(F.c(j),F.cdag(i),Symmetry::qvacuum());
			cout << "i=" << i << ", j=" << j << endl << anticommutator.data().print(true) << endl << endl;
		}
	}

	if (OBSERVABLES)
	{
		qarray<2> Q1 = {S,N};
		qarray<2> Q2 = {S+1,N-1};

		auto H = F.HubbardHamiltonian(U);
		EDSolver<Op> John(H,{Q1,Q2},Eigen::DecompositionOptions::ComputeEigenvectors);
		cout << "gse1=" << John.eigenvalues().data().block[1](0,0) << endl;
		cout << "gse2=" << John.eigenvalues().data().block[0](0,0) << endl;

		Eigen::MatrixXd densityMatrix(L,L); densityMatrix.setZero();
		for(size_t i=0; i<L; i++)
		for(size_t j=0; j<L; j++)
		{
  			auto res = Op::prod(John.groundstate(Q1).adjoint(),Op::prod(F.cdag(i),Op::prod(F.c(j),John.groundstate(Q1),{2,-1}),{1,0}),{1,0});

			densityMatrix(i,j) = sqrt(2.) * res.data().block[0](0,0);
		}
		cout << "densityMatrix" << endl << densityMatrix << endl << endl;

		Eigen::VectorXd n(L); n.setZero();
		for(size_t i=0; i<L; i++)
		{
  			auto res = Op::prod(John.groundstate(Q1).adjoint(),Op::prod(F.n(i),John.groundstate(Q1),{1,0}),{1,0});
			n(i) = res.data().block[0](0,0);
		}
		cout << "n" << endl << n << endl << n.sum() << endl << endl;

		Eigen::VectorXd c(L); c.setZero();
		for(size_t i=0; i<L; i++)
		{
  			auto res = Op::prod(John.groundstate(Q2).adjoint(),Op::prod(F.c(i),John.groundstate(Q1),{2,-1}),{2,-1});
			c(i) = res.data().block[0](0,0);
		}
		cout << "c" << endl << c << endl << endl;

		Eigen::VectorXd cdag(L); cdag.setZero();
		for(size_t i=0; i<L; i++)
		{
			//cdag2 gibt den richtigen hermitische konjugierten Operator!
  			auto res = Op::prod(John.groundstate(Q1).adjoint(),Op::prod(F.cdag2(i),John.groundstate(Q2),{2,1}),{2,1});
			cdag(i) = res.data().block[0](0,0);
		}

		cout << "cdag" << endl << cdag << endl << endl;
		cout << "ratio=" << endl << c.array()/(cdag.array()) << endl << endl;
		
		VMPS::HubbardU1xU1 H_DMRG(L,{{"U",U}});
		cout << H_DMRG.info() << endl;
		VMPS::HubbardU1xU1::Solver Jack(DMRG::VERBOSITY::ON_EXIT);
		Eigenstate<VMPS::HubbardU1xU1::StateXd> g1, g2;
		Jack.edgeState(H_DMRG,g1,{static_cast<int>(L/2),static_cast<int>(L/2)},LANCZOS::EDGE::GROUND);
		
		VMPS::HubbardU1xU1::Solver Lisa(DMRG::VERBOSITY::ON_EXIT);
		Lisa.edgeState(H_DMRG,g2,{static_cast<int>(L/2)-1,static_cast<int>(L/2)},LANCZOS::EDGE::GROUND);
		Eigen::VectorXd c_check(L); c_check.setZero();
		for(size_t i=0; i<L; i++)
		{
			c_check(i) = avg(g2.state, H_DMRG.c(UP,i), g1.state);
		}
		cout << "c_check" << endl << c_check << endl << endl;
		
	}
}

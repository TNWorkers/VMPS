#ifndef VANILLA_GRANDHUBBARDMODEL
#define VANILLA_GRANDHUBBARDMODEL

#include "MpHubbardModel.h"

namespace VMPS
{

class GrandHubbardModel : public MpoQ<0,double>
{
public:
	
	GrandHubbardModel (size_t L_input, double U_input, double mu_input, bool OPEN_BC=false, bool CALC_SQUARE=true);
	
	typedef MpsQ<0,double>                           StateXd;
	typedef MpsQ<0,complex<double> >                 StateXcd;
	typedef DmrgSolverQ<0,GrandHubbardModel>         Solver;
	typedef VumpsSolver<0,GrandHubbardModel>         uSolver;
	typedef MpsQCompressor<0,double,double>          CompressorXd;
	typedef MpsQCompressor<0,complex<double>,double> CompressorXcd;
	typedef MpoQ<0>                                  Operator;
	
	MpoQ<0> n (SPIN_INDEX sigma, size_t loc);
	
private:
	
	FermionBase F;
	
	double U, mu;
};

GrandHubbardModel::
GrandHubbardModel (size_t L_input, double U_input, double mu_input, bool OPEN_BC, bool CALC_SQUARE)
:MpoQ<0> (L_input, 1, vector<qarray<0> >(begin(qloc4dummy),end(qloc4dummy)), {}, labeldummy, "HubbardModel"),
U(U_input), mu(mu_input)
{
	stringstream ss;
	ss << "(U=" << U << ",mu=" << mu << ")";
	this->label += ss.str();
	
	F = FermionBase(1,!isfinite(U));
	
	vector<double> Uvec(1); Uvec[0] = U;
	vector<double> muvec(1); muvec[0] = -mu; // H=H_0-mu*N
	MatrixXd tInter(1,1); tInter(0,0) = 1.;
	HamiltonianTermsXd Terms = HubbardModel::set_operators(F,Uvec,muvec,tInter);
	this->Daux = Terms.auxdim();
	
	SuperMatrix<double> G = ::Generator(Terms);
	this->construct(G, this->W, this->Gvec, OPEN_BC);
	
	if (CALC_SQUARE == true)
	{
		this->construct(tensor_product(G,G), this->Wsq, this->GvecSq);
		this->GOT_SQUARE = true;
	}
	else
	{
		this->GOT_SQUARE = false;
	}
}

MpoQ<0> GrandHubbardModel::
n (SPIN_INDEX sigma, size_t loc)
{
	assert(loc<N_sites);
	stringstream ss;
	ss << "n(" << loc << ",σ=" << sigma << ")";
	MpoQ<0> Mout(N_sites, 1, MpoQ<0>::qloc, {}, labeldummy, ss.str());
	Mout.setLocal(loc, F.n(sigma));
	return Mout;
}

}

#endif

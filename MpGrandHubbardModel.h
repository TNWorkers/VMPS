#ifndef VANILLA_GRANDHUBBARDMODEL
#define VANILLA_GRANDHUBBARDMODEL

#include "MpHubbardModel.h"

namespace VMPS
{

class GrandHubbardModel : public MpoQ<0,double>
{
public:
	
	GrandHubbardModel (size_t L_input, double U_input, double mu_input, bool OPEN_BC=false, bool CALC_SQUARE=true);
	
	GrandHubbardModel (size_t L_input, std::array<double,2> t_input, double U_input, double mu_input, bool OPEN_BC=false, bool CALC_SQUARE=true);
	
	typedef MpsQ<0,double>                           StateXd;
	typedef MpsQ<0,complex<double> >                 StateXcd;
	typedef DmrgSolverQ<0,GrandHubbardModel>         Solver;
	typedef VumpsSolver<0,GrandHubbardModel>         uSolver;
	typedef MpsQCompressor<0,double,double>          CompressorXd;
	typedef MpsQCompressor<0,complex<double>,double> CompressorXcd;
	typedef MpoQ<0>                                  Operator;
	
	MpoQ<0> n (SPIN_INDEX sigma, size_t loc) const;
	MpoQ<0> Sz (size_t loc) const;
	
private:
	
	FermionBase F;
	
	double U, mu;
	std::array<double,2> t;
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

GrandHubbardModel::
GrandHubbardModel (size_t L_input, std::array<double,2> t_input, double U_input, double mu_input, bool OPEN_BC, bool CALC_SQUARE)
:MpoQ<0> (L_input, 1, vector<qarray<0> >(begin(qloc4dummy),end(qloc4dummy)), {}, labeldummy, "HubbardModel"),
U(U_input), mu(mu_input), t(t_input)
{
	stringstream ss;
	ss << "(U=" << U << ",mu=" << mu << ")";
	this->label += ss.str();
	
	F = FermionBase(1,!isfinite(U));
	
	vector<double> Uvec(1); Uvec[0] = U;
	vector<double> muvec(1); muvec[0] = -mu; // H=H_0-mu*N
	MatrixXd tInter01(1,1); tInter01(0,0) = t[0];
	MatrixXd tInter10(1,1); tInter10(0,0) = t[1];
	HamiltonianTermsXd Terms01 = HubbardModel::set_operators(F,Uvec,muvec,tInter01);
	HamiltonianTermsXd Terms10 = HubbardModel::set_operators(F,Uvec,muvec,tInter10);
	this->Daux = Terms01.auxdim();
	
	SuperMatrix<double> G01 = ::Generator(Terms01);
	SuperMatrix<double> G10 = ::Generator(Terms10);
	
	vector<SuperMatrix<double> > G(this->N_sites);
	vector<SuperMatrix<double> > Gsq;
	if (CALC_SQUARE == true)
	{
		Gsq.resize(this->N_sites);
	}
	
	for (size_t l=0; l<this->N_sites; ++l)
	{
		// if OPENBC
		this->Daux = Terms01.auxdim();
		G[l].setMatrix(Daux,F.dim());
		G[l] = (l%2==0)? G01:G10;
		
		if (CALC_SQUARE == true)
		{
			Gsq[l] = tensor_product(G[l],G[l]);
		}
	}
	
	this->construct(G, this->W, this->Gvec);
	
	if (CALC_SQUARE == true)
	{
		this->construct(Gsq, this->Wsq, this->GvecSq);
		this->GOT_SQUARE = true;
	}
	else
	{
		this->GOT_SQUARE = false;
	}
}

MpoQ<0> GrandHubbardModel::
n (SPIN_INDEX sigma, size_t loc) const
{
	assert(loc<N_sites);
	stringstream ss;
	ss << "n(" << loc << ",σ=" << sigma << ")";
	MpoQ<0> Mout(N_sites, 1, MpoQ<0>::qloc, {}, labeldummy, ss.str());
	Mout.setLocal(loc, F.n(sigma));
	return Mout;
}

MpoQ<0> GrandHubbardModel::
Sz (size_t loc) const
{
	assert(loc<N_sites);
	stringstream ss;
	ss << "Sz(" << loc << ")";
	MpoQ<0> Mout(N_sites, 1, MpoQ<0>::qloc, {}, labeldummy, ss.str());
	Mout.setLocal(loc, F.Sz());
	return Mout;
}

}

#endif

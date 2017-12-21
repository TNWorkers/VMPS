#ifndef VANILLA_TRANSVERSEHEISENBERGMODEL
#define VANILLA_TRANSVERSEHEISENBERGMODEL

#include "MpHeisenbergModel.h"

namespace VMPS
{
typedef Sym::U0 Symmetry;

class TransverseHeisenbergModel : public MpoQ<Sym::U0,double>
{
public:
	
	TransverseHeisenbergModel (size_t Lx_input, double Jxy_input, double Jz_input, vector<double> Bz_input, vector<double> Bx_input, size_t D_input=2, 
	                           bool OPEN_BC=false, bool CALC_SQUARE=true);
	
	///@{
	/**Typedef for convenient reference (no need to specify \p Symmetry, \p Scalar all the time).*/
	typedef MpsQ<Symmetry,double>                           StateXd;
	typedef MpsQ<Symmetry,complex<double> >                 StateXcd;
	typedef DmrgSolverQ<Symmetry,TransverseHeisenbergModel> Solver;
	typedef VumpsSolver<Symmetry,TransverseHeisenbergModel> uSolver;
	///@}
	
	MpoQ<Symmetry> Scomp (SPINOP_LABEL Sa, size_t loc) const;
	MpoQ<Symmetry> Sz (size_t loc) const;
	MpoQ<Symmetry> SaSa (size_t loc1, SPINOP_LABEL SOP1, size_t loc2, SPINOP_LABEL SOP2) const;
	MpoQ<Symmetry> SzSz (size_t loc1, size_t loc2) const;
	
private:
	
	double Jxy = -1.;
	double Jz  = -1.;
	size_t D = 2;
	vector<double> Bz;
	vector<double> Bx;
	
	SpinBase B;
};

TransverseHeisenbergModel::
TransverseHeisenbergModel (size_t Lx_input, double Jxy_input, double Jz_input, vector<double> Bz_input, vector<double> Bx_input, size_t D_input, bool OPEN_BC, bool CALC_SQUARE)
:MpoQ<Symmetry>(), Jxy(Jxy_input), Jz(Jz_input), Bz(Bz_input), Bx(Bx_input), D(D_input)
{
	B = SpinBase(1,D);
	
	this->N_sites = Lx_input;
	assert(Bz.size() == this->N_sites);
	assert(Bx.size() == this->N_sites);
	this->N_legs = 1;
	this->Qtot = {};
	this->qlabel = labeldummy;
	this->label = "Transverse"+HeisenbergModel::create_label(D,Jxy,Jz,0,Bz[0],Bx[0]);
	this->format = noFormat;
	this->qloc.resize(this->N_sites);
	
	// create the SuperMatrices
	vector<SuperMatrix<double> > G(this->N_sites);
	vector<SuperMatrix<double> > Gsq;
	if (CALC_SQUARE == true) {Gsq.resize(this->N_sites);}
	
	HamiltonianTermsXd Terms;
	
	vector<qarray<0> > qlocDdummy(D);
	for (size_t s=0; s<D; ++s)
	{
		qlocDdummy[s] = qarray<0>{};
	}
	
	// first site
	if (OPEN_BC)
	{
		Terms = HeisenbergModel::set_operators(B, Jxy,Jz,Bz[0],Bx[0]);
		this->Daux = Terms.auxdim();
		G[0].setRowVector(this->Daux,D);
		G[0] = Generator(Terms).row(this->Daux-1);
		if (CALC_SQUARE == true)
		{
			Gsq[0].setRowVector(this->Daux*this->Daux,D);
			Gsq[0] = tensor_product(G[0],G[0]);
		}
		this->qloc[0] = qlocDdummy;
	}
	
	// middle sites
	size_t l_frst = (OPEN_BC)? 1:0;
	size_t l_last = (OPEN_BC)? this->N_sites-1:this->N_sites;
	for (size_t l=l_frst; l<l_last; ++l)
	{
		Terms = HeisenbergModel::set_operators(B, Jxy,Jz,Bz[l],Bx[l]);
		this->Daux = Terms.auxdim();
		G[l].setMatrix(this->Daux,D);
		G[l] = Generator(Terms);
		if (CALC_SQUARE == true)
		{
			Gsq[l].setMatrix(this->Daux*this->Daux,D);
			Gsq[l] = tensor_product(G[l],G[l]);
		}
		this->qloc[l] = qlocDdummy;
	}
	
	// last site
	if (OPEN_BC)
	{
		size_t last = this->N_sites-1;
		Terms = HeisenbergModel::set_operators(B, Jxy,Jz,Bz[last],Bx[last]);
		this->Daux = Terms.auxdim();
		G[last].setColVector(this->Daux,D);
		G[last] = Generator(Terms).col(0);
		
		if (CALC_SQUARE == true)
		{
			Gsq[last].setColVector(this->Daux*this->Daux,D);
			Gsq[last] = tensor_product(G[last],G[last]);
		}
		this->qloc[last] = qlocDdummy;
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

MpoQ<Sym::U0> TransverseHeisenbergModel::
Scomp (SPINOP_LABEL Sa, size_t loc) const
{
	assert(loc<N_sites);
	stringstream ss;
	ss << Sa;
	MpoQ<Symmetry> Mout(N_sites, 1, MpoQ<Symmetry>::qloc, {}, labeldummy, ss.str());
	Mout.setLocal(loc, B.Scomp(Sa));
	return Mout;
}

MpoQ<Sym::U0> TransverseHeisenbergModel::
Sz (size_t loc) const
{
	assert(loc<N_sites);
	stringstream ss;
	ss << SZ;
	MpoQ<Symmetry> Mout(N_sites, 1, MpoQ<Symmetry>::qloc, {}, labeldummy, ss.str());
	Mout.setLocal(loc, B.Scomp(SZ));
	return Mout;
}

MpoQ<Sym::U0> TransverseHeisenbergModel::
SzSz (size_t loc1, size_t loc2) const
{
	assert(loc1<N_sites and loc2<N_sites);
	stringstream ss;
	ss << "SzSz";
	MpoQ<Symmetry> Mout(N_sites, 1, MpoQ<Symmetry>::qloc, {}, labeldummy, ss.str());
	Mout.setLocal({loc1, loc2}, {B.Scomp(SZ), B.Scomp(SZ)});
	return Mout;
}

MpoQ<Sym::U0>TransverseHeisenbergModel::
SaSa (size_t loc1, SPINOP_LABEL SOP1, size_t loc2, SPINOP_LABEL SOP2) const
{
	assert(loc1<N_sites and loc2<N_sites);
	stringstream ss;
	ss << SOP1 << "(" << loc1 << ")" << SOP2 << "(" << loc2 << ")";
	MpoQ<Symmetry> Mout(N_sites, 1, MpoQ<Symmetry>::qloc, {}, labeldummy, ss.str());
	Mout.setLocal({loc1, loc2}, {B.Scomp(SOP1), B.Scomp(SOP2)});
	return Mout;
}

}

#endif

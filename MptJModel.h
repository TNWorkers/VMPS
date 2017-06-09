#ifndef STRAWBERRY_TJMODEL
#define STRAWBERRY_TJMODEL

#include "MpHubbardModel.h"

namespace VMPS
{

/**MPO representation of \f$H = H_{tJ}+H_{t'}+H_{3-site}\f$ with:
- \f$ H_{tJ} = - \sum_{<ij>\sigma} (c^\dagger_{i\sigma}c_{j\sigma} +h.c.) + J \sum_{<ij>} (\mathbf{S}_{i} \mathbf{S}_{j} - \frac{1}{4} n_in_j) \f$
- \f$ H_{t'} = - t^{\prime} \sum_{<<ij>>\sigma} (c^\dagger_{i\sigma}c_{j\sigma} + h.c.) \f$
- \f$ H_{3-site} = - \frac{J}{4} \sum_{<ijk>\sigma} (c^\dagger_{i\sigma} n_{j,-\sigma} c_{k\sigma} - c^\dagger_{i\sigma} S^{-\sigma}_j c_{k,-\sigma} + h.c.) \f$
\note useful reference: "Effect of the Three-Site Hopping Term on the t-J Model"; Ammon, Troyer, Tsunetsugu (1995) (http://arxiv.org/pdf/cond-mat/9502037v1.pdf)
\note If the nnn-hopping is positive, the ground state energy is lowered.
\warning \f$J>0\f$ is antiferromagnetic*/
class tJModel : public HubbardModel
{
typedef Sym::U1xU1<double> Symmetry;

public:
	
	/**
	\param Lx_input : chain length
	\param J_input : \f$J\f$; if \f$J=0\f$, then we have the infinite-U Hubbard model
	\param THREE_SITE_TERMS : if true, adds the 3-site terms to the Hamiltonian
	\param tPrime_input : \f$t^{\prime}\f$
	\param Ly_input : amount of legs in ladder
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	*/
	tJModel (size_t Lx_input, double J_input=0., bool THREE_SITE_TERMS=true, double tPrime_input=0., size_t Ly_input=1, bool CALC_SQUARE=true);
	
	static HamiltonianTermsXd set_operators (const FermionBase &F, double J=0., bool THREE_SITE_TERMS=true, double tPrime=0.);
	
//	class qarrayIterator;
	
	///@{
	typedef MpsQ<Symmetry,double>           StateXd;
	typedef MpsQ<Symmetry,complex<double> > StateXcd;
	typedef DmrgSolverQ<Symmetry,tJModel>   Solver;
	///@}
	
private:
	
	double J = 0.;
};

HamiltonianTermsXd tJModel::
set_operators (const FermionBase &F, double J, bool THREE_SITE_TERMS, double tPrime)
{
	HamiltonianTermsXd Terms;
	
	for (int leg=0; leg<F.orbitals(); ++leg)
	{
		Terms.tight.push_back(make_tuple(-1., F.cdag(UP,leg), F.sign()*F.c(UP,leg)));
		Terms.tight.push_back(make_tuple(-1., F.cdag(DN,leg), F.sign()*F.c(DN,leg)));
		Terms.tight.push_back(make_tuple(+1., F.c(UP,leg),    F.sign()*F.cdag(UP,leg)));
		Terms.tight.push_back(make_tuple(+1., F.c(DN,leg),    F.sign()*F.cdag(DN,leg)));
	}
	
	if (J != 0.)
	{
		for (int leg=0; leg<F.orbitals(); ++leg)
		{
			Terms.tight.push_back(make_tuple(0.5*J,   F.Sp(leg), F.Sm(leg)));
			Terms.tight.push_back(make_tuple(0.5*J,   F.Sm(leg), F.Sp(leg)));
			Terms.tight.push_back(make_tuple(J,       F.Sz(leg), F.Sz(leg)));
			Terms.tight.push_back(make_tuple(-0.25*J, F.n(leg),  F.n(leg)));
		}
		
		if (THREE_SITE_TERMS)
		{
			assert(F.orbitals() == 1 and "Cannot do a ladder with 3-site terms!");
			// three-site terms without spinflip
			Terms.nextn.push_back(make_tuple(-0.25*J, F.cdag(UP), F.sign()*F.c(UP),    F.n(DN)*F.sign()));
			Terms.nextn.push_back(make_tuple(-0.25*J, F.cdag(DN), F.sign()*F.c(DN),    F.n(UP)*F.sign()));
			Terms.nextn.push_back(make_tuple(+0.25*J, F.c(UP),    F.sign()*F.cdag(UP), F.n(DN)*F.sign()));
			Terms.nextn.push_back(make_tuple(+0.25*J, F.c(DN),    F.sign()*F.cdag(DN), F.n(UP)*F.sign()));
			
			// three-site terms with spinflip
			Terms.nextn.push_back(make_tuple(+0.25*J, F.cdag(DN), F.sign()*F.c(UP),    F.Sp()*F.sign()));
			Terms.nextn.push_back(make_tuple(+0.25*J, F.cdag(UP), F.sign()*F.c(DN),    F.Sm()*F.sign()));
			Terms.nextn.push_back(make_tuple(-0.25*J, F.c(DN),    F.sign()*F.cdag(UP), F.Sm()*F.sign()));
			Terms.nextn.push_back(make_tuple(-0.25*J, F.c(UP),    F.sign()*F.cdag(DN), F.Sp()*F.sign()));
		}
	}
	
	if (tPrime != 0.)
	{
		assert(F.orbitals() == 1 and "Cannot do a ladder with t'-terms!");
		
		Terms.nextn.push_back(make_tuple(-tPrime, F.cdag(UP), F.sign()*F.c(UP),    F.sign()));
		Terms.nextn.push_back(make_tuple(-tPrime, F.cdag(DN), F.sign()*F.c(DN),    F.sign()));
		Terms.nextn.push_back(make_tuple(+tPrime, F.c(UP),    F.sign()*F.cdag(UP), F.sign()));
		Terms.nextn.push_back(make_tuple(+tPrime, F.c(DN),    F.sign()*F.cdag(DN), F.sign()));
	}
	
	Terms.local.push_back(make_tuple(1., F.HubbardHamiltonian(numeric_limits<double>::infinity(),1.,0.,J,false)));
	
	return Terms;
}

tJModel::
tJModel (size_t Lx_input, double J_input, bool THREE_SITE_TERMS, double tPrime_input, size_t Ly_input, bool CALC_SQUARE)
:HubbardModel(Lx_input, numeric_limits<double>::infinity(), 0., tPrime_input, Ly_input), J(J_input)
{
	stringstream ss;
	ss << "tJModel" << "(J=" << J << ",t'=" << tPrime << ",3site=" << boolalpha << THREE_SITE_TERMS << ")";
	this->label = ss.str();
	
	HamiltonianTermsXd Terms = set_operators(F, J,THREE_SITE_TERMS,tPrime);
	SuperMatrix<double> G = Generator(Terms);
	this->Daux = Terms.auxdim();
	
	this->W.clear();
	this->Gvec.clear();
	this->construct(G, this->W, this->Gvec);
	
	if (CALC_SQUARE == true)
	{
		this->Wsq.clear();
		this->GvecSq.clear();
		this->construct(tensor_product(G,G), this->Wsq, this->GvecSq);
		this->GOT_SQUARE = true;
	}
	else
	{
		this->GOT_SQUARE = false;
	}
}

}

#endif

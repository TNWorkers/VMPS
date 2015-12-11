#ifndef STRAWBERRY_TJMODEL
#define STRAWBERRY_TJMODEL

#include "MpHubbardModel.h"

namespace VMPS
{

/**MPO representation of 
\f$
H = - \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} 
    - t^{\prime} \sum_{<<ij>>\sigma} c^\dagger_{i\sigma}c_{j\sigma} 
    + J \sum_{<ij>} \mathbf{S}_{i} \mathbf{S}_{j}
\f$.
\note Three-site terms not implemented.
\note If the nnn-hopping is positive, the ground state energy is lowered.
\warning \f$J>0\f$ is antiferromagnetic*/
class tJModel : public HubbardModel
{
public:
	
	tJModel (size_t Lx_input, double J_input=0., bool THREE_SITE_TERMS=true, double tPrime_input=0., size_t Ly_input=1, bool CALC_SQUARE=true);
	
	static HamiltonianTermsXd set_operators (const FermionBase &F, double J=0., bool THREE_SITE_TERMS=true, double tPrime=0.);
	
//	class qarrayIterator;
	
	///@{
	typedef MpsQ<2,double>           StateXd;
	typedef MpsQ<2,complex<double> > StateXcd;
	typedef DmrgSolverQ<2,tJModel>   Solver;
	///@}
	
private:
	
	double J = 0.;
};

HamiltonianTermsXd tJModel::
set_operators (const FermionBase &F, double J, bool THREE_SITE_TERMS, double tPrime)
{
	HamiltonianTermsXd Terms;
	
	Terms.tight.push_back(make_tuple(-1., F.cdag(UP,0), F.sign() * F.c(UP,0)));
	Terms.tight.push_back(make_tuple(-1., F.cdag(DN,0), F.sign() * F.c(DN,0)));
	Terms.tight.push_back(make_tuple(+1., F.c(UP,0),    F.sign() * F.cdag(UP,0)));
	Terms.tight.push_back(make_tuple(+1., F.c(DN,0),    F.sign() * F.cdag(DN,0)));
	
	if (J != 0.)
	{
		Terms.tight.push_back(make_tuple(0.5*J,   F.Sp(0), F.Sm(0)));
		Terms.tight.push_back(make_tuple(0.5*J,   F.Sm(0), F.Sp(0)));
		Terms.tight.push_back(make_tuple(J,       F.Sz(0), F.Sz(0)));
		Terms.tight.push_back(make_tuple(-0.25*J, F.n(0),  F.n(0)));
		
		if (THREE_SITE_TERMS)
		{
			// three-site terms without spinflip
			Terms.nextn.push_back(make_tuple(+0.5*J, F.cdag(UP), F.sign() * F.c(UP),    F.n(DN) * F.sign()));
			Terms.nextn.push_back(make_tuple(+0.5*J, F.cdag(DN), F.sign() * F.c(DN),    F.n(UP) * F.sign()));
			Terms.nextn.push_back(make_tuple(-0.5*J, F.c(UP),    F.sign() * F.cdag(UP), F.n(DN) * F.sign()));
			Terms.nextn.push_back(make_tuple(-0.5*J, F.c(DN),    F.sign() * F.cdag(DN), F.n(UP) * F.sign()));
		
			// three-site terms with spinflip
			Terms.nextn.push_back(make_tuple(-0.5*J, F.cdag(DN), F.sign() * F.c(UP),    F.Sp() * F.sign()));
			Terms.nextn.push_back(make_tuple(-0.5*J, F.cdag(UP), F.sign() * F.c(DN),    F.Sm() * F.sign()));
			Terms.nextn.push_back(make_tuple(+0.5*J, F.c(DN),    F.sign() * F.cdag(UP), F.Sm() * F.sign()));
			Terms.nextn.push_back(make_tuple(+0.5*J, F.c(UP),    F.sign() * F.cdag(DN), F.Sp() * F.sign()));
		}
	}
	
	if (tPrime != 0.)
	{
		Terms.nextn.push_back(make_tuple(-tPrime, F.cdag(UP), F.sign() * F.c(UP),    F.sign()));
		Terms.nextn.push_back(make_tuple(-tPrime, F.cdag(DN), F.sign() * F.c(DN),    F.sign()));
		Terms.nextn.push_back(make_tuple(+tPrime, F.c(UP),    F.sign() * F.cdag(UP), F.sign()));
		Terms.nextn.push_back(make_tuple(+tPrime, F.c(DN),    F.sign() * F.cdag(DN), F.sign()));
	}
	
	Terms.local.push_back(make_tuple(1., F.HubbardHamiltonian(numeric_limits<double>::infinity(),1.,0.,J,false)));
	
	return Terms;
}

tJModel::
tJModel (size_t Lx_input, double J_input, bool THREE_SITE_TERMS, double tPrime_input, size_t Ly_input, bool CALC_SQUARE)
:HubbardModel(Lx_input, numeric_limits<double>::infinity(), 0., tPrime_input, Ly_input), J(J_input)
{
	stringstream ss;
	ss << "tJModel" << "(J=" << J << ",t'=" << tPrime << ")";
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

//class tJModel::qarrayIterator
//{
//public:
//	
//	/**
//	\param qloc_input : vector of local bases
//	\param l_frst : first site
//	\param l_last : last site
//	\param N_0s : dimension in y-direction
//	*/
//	qarrayIterator (const vector<vector<qarray<2> > > &qloc_input, int l_frst, int l_last, size_t N_0s=1)
//	{
//		if (l_last<0 or l_frst>=qloc_input.size())
//		{
//			N_sites = 0;
//		}
//		else
//		{
//			N_sites = l_last-l_frst+1;
//		}
//		
//		for (int N=0; N<=N_sites*static_cast<int>(N_0s); ++N)
//		for (int Nup=0; Nup<=N; ++Nup)
//		{
//			qarray<2> q = {Nup,N-Nup};
//			qarraySet.insert(q);
//		}
//		
//		it = qarraySet.begin();
//	};
//	
//	qarray<2> operator*() {return value;}
//	
//	qarrayIterator& operator= (const qarray<2> a) {value=a;}
//	bool operator!=           (const qarray<2> a) {return value!=a;}
//	bool operator<=           (const qarray<2> a) {return value<=a;}
//	bool operator<            (const qarray<2> a) {return value< a;}
//	
//	qarray<2> begin()
//	{
//		return *(qarraySet.begin());
//	}
//	
//	qarray<2> end()
//	{
//		return *(qarraySet.end());
//	}
//	
//	void operator++()
//	{
//		++it;
//		value = *it;
//	}
//	
//private:
//	
//	qarray<2> value;
//	
//	set<qarray<2> > qarraySet;
//	set<qarray<2> >::iterator it;
//	
//	int N_sites;
//};

}

#endif

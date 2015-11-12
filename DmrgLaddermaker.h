#ifndef STRAWBERRY_LADDERMAKER_WITH_Q
#define STRAWBERRY_LADDERMAKER_WITH_Q

#include "MpoQ.h"

template<size_t Nq>
qarray<Nq> tensor_product (const qarray<Nq> &a1, const qarray<Nq> &a2)
{
	qarray<Nq> aout;
	for (int q=0; q<Nq; ++q)
	{
		aout[q] = a1[q]+a2[q];
	}
	return aout;
}
//--------------

template<size_t Nq, typename Scalar>
class DmrgLaddermaker : public MpoQ<Nq,Scalar>
{
typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;

public:
	
	DmrgLaddermaker (const MpoQ<Nq,Scalar> &H, bool CALC_SQUARE=true);
	
private:
	
};

template<size_t Nq, typename Scalar>
DmrgLaddermaker<Nq,Scalar>::
DmrgLaddermaker (const MpoQ<Nq,Scalar> &H, bool CALC_SQUARE)
:MpoQ<Nq,Scalar>()
{
	this->N_sites = H.length();
	this->Qtot = qvacuum<Nq>();
	assert(H.Onextn.size() == 0 and "Cannot make a ladder from a Hamiltonian with nearest-neighbour hopping!");
	this->qlabel = H.qlabel;
	this->label = "Ladder<"+H.label+">";
	this->format = H.format;
	
	this->qloc.resize(this->N_sites);
	for (size_t l=0; l<this->N_sites; ++l)
	{
		this->qloc[l].resize(H.locBasis(l).size()*H.locBasis(l).size());
		
		for (size_t s1=0; s1<H.locBasis(l).size(); ++s1)
		for (size_t s2=0; s2<H.locBasis(l).size(); ++s2)
		{
			this->qloc[l][s2+H.locBasis(l).size()*s1] = tensor_product(H.locBasis(l)[s1], H.locBasis(l)[s2]);
		}
	}
	
	MatrixType Id(H.locBasis(0).size(), H.locBasis(0).size());
	Id.setIdentity();
	
	// hopping along upper edge 
	for (int i=0; i<H.Otight.size(); ++i)
	{
		this->Otight.push_back(make_tuple(get<0>(H.Otight[i]),
		                                  kroneckerProduct(get<1>(H.Otight[i]), Id),
		                                  kroneckerProduct(get<2>(H.Otight[i]), Id)
		                                 )
		                      );
	}
	// hopping along lower edge
	for (int i=0; i<H.Otight.size(); ++i)
	{
		this->Otight.push_back(make_tuple(get<0>(H.Otight[i]),
		                                  kroneckerProduct(Id, get<1>(H.Otight[i])),
		                                  kroneckerProduct(Id, get<2>(H.Otight[i]))
		                                 )
		                      );
	}
	// local interaction
	for (int i=0; i<H.Olocal.size(); ++i)
	{
		this->Olocal.push_back(make_tuple(get<0>(H.Olocal[i]),
		                                  kroneckerProduct(Id, get<1>(H.Olocal[i]))
		                                 )
		                      );
		this->Olocal.push_back(make_tuple(get<0>(H.Olocal[i]),
		                                  kroneckerProduct(get<1>(H.Olocal[i]), Id)
		                                 )
		                      );
	}
	// hopping along rungs -> local term in ladder
	for (int i=0; i<H.Otight.size(); ++i)
	{
		this->Olocal.push_back(make_tuple(get<0>(H.Otight[i]),
		                                  kroneckerProduct(get<1>(H.Otight[i]), get<2>(H.Otight[i]))
		                                 )
		                      );
		this->Olocal.push_back(make_tuple(get<0>(H.Otight[i]),
		                                  kroneckerProduct(get<2>(H.Otight[i]), get<1>(H.Otight[i]))
		                                 )
		                      );
	}
	
	this->Daux = 2 + 2*this->Otight.size();
	
	SuperMatrix<double> G = ::Generator(this->Olocal,this->Otight,this->Onextn);
	
	this->construct(G, this->W, this->Gvec);
	
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

#endif

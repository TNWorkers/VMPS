#ifndef STRAWBERRY_LADDERMAKER_WITH_Q
#define STRAWBERRY_LADDERMAKER_WITH_Q

#include "MpoQ.h"
#include "MpHubbardModel.h"

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
	
	DmrgLaddermaker (size_t Lx_input, size_t Ly_input, double U_input, bool CALC_SQUARE=true);
	
private:
	
	size_t N_legs;
};

template<size_t Nq, typename Scalar>
DmrgLaddermaker<Nq,Scalar>::
DmrgLaddermaker (size_t Lx_input, size_t Ly_input, double U, bool CALC_SQUARE)
:MpoQ<Nq,Scalar>()
{
	this->N_sites = Lx_input;
	this->Qtot = qvacuum<Nq>();
	this->qlabel = VMPS::HubbardModel::Nlabel;
	this->label = "Ladder<HubbardModel>";
	this->format = noFormat;
	N_legs = Ly_input;
	
	VMPS::HubbardModel H(Lx_input,U);
	
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
	
	for (size_t s=0; s<16; ++s)
	{
		cout << s << "\t" << this->qloc[0][s] << endl;
	}
	
	MatrixType Id(H.locBasis(0).size(), H.locBasis(0).size());
	Id.setIdentity();
	FermionBase F(N_legs);
	
//	// upper leg
//	this->Otight.push_back(make_tuple(-1., F.cdag(0,UP)*F.fsign_(0)*F.fsign_(1),	F.c(0,UP)));
//	this->Otight.push_back(make_tuple(-1., F.cdag(0,DN)*F.fsign_(0)*F.fsign_(1),	F.c(0,DN)));
//	this->Otight.push_back(make_tuple(+1., F.c(0,UP),	F.cdag(0,UP)*F.fsign_(0)*F.fsign_(1)));
//	this->Otight.push_back(make_tuple(+1., F.c(0,DN),	F.cdag(0,DN)*F.fsign_(0)*F.fsign_(1)));
//	
//	// lower leg
//	this->Otight.push_back(make_tuple(-1., F.cdag(1,UP)*F.fsign_(1)*F.fsign_(0),	F.c(1,UP)));
//	this->Otight.push_back(make_tuple(-1., F.cdag(1,DN)*F.fsign_(1)*F.fsign_(0),	F.c(1,DN)));
//	this->Otight.push_back(make_tuple(+1., F.c(1,UP),	F.cdag(1,UP)*F.fsign_(1)*F.fsign_(0)));
//	this->Otight.push_back(make_tuple(+1., F.c(1,DN),	F.cdag(1,DN)*F.fsign_(1)*F.fsign_(0)));
	
//	// upper leg
//	this->Otight.push_back(make_tuple(-1., F.cdag(0,UP)*F.fsign_(0)*F.fsign_(1),	F.c(0,UP)));
//	this->Otight.push_back(make_tuple(-1., F.cdag(0,DN)*F.fsign_(0,DN)*F.fsign_(1),	F.fsign_(0,UP)*F.c(0,DN)));
//	this->Otight.push_back(make_tuple(+1., F.c(0,UP)*F.fsign_(0)*F.fsign_(1),	F.cdag(0,UP)));
//	this->Otight.push_back(make_tuple(+1., F.c(0,DN)*F.fsign_(0,DN)*F.fsign_(1),	F.fsign_(0,UP)*F.cdag(0,DN)));
//	
//	// lower leg
//	this->Otight.push_back(make_tuple(-1., F.cdag(1,UP)*F.fsign_(1),	F.fsign_(0)*F.c(1,UP)));
//	this->Otight.push_back(make_tuple(-1., F.cdag(1,DN)*F.fsign_(1,DN),	F.fsign_(0)*F.fsign_(1,UP)*F.c(1,DN)));
//	this->Otight.push_back(make_tuple(+1., F.c(1,UP)*F.fsign_(1),	F.fsign_(0)*F.cdag(1,UP)));
//	this->Otight.push_back(make_tuple(+1., F.c(1,DN)*F.fsign_(1,DN),	F.fsign_(0)*F.fsign_(1,UP)*F.cdag(1,DN)));
	
	for (int leg=0; leg<N_legs; ++leg)
	{
		MatrixXd fsignTot;
		fsignTot.setIdentity(F.dim(),F.dim());
		// first supersite:
		for (int i=leg; i<N_legs; ++i)
		{
			fsignTot = fsignTot * F.fsign_(i);
		}
		// second supersite:
		for (int i=0; i<leg; ++i)
		{
			fsignTot = fsignTot * F.fsign_(i);
		}
		cout << "leg=" << leg << " f=" << fsignTot << endl << endl;
		
		this->Otight.push_back(make_tuple(-1., F.cdag(leg,UP), fsignTot * F.c(leg,UP)));
		this->Otight.push_back(make_tuple(-1., F.cdag(leg,DN), fsignTot * F.c(leg,DN)));
		this->Otight.push_back(make_tuple(+1., F.c(leg,UP), fsignTot * F.cdag(leg,UP)));
		this->Otight.push_back(make_tuple(+1., F.c(leg,DN), fsignTot * F.cdag(leg,DN)));
	}
	
	this->Olocal.push_back(make_tuple(1., F.HubbardHamiltonian(U)));
	
//	cout << F.HubbardHamiltonian(U) << endl << endl;
//	cout << U*F.docc(0)+U*F.docc(1) -1.*(F.cdag(0,UP)*F.c(1,UP)+
//	                                      F.cdag(0,DN)*F.c(1,DN)+
//	                                      F.cdag(1,UP)*F.c(0,UP)+ 
//	                                      F.cdag(1,DN)*F.c(0,DN)) << endl;
	
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

#ifndef AUGEROPERATOR
#define AUGEROPERATOR

#include "OperatorFloor.h"
#include "DestructoTron.h"
#include "HubbardModel.h"
#include "DoubleBandHubbardModel.h"

class Auger : public OperatorFloor
{
template<typename> friend class PolynomialBath;

public:
	
	Auger (const HubbardModel &Hbra, const HubbardModel &Hket, int site=-1);
	Auger (const DoubleBandHubbardModel &Hbra, const DoubleBandHubbardModel &Hket, SPIN_INDEX sigma, int site=-1);
	
	Auger (const HoppingParticles &Hbra, const HoppingParticles &Hket, int site=-1);
	
private:
	
	void add_site (const HubbardModel &Hbra, const HubbardModel &Hket, int isite, SparseMatrixXd &Mout);
	void add_site (const DoubleBandHubbardModel &Hbra, const DoubleBandHubbardModel &Hket, SPIN_INDEX sigma, int isite, SparseMatrixXd &Mout);
	
	void add_site (const HoppingParticles &Hbra, const HoppingParticles &Hmid, const HoppingParticles &Hket, int isite, SparseMatrixXd &Mout);
};

Auger::
Auger (const HubbardModel &Hbra, const HubbardModel &Hket, int isite)
:OperatorFloor()
{
	assert
	(
		Hket.subN(UP) == Hbra.subN(UP)+1 and
		Hket.subN(DN) == Hbra.subN(DN)+1 and
		"Mismatched particle numbers in Auger(HubbardModel)!"
	);
	
	storedOperator.resize(Hbra.dim(), Hket.dim());
	
	if (isite == -1)
	{
		SparseMatrixXd Mtemp;
		for (int iL=0; iL<Hket.volume(); ++iL)
		{
			add_site(Hbra,Hket, iL, Mtemp);
			storedOperator += Mtemp;
		}
	}
	else
	{
		add_site(Hbra,Hket, isite, storedOperator);
	}
	
	storedOperator.makeCompressed();
}

Auger::
Auger (const DoubleBandHubbardModel &Hbra, const DoubleBandHubbardModel &Hket, SPIN_INDEX sigma, int isite)
:OperatorFloor()
{
	assert
	(
		Hket.subN(0,sigma)   == Hbra.subN(0,sigma)-1  and
		Hket.subN(0,!sigma)  == Hbra.subN(0,!sigma)   and
		Hket.subN(1, sigma)  == Hbra.subN(1, sigma)+1 and 
		Hket.subN(1,!sigma)  == Hbra.subN(1,!sigma)+1 and
		"Mismatched particle numbers in Auger(DoubleBandHubbardModel)!"
	);
	
	storedOperator.resize(Hbra.dim(), Hket.dim());
	
	if (isite == -1)
	{
		SparseMatrixXd Mtemp;
		for (int iL=0; iL<Hket.volume(); ++iL)
		{
			add_site(Hbra,Hket, sigma,iL, Mtemp);
			storedOperator += Mtemp;
		}
	}
	else
	{
		add_site(Hbra,Hket, sigma,isite, storedOperator);
	}
	
	storedOperator.makeCompressed();
}

Auger::
Auger (const HoppingParticles &Hbra, const HoppingParticles &Hket, int isite)
:OperatorFloor()
{
	HoppingParticles Hmid(Hket.L(), Hket.N()-1, -1., Hket.BC(), BS_UPPER, Hket.D(), Hket.SS());
	
	assert
	(
		Hket.N() == Hbra.N()+2 and
		"Mismatched particle numbers in Auger(HoppingParticles)!"
	);
	
	storedOperator.resize(Hbra.dim(), Hket.dim());
	
	if (isite == -1)
	{
		SparseMatrixXd Mtemp;
		for (int iL=0; iL<Hket.volume(); ++iL)
		{
			add_site(Hbra,Hmid,Hket, iL, Mtemp);
			storedOperator += Mtemp;
		}
	}
	else
	{
		add_site(Hbra,Hmid,Hket, isite, storedOperator);
	}
	
	storedOperator.makeCompressed();
}

void Auger::
add_site (const HubbardModel &Hbra, const HubbardModel &Hket, int isite, SparseMatrixXd &Mout)
{
	DestructoTron Cup(Hbra.basis_ptr(UP), Hket.basis_ptr(UP), isite);
	DestructoTron Cdn(Hbra.basis_ptr(DN), Hket.basis_ptr(DN), isite);
	
	tensor_product(Cup.Operator(), Cdn.Operator(), Mout); // c(UP) x c(DN)
}

void Auger::
add_site (const DoubleBandHubbardModel &Hbra, const DoubleBandHubbardModel &Hket, SPIN_INDEX sigma, int isite, SparseMatrixXd &Mout)
{
	DestructoTron Fsigma (Hbra.basis_ptr(0,sigma), Hket.basis_ptr(0,sigma), isite, CREATE);
	DestructoTron Cup    (Hbra.basis_ptr(1,UP),    Hket.basis_ptr(1,UP),    isite);
	DestructoTron Cdn    (Hbra.basis_ptr(1,DN),    Hket.basis_ptr(1,DN),    isite);

	short int comm_sign;
	
	SparseMatrixXd Cupdn;
	tensor_product(Cup.Operator(), Cdn.Operator(), Cupdn);

	SparseMatrixXd Fupdn;

	if (sigma == UP)
	{
//			comm_sign = pow(-1, Hket.band[1].spin[UP]->N());
//			storedOperator += comm_sign * tensor_product(Cspin, sparse_id<double>(dimCrevspin), Bup, Bdn);
			comm_sign = pow(-1, Hket.subN(1,UP));
			MkronI(Fsigma.Operator(), Hket.subdim(0,!sigma), Fupdn);
	}
	else if (sigma == DN)
	{
//			comm_sign = pow(-1, 1+Hket.band[1].spin[UP]->N()+Hket.band[0].spin[UP]->N());
//			storedOperator += comm_sign * tensor_product(sparse_id<double>(dimCrevspin), Cspin, Bup, Bdn);
			comm_sign = pow(-1, 1 + Hket.subN(1,UP) + Hket.subN(0,UP));
			IkronM(Hket.subdim(0,!sigma), Fsigma.Operator(), Fupdn);
	}
	
	tensor_product(Fupdn,Cupdn, Mout);
	Mout *= comm_sign;
}

void Auger::
add_site (const HoppingParticles &Hbra, const HoppingParticles &Hmid, const HoppingParticles &Hket, int isite, SparseMatrixXd &Mout)
{
	if (isite+1 >= Hket.volume())
	{
		assert(Hket.D() == DIM1 and Hket.BC() == BC_PERIODIC);
	}
	DestructoTron C1(Hmid.basis_ptr(), Hket.basis_ptr(), (isite+1)%Hket.volume(), ANNIHILATE, Hket.SS());
	DestructoTron C2(Hbra.basis_ptr(), Hmid.basis_ptr(), isite,                   ANNIHILATE, Hket.SS());
	Mout = C2.Operator() * C1.Operator();
}

#endif

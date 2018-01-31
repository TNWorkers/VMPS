#ifndef PHOTOCOREVALENCE
#define PHOTOCOREVALENCE

#include "DestructoTron.h"
#include "OperatorFloor.h"
#include "DoubleBandHubbardModel.h"
#include "TensorProducts.h"

class Photo : public OperatorFloor
{
public:
	
	Photo(){};
	
	Photo (const HubbardModel &Hbra, const HubbardModel &Hket, SPIN_INDEX sigma, int isite=-1);
	Photo (const DoubleBandHubbardModel &Hbra, const DoubleBandHubbardModel &Hket, SPIN_INDEX sigma, int isite=-1);
	
	Photo (const HoppingParticles &Hbra, const HoppingParticles &Hket, int isite=-1);
	
private:
	
	void add_site (const HubbardModel &Hbra, const HubbardModel &Hket, SPIN_INDEX sigma, int isite, SparseMatrixXd &Mout);
	void add_site (const DoubleBandHubbardModel &Hbra, const DoubleBandHubbardModel &Hket, SPIN_INDEX sigma, int isite, SparseMatrixXd &Mout);
	
	void add_site (const HoppingParticles &Hbra, const HoppingParticles &Hket, int isite, SparseMatrixXd &Mout);
};

Photo::
Photo (const DoubleBandHubbardModel &Hbra, const DoubleBandHubbardModel &Hket, SPIN_INDEX sigma, int isite)
:OperatorFloor()
{
	assert
	(
		Hket.subN(0,sigma)   == Hbra.subN(0,sigma)+1 and
		Hket.subN(0,!sigma)  == Hbra.subN(0,!sigma)  and
		Hket.subN(1, sigma)  == Hbra.subN(1, sigma)  and 
		Hket.subN(1,!sigma)  == Hbra.subN(1,!sigma)  and
		"Mismatched particle numbers in Photo(DoubleBandHubbardModel)!"
	);
	
	storedOperator.resize(Hbra.dim(),Hket.dim());
	
	if (isite==-1)
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

Photo::
Photo (const HubbardModel &Hbra, const HubbardModel &Hket, SPIN_INDEX sigma, int isite)
:OperatorFloor()
{
	assert
	(
		Hket.subN(sigma)   == Hbra.subN(sigma)+1 and
		Hket.subN(!sigma)  == Hbra.subN(!sigma)  and
		"Mismatched particle numbers in Photo(HubbardModel)!"
	);
	
	storedOperator.resize(Hbra.dim(),Hket.dim());
	
	if (isite==-1)
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

Photo::
Photo (const HoppingParticles &Hbra, const HoppingParticles &Hket, int isite)
:OperatorFloor()
{
	assert
	(
		Hket.N()   == Hbra.N()+1 and
		"Mismatched particle numbers in Photo(HoppingParticles)!"
	);
	
	storedOperator.resize(Hbra.dim(),Hket.dim());
	
	if (isite==-1)
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

void Photo::
add_site (const DoubleBandHubbardModel &Hbra, const DoubleBandHubbardModel &Hket, SPIN_INDEX sigma, int isite, SparseMatrixXd &Mout)
{
	DestructoTron D(Hbra.basis_ptr(0,sigma), Hket.basis_ptr(0,sigma), isite);
	
	switch (sigma)
	{
		case UP: // c(0,UP) x 1(0,DN) x 1(1)
			MkronI(D.Operator(), Hket.subdim(0,!sigma)*Hket.subdim(1), Mout);
			break;
			
		case DN: // 1(0,UP) x c(0,DN) x 1(1)
			SparseMatrixXd Mtemp;
			IkronM(Hket.subdim(0,!sigma), D.Operator(), Mtemp);
			MkronI(Mtemp, Hket.subdim(1), Mout);
			
			short int comm_sign = pow(-1, Hket.subN(0,UP));
			Mout *= comm_sign;
			break;
	}
}

void Photo::
add_site (const HubbardModel &Hbra, const HubbardModel &Hket, SPIN_INDEX sigma, int isite, SparseMatrixXd &Mout)
{
	DestructoTron D(Hbra.basis_ptr(sigma), Hket.basis_ptr(sigma), isite);
	
	switch (sigma)
	{
		case UP: // c(0,UP) x 1(0,DN)
			MkronI(D.Operator(), Hket.subdim(!sigma), Mout);
			break;
			
		case DN: // 1(0,UP) x c(0,DN)
			SparseMatrixXd Mtemp;
			IkronM(Hket.subdim(!sigma), D.Operator(), Mout);
			
			short int comm_sign = pow(-1, Hket.subN(UP));
			Mout *= comm_sign;
			break;
	}
}

void Photo::
add_site (const HoppingParticles &Hbra, const HoppingParticles &Hket, int isite, SparseMatrixXd &Mout)
{
	DestructoTron C(Hbra.basis_ptr(), Hket.basis_ptr(), isite, ANNIHILATE, Hket.SS());
	Mout = C.Operator();
}

#endif

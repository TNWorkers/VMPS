#ifndef STRAWBERRY_DOUBLONMODEL
#define STRAWBERRY_DOUBLONMODEL

#include "MpHeisenbergModel.h"

namespace VMPS
{

class DoublonModel : public MpoQ<1,double>
{
public:
	
	static HamiltonianTermsXd set_operators_Hdeff (double U);
	
	/**
	\param Lx_input : chain length
	\param U : 
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	*/
	DoublonModel (int Lx_input, double U, bool CALC_SQUARE=true);
	
	static const std::array<string,1> Nlabel;
	
	const vector<qarray<1> > qloc();
	
	///@{
	/**Typedefs for convenient reference (no need to specify \p Nq, \p Scalar all the time).*/
	typedef MpsQ<1,double>                           StateXd;
	typedef MpsQ<1,complex<double> >                 StateXcd;
	typedef DmrgSolverQ<1,DoublonModel>              Solver;
	typedef MpsQCompressor<1,double,double>          CompressorXd;
	typedef MpsQCompressor<1,complex<double>,double> CompressorXcd;
	typedef MpoQ<1>                                  Operator;
	///@}
	
	MpoQ<1> n (size_t locx);
	MpoQ<1> d (size_t locx);
	MpoQ<1,complex<double> > doublonPacket (complex<double> (*f)(int));
	
private:
	
	double U;
};

const std::array<string,1> DoublonModel::Nlabel{"N"};

const vector<qarray<1> > DoublonModel::
qloc()
{
	vector<qarray<1> > vout;
	vout.push_back(qarray<1>{1});
	vout.push_back(qarray<1>{0});
	return vout;
};

HamiltonianTermsXd DoublonModel::
set_operators_Hdeff (double U)
{
	assert(U != 0);
	double J = 4./U;
	
	HamiltonianTermsXd Terms;
	
	SparseMatrixXd n(2,2); n.coeffRef(0,0) = 1.;
	SparseMatrixXd d(2,2); d.coeffRef(1,0) = 1.;
	
	Terms.tight.push_back(make_tuple(0.5*J, d.transpose(), d));
	Terms.tight.push_back(make_tuple(0.5*J, d, d.transpose()));
	Terms.tight.push_back(make_tuple(-J, n, n));
	
	Terms.local.push_back(make_tuple(J+U,n));
	
	return Terms;
}

DoublonModel::
DoublonModel (int Lx_input, double U_input, bool CALC_SQUARE)
:MpoQ<1> (Lx_input, 1, DoublonModel::qloc(), {0}, DoublonModel::Nlabel, "", noFormat),
U(U_input)
{
	assert(U != 0.);
	stringstream ss;
	ss << "DoublonModel(U=" << U << ",|J|=" << 4./U << ")";
	this->label = ss.str();
	
	HamiltonianTermsXd Terms = set_operators_Hdeff(U);
	SuperMatrix<double> G = Generator(Terms);
	this->Daux = Terms.auxdim();
	
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

MpoQ<1> DoublonModel::
n (size_t locx)
{
	assert(locx<N_sites);
	stringstream ss;
	ss << "n(" << locx << ")";
	MpoQ<1> Mout(N_sites, 1, DoublonModel::qloc(), {0}, DoublonModel::Nlabel, ss.str(), noFormat);
	SparseMatrixXd n(2,2); n.coeffRef(0,0) = 1.;
	Mout.setLocal(locx,n);
	return Mout;
}

MpoQ<1> DoublonModel::
d (size_t locx)
{
	assert(locx<N_sites);
	stringstream ss;
	ss << "d(" << locx << ")";
	MpoQ<1> Mout(N_sites, 1, DoublonModel::qloc(), {-1}, DoublonModel::Nlabel, ss.str(), noFormat);
	SparseMatrixXd d(2,2); d.coeffRef(1,0) = 1.;
	Mout.setLocal(locx,d);
	return Mout;
}

MpoQ<1,complex<double> > DoublonModel::
doublonPacket (complex<double> (*f)(int))
{
	stringstream ss;
	ss << "doublonPacket";
	MpoQ<1,complex<double> > Mout(N_sites, 1, DoublonModel::qloc(), {-1}, DoublonModel::Nlabel, ss.str(), noFormat);
	SparseMatrixXd d(2,2); d.coeffRef(1,0) = 1.;
	Mout.setLocalSum(d, f);
	return Mout;
}

}

#endif

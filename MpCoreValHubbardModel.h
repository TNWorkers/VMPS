#ifndef STRAWBERRY_COREVALHUBBARDMODEL
#define STRAWBERRY_COREVALHUBBARDMODEL

#include "MpoQ.h"
#include "FermionBase.h"

namespace VMPS
{

class CoreValHubbardModel : public MpoQ<3,double>
{
public:
	
	/**
	\param L_input : chain length
	\param U_input : \f$U\f$
	\param Ucv_input : \f$U_{cv}\f$
	\param Ec_input : \f$E_{c}\f$
	\param V_input : \f$V\f$
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	*/
	CoreValHubbardModel (size_t L_input, double U_input, double Ucv_input, double Ec_input, double V_input=0., bool CALC_SQUARE=true);
	
	static SuperMatrix<double> Generator (double U, double U_cv, double Ec, double V=0.);
	
	MpoQ<3> Hsq();
	
	static const std::array<qarray<3>,8> qloc;
	
	static const std::array<string,3> Nlabel;
	
	typedef MpsQ<3,double>                           StateXd;
	typedef MpsQ<3,complex<double> >                 StateXcd;
	typedef DmrgSolverQ<3,CoreValHubbardModel>       Solver;
	typedef MpsQCompressor<3,double,double>          CompressorXd;
	typedef MpsQCompressor<3,complex<double>,double> CompressorXcd;
	typedef MpoQ<3,double>                           OperatorXd;
	typedef MpoQ<3,complex<double> >                 OperatorXcd;
	
	static MpoQ<3> Auger (size_t L, size_t loc);
	static MpoQ<3> annihilatorCore (size_t L, size_t loc);
	
private:
	
	double U;
	double Ucv;
	double Ec;
	double V=0.;
};

const std::array<qarray<3>,8> CoreValHubbardModel::qloc {qarray<3>{0,0,0}, qarray<3>{0,1,0}, qarray<3>{0,0,1}, qarray<3>{0,1,1},
                                                         qarray<3>{1,0,0}, qarray<3>{1,1,0}, qarray<3>{1,0,1}, qarray<3>{1,1,1}};
const std::array<string,3>    CoreValHubbardModel::Nlabel{"Nf↑","Nc↑","Nc↓"};

SuperMatrix<double> CoreValHubbardModel::
Generator (double U, double Ucv, double Ec, double V)
{
	size_t Daux = 6;
	if (V != 0.) {Daux += 1;}
	
	vector<MatrixXd> col;
	vector<MatrixXd> row;
	MatrixXd Id2(2,2); Id2.setIdentity();
	MatrixXd Id4(4,4); Id4.setIdentity();
	
	// first col (except corner element)
	col.push_back(MatrixXd::Identity(8,8));
	col.push_back(kroneckerProduct(Id2,FermionBase::cUP.transpose()));
	col.push_back(kroneckerProduct(Id2,FermionBase::cDN.transpose()));
	col.push_back(kroneckerProduct(Id2,FermionBase::cUP));
	col.push_back(kroneckerProduct(Id2,FermionBase::cDN));
	if (V != 0.)
	{
		col.push_back(kroneckerProduct(Id2,FermionBase::n));
	}
	
	// last row (except corner element)
	row.push_back(kroneckerProduct(Id2,-FermionBase::fsign * FermionBase::cUP));
	row.push_back(kroneckerProduct(Id2,-FermionBase::fsign * FermionBase::cDN));
	row.push_back(kroneckerProduct(Id2, FermionBase::fsign * FermionBase::cUP.transpose()));
	row.push_back(kroneckerProduct(Id2, FermionBase::fsign * FermionBase::cDN.transpose()));
	if (V != 0.)
	{
		col.push_back(kroneckerProduct(Id2,V * FermionBase::n));
	}
	row.push_back(MatrixXd::Identity(8,8));
	
	SuperMatrix<double> G;
	G.setMatrix(Daux,8);
	G.setZero();
	
	for (size_t i=0; i<Daux-1; ++i)
	{
		G(i,0)        = col[i];
		G(Daux-1,i+1) = row[i];
	}
	
	// corner element
	G(Daux-1,0) = U * kroneckerProduct(Id2,FermionBase::d);
	
	if (Ucv != 0.)
	{
		MatrixXd nf(2,2); nf.setZero();
		nf(0,0) = 1.;
		nf(1,1) = 2.;
		G(Daux-1,0) += Ucv * kroneckerProduct(nf,FermionBase::n);
	}
	if (Ec != 0.)
	{
		MatrixXd Eonsite(2,2); Eonsite.setZero();
		Eonsite(0,0) = Ec;
		Eonsite(1,1) = 2.*Ec;
		G(Daux-1,0) += kroneckerProduct(Eonsite,Id4);
	}
	
	return G;
}

CoreValHubbardModel::
CoreValHubbardModel (size_t L_input, double U_input, double Ucv_input, double Ec_input, double V_input, bool CALC_SQUARE)
:MpoQ<3> (L_input, vector<qarray<3> >(begin(CoreValHubbardModel::qloc),end(CoreValHubbardModel::qloc)), {0,0,0}, CoreValHubbardModel::Nlabel, "CoreValHubbardModel"),
	U(U_input), Ucv(Ucv_input), Ec(Ec_input), V(V_input)
{
	stringstream ss;
	ss << "(U=" << U << ",Ucv=" << Ucv << ",Ec=" << Ec << ",V=" << V << ")";
	this->label += ss.str();
	
	this->Daux = (V==0.)? 6 : 7;
	
	SuperMatrix<double> G = Generator(U,Ucv,Ec,V);
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

MpoQ<3> CoreValHubbardModel::
Hsq()
{
	SuperMatrix<double> G = Generator(U,Ucv,Ec,V);
	MpoQ<3> Mout(this->N_sites, tensor_product(G,G), vector<qarray<3> >(begin(CoreValHubbardModel::qloc),end(CoreValHubbardModel::qloc)), 
	             {0,0,0}, CoreValHubbardModel::Nlabel, "CoreValHubbardModel H^2");
	return Mout;
}

MpoQ<3> CoreValHubbardModel::
annihilatorCore (size_t L, size_t loc)
{
	assert(loc<L);
	stringstream ss;
	ss << "Auger(" << loc << ")";
	
	MatrixXd Id4(4,4); Id4.setIdentity();
	vector<SuperMatrix<double> > M(L);
	for (size_t l=0; l<loc; ++l)
	{
		M[l].setMatrix(1,8);
		M[l](0,0) = kroneckerProduct(FermionBase::fsign,Id4);
	}
	M[loc].setMatrix(1,8);
	M[loc](0,0) = kroneckerProduct(FermionBase::cUP, Id4);
	for (size_t l=loc+1; l<L; ++l)
	{
		M[l].setMatrix(1,8);
		M[l](0,0).setIdentity();
	}
	
	return MpoQ<3>(L, M, vector<qarray<3> >(begin(CoreValHubbardModel::qloc),end(CoreValHubbardModel::qloc)), {-1,0,0}, CoreValHubbardModel::Nlabel, ss.str());
}

MpoQ<3> CoreValHubbardModel::
Auger (size_t L, size_t loc)
{
	assert(loc<L);
	stringstream ss;
	ss << "Auger(" << loc << ")";
	
	MatrixXd Id4(4,4); Id4.setIdentity();
	vector<SuperMatrix<double> > M(L);
	for (size_t l=0; l<loc; ++l)
	{
		M[l].setMatrix(1,8);
		M[l](0,0) = kroneckerProduct(FermionBase::fsign,Id4);
	}
	M[loc].setMatrix(1,8);
	M[loc](0,0) = kroneckerProduct(FermionBase::cUP.transpose(), FermionBase::cUP*FermionBase::cDN);
	for (size_t l=loc+1; l<L; ++l)
	{
		M[l].setMatrix(1,8);
		M[l](0,0).setIdentity();
	}
	
	return MpoQ<3>(L, M, vector<qarray<3> >(begin(CoreValHubbardModel::qloc),end(CoreValHubbardModel::qloc)), {+1,-1,-1}, CoreValHubbardModel::Nlabel, ss.str());
}

}

#endif

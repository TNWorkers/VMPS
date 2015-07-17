#ifndef STRAWBERRY_HUBBARDMODEL
#define STRAWBERRY_HUBBARDMODEL

#include "MpoQ.h"
#include "FermionBase.h"

namespace VMPS
{

/**MPO representation of 
\f$
H = - \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} -t^{\prime} \sum_{<<ij>>\sigma} c^\dagger_{i\sigma}c_{j\sigma} + U \sum_i n_{i\uparrow} n_{i\downarrow}
\f$.
\note If the nnn-hopping is positive, the ground state energy is lowered.*/
class HubbardModel : public MpoQ<2,double>
{
public:
	
	/**
	\param L_input : chain length
	\param U_input : \f$U\f$
	\param V_input : \f$V\f$
	\param tPrime_input : \f$t^{\prime}\f$ next nearest neighbour (nnn) hopping. \f$t^{\prime}>0\f$ is common sign.
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	*/
	HubbardModel (size_t L_input, double U_input, double V_input=0., double tPrime_input=0., bool CALC_SQUARE=true);
	
	static SuperMatrix<double> Generator (double U, double V=0., double tPrime=0.);
	
	MpoQ<2> Hsq();
	
	/**local basis: \f$\{ \left|0,0\right>, \left|\uparrow,0\right>, \left|0,\downarrow\right>, \left|\uparrow\downarrow\right> \}\f$.
	The quantum numbers are \f$N_{\uparrow}\f$ and \f$N_{\downarrow}\f$. Used by default.*/
	static const std::array<qarray<2>,4> qloc;
	
	/**local basis: \f$\{ \left|0,0\right>, \left|\uparrow,0\right>, \left|0,\downarrow\right>, \left|\uparrow\downarrow\right> \}\f$.
	The quantum numbers are \f$N=N_{\uparrow}+N_{\downarrow}\f$ and \f$2M=N_{\uparrow}-N_{\downarrow}\f$. Used in combination with KondoModel.*/
	static const std::array<qarray<2>,4> qlocNM;
	
	/**Labels the conserved quantum numbers as \f$N_\uparrow\f$, \f$N_\downarrow\f$.*/
	static const std::array<string,2> Nlabel;
	
	/**Real MpsQ for convenient reference (no need to specify D, Nq all the time).*/
	typedef MpsQ<2,double>                           StateXd;
	/**Complex MpsQ for convenient reference (no need to specify D, Nq all the time).*/
	typedef MpsQ<2,complex<double> >                 StateXcd;
	typedef DmrgSolverQ<2,HubbardModel>              Solver;
	typedef MpsQCompressor<2,double,double>          CompressorXd;
	typedef MpsQCompressor<2,complex<double>,double> CompressorXcd;
	typedef MpoQ<2,double>                           OperatorXd;
	typedef MpoQ<2,complex<double> >                 OperatorXcd;
	
	static MpoQ<2> Auger (size_t L, size_t loc);
	static MpoQ<2> Aps (size_t L, size_t loc);
	static MpoQ<2> annihilator (size_t L, size_t loc, SPIN_INDEX sigma);
	static MpoQ<2> creator     (size_t L, size_t loc, SPIN_INDEX sigma);
	static MpoQ<2> d (size_t L, size_t loc); // double occupancy
	static MpoQ<2> n (size_t L, SPIN_INDEX sigma, size_t loc);
	static MpoQ<2> Sz (size_t L, size_t loc);
	static MpoQ<2> triplon (size_t L, size_t loc, SPIN_INDEX sigma);
	static MpoQ<2> antitriplon (size_t L, size_t loc, SPIN_INDEX sigma);
	static MpoQ<2> quadruplon (size_t L, size_t loc);
	
private:
	
	double U;
	double V=0.;
	double tPrime=0.;
};

const std::array<qarray<2>,4> HubbardModel::qloc {qarray<2>{0,0}, qarray<2>{1,0}, qarray<2>{0,1}, qarray<2>{1,1}};
const std::array<qarray<2>,4> HubbardModel::qlocNM {qarray<2>{0,0}, qarray<2>{1,1}, qarray<2>{1,-1}, qarray<2>{2,0}};
const std::array<string,2>    HubbardModel::Nlabel{"N↑","N↓"};

SuperMatrix<double> HubbardModel::
Generator (double U, double V, double tPrime)
{
	SuperMatrix<double> G;
	if (V==0)
	{
		if (tPrime==0)
		{
			size_t Daux = 6;
			G.setMatrix(Daux,4);
			G.setZero();
			
			G(0,0).setIdentity();
			G(1,0) = FermionBase::cUP.transpose();
			G(2,0) = FermionBase::cDN.transpose();
			G(3,0) = FermionBase::cUP;
			G(4,0) = FermionBase::cDN;
			
			G(Daux-1,0) = U*FermionBase::d;
			
			// note: FermionBase::fsign takes care of the fermionic sign
			G(Daux-1,1) = FermionBase::fsign * FermionBase::cUP;
			G(Daux-1,2) = FermionBase::fsign * FermionBase::cDN;
			G(Daux-1,3) = -FermionBase::fsign * FermionBase::cUP.transpose();
			G(Daux-1,4) = -FermionBase::fsign * FermionBase::cDN.transpose();
			
			G(Daux-1,Daux-1).setIdentity();
		}
		else
		{
			size_t Daux = 14;
			G.setMatrix(Daux,4);
			G.setZero();
			
			G(0,0).setIdentity();
			G(1,0) = tPrime*FermionBase::cDN;
			G(2,0) = tPrime*FermionBase::cUP;
			G(3,0) = tPrime*FermionBase::cDN.transpose();
			G(4,0) = tPrime*FermionBase::cUP.transpose();
			G(5,0) = FermionBase::cUP.transpose();
			G(6,0) = FermionBase::cDN.transpose();
			G(7,0) = FermionBase::cUP;
			G(8,0) = FermionBase::cDN;
			
			G(13,0) = U*FermionBase::d;
			
			G(13,5) = FermionBase::fsign * FermionBase::cUP;
			G(13,6) = FermionBase::fsign * FermionBase::cDN;
			G(13,7) = -FermionBase::fsign * FermionBase::cUP.transpose();
			G(13,8) = -FermionBase::fsign * FermionBase::cDN.transpose();
			G(13,9) = -FermionBase::fsign * FermionBase::cUP;
			G(13,10) = -FermionBase::fsign * FermionBase::cDN;
			G(13,11) = FermionBase::fsign * FermionBase::cUP.transpose();
			G(13,12) = FermionBase::fsign * FermionBase::cDN.transpose();
			
			G(9,4) = FermionBase::fsign;
			G(10,3) = FermionBase::fsign;
			G(11,2) = FermionBase::fsign;
			G(12,1) = FermionBase::fsign;
			G(13,13).setIdentity();
		}
	}
	
	else
	{
		if (tPrime==0)
		{
			size_t Daux = 7;
			G.setMatrix(Daux,4);
			G.setZero();
			
			G(0,0).setIdentity();
			G(1,0) = FermionBase::cUP.transpose();
			G(2,0) = FermionBase::cDN.transpose();
			G(3,0) = FermionBase::cUP;
			G(4,0) = FermionBase::cDN;
			G(5,0) = FermionBase::n;
			
			G(Daux-1,0) = U*FermionBase::d;
			
			// note: FermionBase::fsign takes care of the fermionic sign
			G(Daux-1,1) = FermionBase::fsign * FermionBase::cUP;
			G(Daux-1,2) = FermionBase::fsign * FermionBase::cDN;
			G(Daux-1,3) = -FermionBase::fsign * FermionBase::cUP.transpose();
			G(Daux-1,4) = -FermionBase::fsign * FermionBase::cDN.transpose();
			G(Daux-1,5) = V*(FermionBase::n);
			G(Daux-1,Daux-1).setIdentity();
		}
		else
		{
			size_t Daux = 15;
			G.setMatrix(Daux,4);
			G.setZero();
			
			G(0,0).setIdentity();
			G(1,0) = tPrime*FermionBase::cDN;
			G(2,0) = tPrime*FermionBase::cUP;
			G(3,0) = tPrime*FermionBase::cDN.transpose();
			G(4,0) = tPrime*FermionBase::cUP.transpose();
			G(5,0) = FermionBase::cUP.transpose();
			G(6,0) = FermionBase::cDN.transpose();
			G(7,0) = FermionBase::cUP;
			G(8,0) = FermionBase::cDN;
			G(9,0) = FermionBase::n;
			
			G(14,0) = U*FermionBase::d;
			
			G(14,5) = FermionBase::fsign * FermionBase::cUP;
			G(14,6) = FermionBase::fsign * FermionBase::cDN;
			G(14,7) = -FermionBase::fsign * FermionBase::cUP.transpose();
			G(14,8) = -FermionBase::fsign * FermionBase::cDN.transpose();
			G(14,9) = V*(FermionBase::n);
			G(14,10) = -FermionBase::fsign * FermionBase::cUP;
			G(14,11) = -FermionBase::fsign * FermionBase::cDN;
			G(14,12) = FermionBase::fsign * FermionBase::cUP.transpose();
			G(14,13) = FermionBase::fsign * FermionBase::cDN.transpose();
			
			G(10,4) = FermionBase::fsign;
			G(11,3) = FermionBase::fsign;
			G(12,2) = FermionBase::fsign;
			G(13,1) = FermionBase::fsign;
			G(14,14).setIdentity();
		}
	}
	return G;
}

HubbardModel::
HubbardModel (size_t L_input, double U_input, double V_input, double tPrime_input, bool CALC_SQUARE)
:MpoQ<2> (L_input, vector<qarray<2> >(begin(HubbardModel::qloc),end(HubbardModel::qloc)), {0,0}, HubbardModel::Nlabel, "HubbardModel"),
	U(U_input), V(V_input), tPrime(tPrime_input)
{
	stringstream ss;
	ss << "(U=" << U << ",V=" << V << ",t'=" << tPrime << ")";
	this->label += ss.str();

	if (tPrime == 0)
	{
		this->Daux = (V==0.)? 6 : 7;
	}
	else
	{
		this->Daux = (V==0.)? 14 : 15;
	}
	
	SuperMatrix<double> G = Generator(U,V,tPrime);
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

MpoQ<2> HubbardModel::
Hsq()
{
	SuperMatrix<double> G = Generator(U,V);
	MpoQ<2> Mout(this->N_sites, tensor_product(G,G), vector<qarray<2> >(begin(HubbardModel::qloc),end(HubbardModel::qloc)), 
	             {0,0}, HubbardModel::Nlabel, "HubbardModel H^2");
	return Mout;
}

MpoQ<2> HubbardModel::
Auger (size_t L, size_t loc)
{
	assert(loc<L);
	stringstream ss;
	ss << "Auger(" << loc << ")";
	MpoQ<2> Mout(L, vector<qarray<2> >(begin(HubbardModel::qloc),end(HubbardModel::qloc)), {-1,-1}, HubbardModel::Nlabel, ss.str());
	Mout.setLocal(loc, FermionBase::cUP*FermionBase::cDN);
	return Mout;
}

MpoQ<2> HubbardModel::
Aps (size_t L, size_t loc)
{
	assert(loc<L);
	stringstream ss;
	ss << "Aps(" << loc << ")";
	MpoQ<2> Mout(L, vector<qarray<2> >(begin(HubbardModel::qloc),end(HubbardModel::qloc)), {+1,+1}, HubbardModel::Nlabel, ss.str());
	Mout.setLocal(loc, FermionBase::cDN.transpose()*FermionBase::cUP.transpose());
	return Mout;
}

MpoQ<2> HubbardModel::
annihilator (size_t L, size_t loc, SPIN_INDEX sigma)
{
	assert(loc<L);
	stringstream ss;
	ss << "c(" << loc << ",σ=" << sigma << ")";
	qarray<2> qdiff;
	(sigma==UP) ? qdiff = {-1,0} : qdiff = {0,-1};
	
	vector<SuperMatrix<double> > M(L);
	for (size_t l=0; l<loc; ++l)
	{
		M[l].setMatrix(1,4);
		M[l](0,0) = FermionBase::fsign;
	}
	M[loc].setMatrix(1,4);
	M[loc](0,0) = (sigma==UP)? FermionBase::cUP : FermionBase::cDN;
	for (size_t l=loc+1; l<L; ++l)
	{
		M[l].setMatrix(1,4);
		M[l](0,0).setIdentity();
	}
	
	return MpoQ<2>(L, M, vector<qarray<2> >(begin(HubbardModel::qloc),end(HubbardModel::qloc)), qdiff, HubbardModel::Nlabel, ss.str());
}

MpoQ<2> HubbardModel::
creator (size_t L, size_t loc, SPIN_INDEX sigma)
{
	assert(loc<L);
	stringstream ss;
	ss << "c†(" << loc << ",σ=" << sigma << ")";
	qarray<2> qdiff;
	(sigma==UP) ? qdiff = {+1,0} : qdiff = {0,+1};
	
	vector<SuperMatrix<double> > M(L);
	for (size_t l=0; l<loc; ++l)
	{
		M[l].setMatrix(1,4);
		M[l](0,0) = FermionBase::fsign;
	}
	M[loc].setMatrix(1,4);
	M[loc](0,0) = (sigma==UP)? FermionBase::cUP.transpose() : FermionBase::cDN.transpose();
	for (size_t l=loc+1; l<L; ++l)
	{
		M[l].setMatrix(1,4);
		M[l](0,0).setIdentity();
	}
	
	return MpoQ<2>(L, M, vector<qarray<2> >(begin(HubbardModel::qloc),end(HubbardModel::qloc)), qdiff, HubbardModel::Nlabel, ss.str());
}

MpoQ<2> HubbardModel::
triplon (size_t L, size_t loc, SPIN_INDEX sigma)
{
	assert(loc<L);
	stringstream ss;
	ss << "triplon(" << loc << ")" << "c(" << loc+1 << ",σ=" << sigma << ")";
	qarray<2> qdiff;
	(sigma==UP) ? qdiff = {-2,-1} : qdiff = {-1,-2};
	
	vector<SuperMatrix<double> > M(L);
	for (size_t l=0; l<loc; ++l)
	{
		M[l].setMatrix(1,4);
		M[l](0,0) = FermionBase::fsign;
	}
	// c(loc,UP)*c(loc,DN)
	M[loc].setMatrix(1,4);
	M[loc](0,0) = FermionBase::cUP*FermionBase::cDN;
	// c(loc+1,UP|DN)
	M[loc+1].setMatrix(1,4);
	M[loc+1](0,0) = (sigma==UP)? FermionBase::cUP : FermionBase::cDN;
	for (size_t l=loc+2; l<L; ++l)
	{
		M[l].setMatrix(1,4);
		M[l](0,0).setIdentity();
	}
	
	return MpoQ<2>(L, M, vector<qarray<2> >(begin(HubbardModel::qloc),end(HubbardModel::qloc)), qdiff, HubbardModel::Nlabel, ss.str());
}

MpoQ<2> HubbardModel::
antitriplon (size_t L, size_t loc, SPIN_INDEX sigma)
{
	assert(loc<L);
	stringstream ss;
	ss << "antitriplon(" << loc << ")" << "c(" << loc+1 << ",σ=" << sigma << ")";
	qarray<2> qdiff;
	(sigma==UP) ? qdiff = {+2,+1} : qdiff = {+1,+2};
	
	vector<SuperMatrix<double> > M(L);
	for (size_t l=0; l<loc; ++l)
	{
		M[l].setMatrix(1,4);
		M[l](0,0) = FermionBase::fsign;
	}
	// c†(loc,DN)*c†(loc,UP)
	M[loc].setMatrix(1,4);
	M[loc](0,0) = FermionBase::cDN.transpose()*FermionBase::cUP.transpose();
	// c†(loc+1,UP|DN)
	M[loc+1].setMatrix(1,4);
	M[loc+1](0,0) = (sigma==UP)? FermionBase::cUP.transpose() : FermionBase::cDN.transpose();
	for (size_t l=loc+2; l<L; ++l)
	{
		M[l].setMatrix(1,4);
		M[l](0,0).setIdentity();
	}
	
	return MpoQ<2>(L, M, vector<qarray<2> >(begin(HubbardModel::qloc),end(HubbardModel::qloc)), qdiff, HubbardModel::Nlabel, ss.str());
}

MpoQ<2> HubbardModel::
quadruplon (size_t L, size_t loc)
{
	assert(loc<L);
	stringstream ss;
	ss << "Auger(" << loc << ")" << "Auger(" << loc+1 << ")";
	
	vector<SuperMatrix<double> > M(L);
	for (size_t l=0; l<loc; ++l)
	{
		M[l].setMatrix(1,4);
		M[l](0,0).setIdentity();
	}
	// c(loc,UP)*c(loc,DN)
	M[loc].setMatrix(1,4);
	M[loc](0,0) = FermionBase::cUP*FermionBase::cDN;
	// c(loc+1,UP)*c(loc+1,DN)
	M[loc+1].setMatrix(1,4);
	M[loc+1](0,0) = FermionBase::cUP*FermionBase::cDN;
	for (size_t l=loc+2; l<L; ++l)
	{
		M[l].setMatrix(1,4);
		M[l](0,0).setIdentity();
	}
	
	return MpoQ<2>(L, M, vector<qarray<2> >(begin(HubbardModel::qloc),end(HubbardModel::qloc)), {-2,-2}, HubbardModel::Nlabel, ss.str());
}

MpoQ<2> HubbardModel::
d (size_t L, size_t loc)
{
	assert(loc<L);
	stringstream ss;
	ss << "double_occ(" << loc << ")";
	MpoQ<2> Mout(L, vector<qarray<2> >(begin(HubbardModel::qloc),end(HubbardModel::qloc)), {0,0}, HubbardModel::Nlabel, ss.str());
	Mout.setLocal(loc, FermionBase::d);
	return Mout;
}

MpoQ<2> HubbardModel::
n (size_t L, SPIN_INDEX sigma, size_t loc)
{
	assert(loc<L);
	stringstream ss;
	ss << "n(" << loc << ",σ=" << sigma << ")";
	MpoQ<2> Mout(L, vector<qarray<2> >(begin(HubbardModel::qloc),end(HubbardModel::qloc)), {0,0}, HubbardModel::Nlabel, ss.str());
	(sigma==UP)? Mout.setLocal(loc, FermionBase::cUP.transpose()*FermionBase::cUP):
	             Mout.setLocal(loc, FermionBase::cDN.transpose()*FermionBase::cDN);
	return Mout;
}

MpoQ<2> HubbardModel::
Sz (size_t L, size_t loc)
{
	assert(loc<L);
	stringstream ss;
	ss << "Sz(" << loc << ")";
	MpoQ<2> Mout(L, vector<qarray<2> >(begin(HubbardModel::qloc),end(HubbardModel::qloc)), {0,0}, HubbardModel::Nlabel, ss.str());
	Mout.setLocal(loc, FermionBase::Sz);
	return Mout;
}

}

#endif

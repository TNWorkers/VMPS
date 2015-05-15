#ifndef STRAWBERRY_KONDOMODEL
#define STRAWBERRY_KONDOMODEL

#include "MpHubbardModel.h"
#include "MpHeisenbergModel.h"

namespace VMPS
{

class KondoModel : public MpoQ<8,2,double>
{
public:
	
	KondoModel (size_t L_input, double J_input, double hz_input=0., bool CALC_SQUARE=true);
	
	template<size_t D> static SuperMatrix<D*4> Generator (const Eigen::Matrix<double,D,D,RowMajor> &Sp, 
	                                                      const Eigen::Matrix<double,D,D,RowMajor> &Sz,
	                                                      double J, double hz);
	
	static std::array<string,2> N_halveM (qarray<2> qnum);
	static const std::array<string,2> NMlabel;
	static const std::array<qarray<2>,8> qloc;
	
	/**Real MpsQ for convenient reference (no need to specify D, Nq all the time).*/
	typedef MpsQ<8,2,double>                           StateXd;
	/**Complex MpsQ for convenient reference (no need to specify D, Nq all the time).*/
	typedef MpsQ<8,2,complex<double> >                 StateXcd;
	typedef DmrgSolverQ<8,2,KondoModel>                Solver;
	typedef MpsQCompressor<8,2,double,double>          CompressorXd;
	typedef MpsQCompressor<8,2,complex<double>,double> CompressorXcd;
	typedef MpoQ<8,2>                                  Operator;
	
	class qarrayIterator;
	
private:
	
	double J, hz;
};

const std::array<qarray<2>,8> KondoModel::qloc
{
	qarray<2>{0,+1},
	qarray<2>{1,+2},
	qarray<2>{1, 0},
	qarray<2>{2,+1},
	
	qarray<2>{0,-1},
	qarray<2>{1, 0},
	qarray<2>{1,-2},
	qarray<2>{2,-1}
};

const std::array<string,2> KondoModel::NMlabel{"N","M"};

template<size_t D>
SuperMatrix<D*4> KondoModel::
Generator (const Eigen::Matrix<double,D,D,RowMajor> &Sp, 
           const Eigen::Matrix<double,D,D,RowMajor> &Sz, 
           double J, double hz)
{
	size_t Daux = 6;
	SuperMatrix<D*4> G;
	G.setMatrix(Daux);
	G.setZero();
	
	MatrixXd Id4(4,4); Id4.setIdentity();
	MatrixXd IdSpins(Sz.rows(), Sz.cols()); IdSpins.setIdentity();
	
	G(0,0).setIdentity();
	G(1,0) = kroneckerProduct(HubbardModel::cUP.transpose(), IdSpins);
	G(2,0) = kroneckerProduct(HubbardModel::cDN.transpose(), IdSpins);
	G(3,0) = kroneckerProduct(HubbardModel::cUP, IdSpins);
	G(4,0) = kroneckerProduct(HubbardModel::cDN, IdSpins);
	
	G(Daux-1,0) = -0.5*J * kroneckerProduct(HubbardModel::Sp.transpose(), Sp)
	              -0.5*J * kroneckerProduct(HubbardModel::Sp, Sp.transpose())
	              -J * kroneckerProduct(HubbardModel::Sz, Sz)
//	              +hz * kroneckerProduct(HubbardModel::Sz, Id2)
	              +hz * kroneckerProduct(Id4, Sz);
	
	// note: fsign takes care of the fermionic sign
	G(Daux-1,1) = kroneckerProduct( HubbardModel::fsign * HubbardModel::cUP, IdSpins);
	G(Daux-1,2) = kroneckerProduct( HubbardModel::fsign * HubbardModel::cDN, IdSpins);
	G(Daux-1,3) = kroneckerProduct(-HubbardModel::fsign * HubbardModel::cUP.transpose(), IdSpins);
	G(Daux-1,4) = kroneckerProduct(-HubbardModel::fsign * HubbardModel::cDN.transpose(), IdSpins);
	G(Daux-1,Daux-1).setIdentity();
	
	return G;
}

KondoModel::
KondoModel (size_t L_input, double J_input, double hz_input, bool CALC_SQUARE)
:MpoQ<8,2> (L_input, KondoModel::qloc, {0,0}, KondoModel::NMlabel, "KondoModel"),
J(J_input), hz(hz_input)
{
	stringstream ss;
	ss << "(J=" << J << ",hz=" << hz << ")";
	this->label += ss.str();
	
	this->Daux = 6;
	this->N_sv = this->Daux;
	
	SuperMatrix<8> G = Generator<2>(HeisenbergModel::Sp, HeisenbergModel::Sz, J, hz);
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

std::array<string,2> KondoModel::
N_halveM (qarray<2> qnum)
{
	std::array<string,2> out;
	stringstream ssN;
	ssN << qnum[0];
	out[0] == ssN.str();
	
	stringstream ssM;
	rational<int> m = rational<int>(qnum[1],2);
	if (m.numerator()==0) {ssM << 0;}
	else if (m.denominator()==1) {ssM << m.numerator();}
	else {ssM << m;}
	out[1] = ssM.str();
	
	return out;
}

class KondoModel::qarrayIterator
{
public:
	
	qarrayIterator (std::array<qarray<2>,8> qloc_dummy, int L_input)
	:N_sites(L_input)
	{
		for (int Sz=-N_sites; Sz<=N_sites; Sz+=2)
		for (int Nup=0; Nup<=N_sites; ++Nup)
		for (int Ndn=0; Ndn<=N_sites; ++Ndn)
		{
			qarray<2> q = {Nup+Ndn, Sz+Nup-Ndn};
			qarraySet.insert(q);
		}
		
		it = qarraySet.begin();
	};
	
	qarray<2> operator*() {return value;}
	
	qarrayIterator& operator= (const qarray<2> a) {value=a;}
	bool operator!=          (const qarray<2> a) {return value!=a;}
	bool operator<=          (const qarray<2> a) {return value<=a;}
	bool operator<           (const qarray<2> a) {return value< a;}
	
	qarray<2> begin()
	{
		return *(qarraySet.begin());
	}
	
	qarray<2> end()
	{
		return *(qarraySet.end());
	}
	
	void operator++()
	{
		++it;
		value = *it;
	}
	
private:
	
	qarray<2> value;
	
	set<qarray<2> > qarraySet;
	set<qarray<2> >::iterator it;
	
	int N_sites;
};

};

#endif

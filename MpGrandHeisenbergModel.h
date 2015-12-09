#ifndef VANILLA_GRANDHEISENBERGMODEL
#define VANILLA_GRANDHEISENBERGMODEL

#include "MpHeisenbergModel.h"

namespace VMPS
{

/**MPO representation of 
\f$
H = - J_{xy} \sum_{<ij>} \left(S^x_iS^x_j+S^y_iS^y_j\right) - J_z \sum_{<ij>} S^z_iS^z_j - B_z \sum_i S^z_i - B_x \sum_i S^x_i 
\f$.
\param D : \f$D=2S+1\f$ where \f$S\f$ is the spin
\note \f$J<0\f$ : antiferromagnetic*/
class GrandHeisenbergModel : public MpoQ<0,double>
{
public:
	
	GrandHeisenbergModel (int L_input, double Jxy_input=-1., double Jz_input=numeric_limits<double>::infinity(), 
	                      double Bz_input=0., double Bx_input=0., bool CALC_SQUARE=true, size_t D_input=2);
	
	///@{
	/**Typedef for convenient reference (no need to specify \p Nq, \p Scalar all the time).*/
	typedef MpsQ<0,double>                           StateXd;
	typedef MpsQ<0,complex<double> >                 StateXcd;
	typedef DmrgSolverQ<0,GrandHeisenbergModel>      Solver;
	typedef MpsQCompressor<0,double,double>          CompressorXd;
	typedef MpsQCompressor<0,complex<double>,double> CompressorXcd;
	typedef MpoQ<0>                                  Operator;
	///@}
	
	MpoQ<0> SzSz (size_t loc1, size_t loc2);
	MpoQ<0> Sz   (size_t loc);
	
private:
	
	double Jxy, Jz;
	double Bz, Bx;
	size_t D;
	SpinBase S;
};

GrandHeisenbergModel::
GrandHeisenbergModel (int L_input, double Jxy_input, double Jz_input, double Bz_input, double Bx_input, bool CALC_SQUARE, size_t D_input)
	:MpoQ<0> (L_input, 1, vector<qarray<0> >(begin(qloc2dummy),end(qloc2dummy)), {}, labeldummy, ""),
	Jxy(Jxy_input), Jz(Jz_input), Bz(Bz_input), Bx(Bx_input), D(D_input)
{
	if (Jz==numeric_limits<double>::infinity()) {Jz=Jxy;} // default: Jxy=Jz
	assert(Jxy != 0. or Jz != 0.);
	this->label = HeisenbergModel::create_label(D,Jxy,Jz,0.,Bz,Bx);
	
	S = SpinBase(1,D);
	HamiltonianTermsXd Terms = HeisenbergModel::set_operators(S, Jxy,Jz,Bz,Bx);

	SuperMatrix<double> G = ::Generator(Terms);
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

MpoQ<0> GrandHeisenbergModel::
Sz (size_t loc)
{
	assert(loc<N_sites);
	stringstream ss;
	ss << "Sz(" << loc << ")";
	MpoQ<0> Mout(N_sites, 1, vector<qarray<0> >(begin(qloc2dummy),end(qloc2dummy)), {}, labeldummy, "");
	Mout.setLocal(loc, S.Scomp(SZ));
	return Mout;
}

MpoQ<0> GrandHeisenbergModel::
SzSz (size_t loc1, size_t loc2)
{
	assert(loc1<N_sites and loc2<N_sites);
	stringstream ss;
	ss << "Sz(" << loc1 << ")" <<  "Sz(" << loc2 << ")";
	MpoQ<0> Mout(N_sites, 1, vector<qarray<0> >(begin(qloc2dummy),end(qloc2dummy)), {}, labeldummy, "");
	Mout.setLocal({loc1, loc2}, {S.Scomp(SZ), S.Scomp(SZ)});
	return Mout;
}

}

#endif

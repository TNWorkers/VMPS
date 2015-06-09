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
template<int D=2>
class GrandHeisenbergModel : public MpoQ<0,double>
{
public:
	
	GrandHeisenbergModel (int L_input, double Jxy_input=-1., double Jz_input=numeric_limits<double>::infinity(), 
	                      double Bz_input=0., double Bx_input=0., bool CALC_SQUARE=true);
	
	///@{
	/**Typedef for convenient reference (no need to specify \p Nq, \p Scalar all the time).*/
	typedef MpsQ<0,double>                           StateXd;
	typedef MpsQ<0,complex<double> >                 StateXcd;
	typedef DmrgSolverQ<0,GrandHeisenbergModel>      Solver;
	typedef MpsQCompressor<0,double,double>          CompressorXd;
	typedef MpsQCompressor<0,complex<double>,double> CompressorXcd;
	typedef MpoQ<0>                                  Operator;
	///@}
	
	static const std::array<qarray<0>,2> qloc;
	
private:
	
	double Jxy, Jz;
	double Bz, Bx;
};

template<int D>
GrandHeisenbergModel<D>::
GrandHeisenbergModel (int L_input, double Jxy_input, double Jz_input, double Bz_input, double Bx_input, bool CALC_SQUARE)
:MpoQ<0> (L_input, vector<qarray<0> >(begin(qloc2dummy),end(qloc2dummy)), {}, labeldummy, ""),
Jxy(Jxy_input), Jz(Jz_input), Bz(Bz_input), Bx(Bx_input)
{
	if (Jz==numeric_limits<double>::infinity()) {Jz=Jxy;} // default: Jxy=Jz
	assert(Jxy != 0. or Jz != 0.);
	this->label = HeisenbergModel::create_label(frac(D-1,2),Jxy,Jz,Bz,Bx);
	
	this->Daux = HeisenbergModel::calc_Daux(Jxy,Jz);
	this->N_sv = this->Daux;
	
	SuperMatrix<double> G = HeisenbergModel<D>::Generator(Jxy,Jz,Bz,Bx);
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

}

#endif

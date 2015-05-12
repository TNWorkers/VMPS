#ifndef VANILLA_GRANDHUBBARDMODEL
#define VANILLA_GRANDHUBBARDMODEL

#include "Mpo.h"
#include "MpHubbardModel.h"
#include "MpsCompressor.h"

namespace VMPS
{

class GrandHubbardModel : public Mpo<4>
{
public:
	
	GrandHubbardModel (size_t L_input, double U_input, double V_input=0., bool CALC_SQUARE=true);
	
	typedef Mps<4,double>                     StateXd;
	typedef Mps<4,complex<double> >           StateXcd;
	typedef DmrgSolver<4,GrandHubbardModel>   Solver;
	typedef MpsCompressor<4,double>           CompressorXd;
	typedef MpsCompressor<4,complex<double> > CompressorXcd;
	typedef Mpo<4>                          Operator;
	
private:
	
	double U, V;
};

GrandHubbardModel::
GrandHubbardModel (size_t L_input, double U_input, double V_input, bool CALC_SQUARE)
:Mpo<4>(L_input,6), U(U_input) V(V_input)
{
	if (V!=0.) {Daux=7;}
	SuperMatrix<4> G = HubbardModel::Generator(U,V);
	
	construct(G, this->W, this->Gvec);
	
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

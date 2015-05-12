#ifndef VANILLA_GRANDHEISENBERGMODEL
#define VANILLA_GRANDHEISENBERGMODEL

#include "Mpo.h"
#include "Mps.h"
#include "MpHeisenbergSector.h"
#include "MpsCompressor.h"

namespace VMPS
{

class GrandHeisenbergModel : public Mpo<2>
{
public:
	
	GrandHeisenbergModel (int L_input, double Jxy_input=-1., double Jz_input=numeric_limits<double>::infinity(), double hz_input=0., double hx_input=0., bool CALC_SQUARE=true);
	
	typedef Mps<2,double>                     StateXd;
	typedef Mps<2,complex<double> >           StateXcd;
	typedef DmrgSolver<2,GrandHeisenbergModel>     Solver;
	typedef MpsCompressor<2,double>           CompressorXd;
	typedef MpsCompressor<2,complex<double> > CompressorXcd;
	typedef Mpo<2>                            Operator;
	
private:
	
	double Jxy, Jz;
	double hz, hx;
};

GrandHeisenbergModel::
GrandHeisenbergModel (int L_input, double Jxy_input, double Jz_input, double hz_input, double hx_input, bool CALC_SQUARE)
:Mpo<2>(L_input,5), Jxy(Jxy_input), Jz(Jz_input), hz(hz_input), hx(hx_input)
{
	if (Jz==numeric_limits<double>::infinity()) {Jz=Jxy;} // default: Jxy=Jz
	
	SuperMatrix<2> G = HeisenbergModel::Generator(Jxy,Jz,hz,hx);
	construct(G, this->W, this->Gvec);
	
	if (CALC_SQUARE == true)
	{
		this->construct<25>(tensor_product(G,G), this->Wsq, this->GvecSq);
		this->GOT_SQUARE = true;
	}
	else
	{
		this->GOT_SQUARE = false;
	}
}

}

#endif

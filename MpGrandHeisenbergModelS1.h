#ifndef VANILLA_GRANDHEISENBERGMODEL_SPINONE
#define VANILLA_GRANDHEISENBERGMODEL_SPINONE

#include "Mpo.h"
#include "Mps.h"
#include "MpHeisenbergSectorS1.h"
#include "MpsCompressor.h"

namespace VMPS
{

class GrandHeisenbergModelS1 : public Mpo<3>
{
public:
	
	GrandHeisenbergModelS1 (int L_input, 
	                   double Jxy_input=-1., double Jz_input=numeric_limits<double>::infinity(), 
	                   double hz_input=0., double hx_input=0., 
	                   bool CALC_SQUARE=true);
	
	typedef Mps<3,double>                     StateXd;
	typedef Mps<3,complex<double> >           StateXcd;
	typedef DmrgSolver<3,GrandHeisenbergModelS1>   Solver;
	typedef MpsCompressor<3,double>           CompressorXd;
	typedef MpsCompressor<3,complex<double> > CompressorXcd;
	typedef Mpo<3>                            Operator;
	
private:
	
	double Jxy, Jz;
	double hz, hx;
};

GrandHeisenbergModelS1::
GrandHeisenbergModelS1 (int L_input, double Jxy_input, double Jz_input, double hz_input, double hx_input, bool CALC_SQUARE)
:Mpo<3>(L_input,5), Jxy(Jxy_input), Jz(Jz_input), hz(hz_input), hx(hx_input)
{
	if (Jz==numeric_limits<double>::infinity()) {Jz=Jxy;} // default: Jxy=Jz
	
	SuperMatrix<3> G = HeisenbergModelS1::Generator(Jxy,Jz,hz,hx);
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

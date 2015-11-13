#ifndef VANILLA_GRANDHUBBARDMODEL
#define VANILLA_GRANDHUBBARDMODEL

#include "MpHubbardModel.h"

namespace VMPS
{

class GrandHubbardModel : public MpoQ<0,double>
{
public:
	
	GrandHubbardModel (size_t L_input, double U_input, double V_input=0., bool CALC_SQUARE=true);
	
	typedef MpsQ<0,double>                     StateXd;
	typedef MpsQ<0,complex<double> >           StateXcd;
	typedef DmrgSolverQ<0,GrandHubbardModel>   Solver;
	typedef MpsQCompressor<0,double>           CompressorXd;
	typedef MpsQCompressor<0,complex<double> > CompressorXcd;
	typedef MpoQ<0>                            Operator;
	
private:
	
	double U, V;
};

GrandHubbardModel::
GrandHubbardModel (size_t L_input, double U_input, double V_input, bool CALC_SQUARE)
:MpoQ<0> (L_input, vector<qarray<0> >(begin(qloc4dummy),end(qloc4dummy)), {}, labeldummy, "HubbardModel"),
U(U_input), V(V_input)
{
	stringstream ss;
	ss << "(U=" << U << ",V=" << V << ")";
	this->label += ss.str();
	
	HubbardModel::set_operators(Olocal,Otight,Onextn, U,V);
	this->Daux = 2 + Otight.size() + 2*Onextn.size();

	SuperMatrix<double> G = ::Generator(Olocal,Otight,Onextn);
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

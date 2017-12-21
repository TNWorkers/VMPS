#include "SuperMatrix.h"
#include "MpHubbardModel.h"

namespace VMPS
{
/**MPO representation of 
\f$
H = -t \sum_{(i,j)\in \mathcal{T},\sigma} c^\dagger_{i\sigma}c_{j\sigma} -t^{\prime} \sum_{(i,j)\in \mathcal{T^{\prime}},\sigma} c^\dagger_{i\sigma}c_{j\sigma} - V \sum_{i} c^\dagger_{i\sigma}  f_{i\sigma} + h.c. + \frac{U}{2} \sum_{i,\sigma}n^{f}_{i\sigma}n^{f}_{i-\sigma}
\f$.
*/
class AndersonModel : public HubbardModel
{
public:

	/**Does nothing.*/
	AndersonModel() : HubbardModel(){};

	/**Constructs an Anderson Lattice Model.
	\param L_input : chain length
	\param Hyb_input : \f$V\f$ (hybridization term)
	\param U_input : \f$U\f$ (local Hubbard interaction)
	\param tPrime_input : nnn hopping \f$t^{\prime}\f$ (\f$t^{\prime}>0\f$ is common sign).
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$*/
	AndersonModel (size_t L_input, double Hyb_input=-1., double U_input=0., double tPrime_input=1., bool CALC_SQUARE=false);
	
	///@{
	/**Typedef for convenient reference (no need to specify \p Nq, \p Scalar all the time).*/
	typedef DmrgSolverQ<2,AndersonModel> Solver;
	typedef MpsQ<2,double> StateXd;
	typedef MpoQ<2> Operator;
	///@}
	
protected:
	
	double Hyb;
	vector<double> Uvec;
	MatrixXd tInter;
};

AndersonModel::
AndersonModel (size_t L_input, double Hyb_input, double U_input, double tPrime_input, bool CALC_SQUARE)
	:HubbardModel(),Hyb(Hyb_input)
{
	// assign stuff
	this->N_sites = L_input;
	this->N_legs = 2;
	this->Qtot = {0,0};
	this->qlabel = HubbardModel::Nlabel;
	this->label = "Anderson Model";
	this->format = noFormat;
	this->tPrime = tPrime_input;
	this->U = U_input;

	stringstream ss;
	ss << "(U=" << U << ",V=" << Hyb << ",t'=" << tPrime << ")";
	this->label += ss.str();
	
	F = FermionBase(N_legs);
	MpoQ<2>::qloc.resize(N_sites);
	for (size_t l=0; l<this->N_sites; ++l)
	{
		MpoQ<2>::qloc[l].resize(F.dim());
		for (size_t i=0; i<F.dim(); i++)
		{
			MpoQ<2>::qloc[l][i] = F.qNums(i,false);
		}
	}
	
	Uvec.resize(N_legs); Uvec[1]=U;
	tInter.resize(N_legs,N_legs); tInter.setZero(); tInter(0,0) = 1.;
	
	HamiltonianTermsXd Terms = set_operators(F,Uvec,tInter,0.,tPrime,Hyb);
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

}

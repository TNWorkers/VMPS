#ifndef STRAWBERRY_HEISENBERGMODEL
#define STRAWBERRY_HEISENBERGMODEL

#include <boost/rational.hpp>
typedef boost::rational<int> frac;

#include "MpoQ.h"
#include "SpinBase.h"

namespace VMPS
{

/**MPO representation of 
\f$
H = - J_{xy} \sum_{<ij>} \left(S^x_iS^x_j+S^y_iS^y_j\right) - J_z \sum_{<ij>} S^z_iS^z_j - B_z \sum_i S^z_i
\f$.
\param D : \f$D=2S+1\f$ where \f$S\f$ is the spin
\note \f$J<0\f$ : antiferromagnetic*/
template<size_t D=2>
class HeisenbergModel : public MpoQ<1,double>
{
public:
	
	/**
	\param L_input : chain length
	\param Jxy_input : \f$J_{xy}\f$, default \f$J_{xy}=-1\f$
	\param Jz_input : \f$J_z\f$, default \f$J_{xy}=J_z\f$ (Heisenberg, otherwise XXZ)
	\param Bz_input : external field in z-direction
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	*/
	HeisenbergModel (int L_input, double Jxy_input=-1., double Jz_input=numeric_limits<double>::infinity(), double Bz_input=0., bool CALC_SQUARE=true);
	
	/**Creates the MPO generator matrix for the Heisenberg model (of any spin)
	\f$G = \left(
	\begin{array}{ccccc}
	1 & 0 & 0 & 0 & 0 \\
	S^+ & 0 & 0 & 0 & 0 \\
	S^- & 0 & 0 & 0 & 0 \\
	S^z & 0 & 0 & 0 & 0 \\
	h_zS^z+h_xS^x & -\frac{J_{xy}}{2}S^- & -\frac{J_{xy}}{2}/2S^+ & -\frac{J_z}{2}S^z & 1
	\end{array}
	\right)\f$.
	The fourth row and column are missing when \f$J_{xy}=0\f$. Uses the appropriate spin operators for a given \p S.*/
	static SuperMatrix<double> Generator (double Jxy, double Jz, double Bz, double Bx);
	
	//---label stuff---
	///@{
	/**Creates a label for this MpoQ to have a nice output.
	\param S : spin
	\param Jz : \f$J_z\f$
	\param Jxy : \f$J_{xy}\f$
	\param Bz : \f$B_{z}\f$
	\param Bx : \f$B_{x}\f$ (when called by GrandHeisenbergModel, otherwise 0)*/
	static string create_label (frac S, double Jxy, double Jz, double Bz, double Bx)
	{
		stringstream ss;
		if      (Jz == Jxy) {ss << "Heisenberg(S=" << S << ",J=" << Jz;}
		else if (Jxy == 0.) {ss << "Ising(S=" << S << ",J=" << Jz;}
		else if (Jz == 0.)  {ss << "XX(S=" << S << ",J=" << Jxy;}
		else                {ss << "XXZ(S=" << S << ",Jxy=" << Jxy << ",Jz=" << Jz;}
		if (Bz != 0.) {ss << ",Bz=" << Bz;}
		if (Bx != 0.) {ss << ",Bx=" << Bx;}
		ss << ")";
		return ss.str();
	}
	/**local basis: \f$\{ \left|\uparrow\right>, \left|\downarrow\right> \}\f$*/
	static const std::array<qarray<1>,D> qloc;
	/**Makes half-integers in the output.*/
	static string halve (qarray<1> qnum);
	/**Labels the conserved quantum number as "M".*/
	static const std::array<string,1> maglabel;
	///@}
	
	MpoQ<1> Hsq();
	
	///@{
	/**Typedef for convenient reference (no need to specify \p Nq, \p Scalar all the time).*/
	typedef MpsQ<1,double>                           StateXd;
	typedef MpsQ<1,complex<double> >                 StateXcd;
	typedef DmrgSolverQ<1,HeisenbergModel>           Solver;
	typedef MpsQCompressor<1,double,double>          CompressorXd;
	typedef MpsQCompressor<1,complex<double>,double> CompressorXcd;
	typedef MpoQ<1>                                  Operator;
	///@}
	
	/**Calculates the necessary auxiliary dimension, detecting when \p Jxy or \p Jz are zero.*/
	static size_t calc_Daux (double Jxy, double Jz)
	{
		size_t res = 2;
		res += (Jxy!=0.)? 2 : 0;
		res += (Jz !=0.)? 1 : 0;
		return res;
	}
	
private:
	
	double Jxy, Jz;
	double Bz;
};

template<size_t D> const std::array<string,1> HeisenbergModel<D>::maglabel{"M"};

template<> const std::array<qarray<1>,2> HeisenbergModel<2>::qloc {qarray<1>{+1}, qarray<1>{-1}};
template<> const std::array<qarray<1>,3> HeisenbergModel<3>::qloc {qarray<1>{+2}, qarray<1>{0}, qarray<1>{-2}};

template<size_t D>
SuperMatrix<double> HeisenbergModel<D>::
Generator (double Jxy, double Jz, double Bz, double Bx)
{
	SuperMatrix<double> G;
	size_t Daux = calc_Daux(Jxy,Jz);
	G.setMatrix(Daux,D);
	G.setZero();
	
	// left column
	G(0,0).setIdentity();
	if (Jxy != 0.)
	{
		G(1,0) = SpinBase<D>::Sp;
		G(2,0) = SpinBase<D>::Sp.transpose();
		if (Jz!=0.) {G(3,0) = SpinBase<D>::Sz;}
	}
	else
	{
		G(1,0) = SpinBase<D>::Sz;
	}
	
	// corner element
	G(Daux-1,0) = -Bz*SpinBase<D>::Sz -Bx*SpinBase<D>::Sx;
	
	// last row
	if (Jxy != 0.)
	{
		G(Daux-1,1) = -0.5*Jxy*SpinBase<D>::Sp.transpose();
		G(Daux-1,2) = -0.5*Jxy*SpinBase<D>::Sp;
		if (Jz!=0.) {G(Daux-1,3) = -Jz*SpinBase<D>::Sz;}
	}
	else
	{
		G(Daux-1,1) = SpinBase<D>::Sz;
	}
	G(Daux-1,Daux-1).setIdentity();
	
	return G;
}

template<size_t D>
HeisenbergModel<D>::
HeisenbergModel (int L_input, double Jxy_input, double Jz_input, double Bz_input, bool CALC_SQUARE)
:MpoQ<1> (L_input, vector<qarray<1> >(begin(HeisenbergModel<D>::qloc),end(HeisenbergModel<D>::qloc)), {0}, HeisenbergModel::maglabel, "", HeisenbergModel::halve),
Jxy(Jxy_input), Jz(Jz_input), Bz(Bz_input)
{
	if (Jz==numeric_limits<double>::infinity()) {Jz=Jxy;} // default: Jxy=Jz
	assert(Jxy != 0. or Jz != 0.);
	this->label = create_label(frac(D-1,2),Jxy,Jz,Bz,0.);
	
	this->Daux = calc_Daux(Jxy,Jz);
	this->N_sv = this->Daux;
	
	SuperMatrix<double> G = Generator(Jxy,Jz,Bz,0.);
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

template<size_t D>
MpoQ<1> HeisenbergModel<D>::
Hsq()
{
	SuperMatrix<double> W = Generator(Jxy,Jz,Bz,0.);
	MpoQ<1> Mout(this->N_sites, tensor_product(W,W), 
	             vector<qarray<1> >(begin(HeisenbergModel<D>::qloc),end(HeisenbergModel<D>::qloc)), 
	             {0}, HeisenbergModel::maglabel, "", HeisenbergModel::halve);
	Mout.label = create_label(frac(D-1,2),Jxy,Jz,Bz,0.) + "H^2";
	return Mout;
}

template<size_t D>
string HeisenbergModel<D>::
halve (qarray<1> qnum)
{
	stringstream ss;
	ss << "(";
	rational<int> m = rational<int>(qnum[0],2);
	if      (m.numerator()   == 0) {ss << 0;}
	else if (m.denominator() == 1) {ss << m.numerator();}
	else {ss << m;}
	ss << ")";
	return ss.str();
}

}

#endif

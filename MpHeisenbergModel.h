#ifndef STRAWBERRY_HEISENBERGMODEL
#define STRAWBERRY_HEISENBERGMODEL

#include <boost/rational.hpp>
typedef boost::rational<int> frac;

#include "MpoQ.h"

namespace VMPS
{

/**MPO representation of \f$H = - J_{xy} \sum_{<ij>} \left(S^x_iS^x_j+S^y_iS^y_j\right) - J_z \sum_{<ij>} S^z_iS^z_j \f$.
\note S=1/2 implicit
\note \f$J<0\f$ : antiferromagnetic*/
class HeisenbergModel : public MpoQ<2,1,double>
{
public:
	
	/**
	\param L_input : chain length
	\param Jxy_input : \f$J_{xy}\f$, default \f$J_{xy}=-1\f$
	\param Jz_input : \f$J_z\f$, default \f$J_{xy}=J_z\f$ (Heisenberg, otherwise XXZ)
	\param hz_input : external field in z-direction
	\param hx_input : external field in x-direction
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	*/
	HeisenbergModel (int L_input,
	                 double Jxy_input=-1., double Jz_input=numeric_limits<double>::infinity(), 
	                 double hz_input=0., double hx_input=0.,
	                 bool CALC_SQUARE=true);
	
	///@{
	/**
	\f$S^z = \left(
	\begin{array}{cc}
	0.5 & 0 \\
	0 & -0.5 \\
	\end{array}
	\right)\f$
	*/
	static const Eigen::Matrix<double,2,2,RowMajor> Sz;
	/**
	\f$S^+ = \left(
	\begin{array}{cc}
	0 & 1 \\
	0 & 0 \\
	\end{array}
	\right)\f$
	*/
	static const Eigen::Matrix<double,2,2,RowMajor> Sp;
	/**
	\f$S^x = \left(
	\begin{array}{cc}
	0 & 0.5 \\
	0.5 & 0 \\
	\end{array}
	\right)\f$
	*/
	static const Eigen::Matrix<double,2,2,RowMajor> Sx;
	///@}
	
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
	template<size_t D> static SuperMatrix<D> Generator (const Eigen::Matrix<double,D,D,RowMajor> &Sz, 
	                                                    const Eigen::Matrix<double,D,D,RowMajor> &Sp, 
	                                                    const Eigen::Matrix<double,D,D,RowMajor> &Sx, 
	                                                    double Jxy, double Jz, double hz, double hx);
	
	//---label stuff---
	///@{
	/**Creates a label for this MpoQ to have a nice output.
	\param S : spin*/
	static string create_label (frac S, double Jz, double Jxy)
	{
		stringstream ss;
		if (Jz == Jxy)   {ss << "Heisenberg(S=" << S << ",J=" << Jz << ")";}
		else if (Jz==0.) {ss << "XY(S=" << S << ",J=" << Jxy << ")";}
		else             {ss << "XXZ(S=" << S << ",Jxy=" << Jxy << ",Jz=" << Jz << ")";}
		return ss.str();
	}
	/**local basis: \f$\{ \left|\uparrow\right>, \left|\downarrow\right> \}\f$*/
	static const std::array<qarray<1>,2> qloc;
	/**Makes half-integers in the output.*/
	static string halve (qarray<1> qnum);
	/**Labels the conserved quantum number as "M".*/
	static const std::array<string,1> maglabel;
	///@}
	
	MpoQ<2,1> Hsq();
	
	/**Real MpsQ for convenient reference (no need to specify D, Nq all the time).*/
	typedef MpsQ<2,1,double>                           StateXd;
	/**Complex MpsQ for convenient reference (no need to specify D, Nq all the time).*/
	typedef MpsQ<2,1,complex<double> >                 StateXcd;
	typedef DmrgSolverQ<2,1,HeisenbergModel>           Solver;
	typedef MpsQCompressor<2,1,double,double>          CompressorXd;
	typedef MpsQCompressor<2,1,complex<double>,double> CompressorXcd;
	typedef MpoQ<2,1>                                  Operator;
	
private:
	
	double Jxy, Jz;
	double hz, hx;
};

static const double Sz_data[] = {0.5, 0.,  0., -0.5};
static const double Sp_data[] = {0.,  1.,  0.,  0.};
static const double Sx_data[] = {0.,  0.5, 0.5, 0.};

const Eigen::Matrix<double,2,2,RowMajor> HeisenbergModel::Sz(Sz_data);
const Eigen::Matrix<double,2,2,RowMajor> HeisenbergModel::Sp(Sp_data);
const Eigen::Matrix<double,2,2,RowMajor> HeisenbergModel::Sx(Sx_data);

const std::array<qarray<1>,2>   HeisenbergModel::qloc {qarray<1>{+1}, qarray<1>{-1}};
const std::array<string,1>      HeisenbergModel::maglabel{"M"};

template<size_t D>
SuperMatrix<D,double> HeisenbergModel::
Generator (const Eigen::Matrix<double,D,D,RowMajor> &Sz, 
           const Eigen::Matrix<double,D,D,RowMajor> &Sp, 
           const Eigen::Matrix<double,D,D,RowMajor> &Sx, 
           double Jxy, double Jz, double hz, double hx)
{
	SuperMatrix<D,double> G;
	size_t Daux = (Jz==0.)? 4 : 5;
	G.setMatrix(Daux);
	G.setZero();
	
	G(0,0).setIdentity();
	G(1,0) = Sp;
	G(2,0) = Sp.transpose();
	if (Jz!=0.) {G(3,0) = Sz;}
	
	G(Daux-1,0) = hz*Sz + hx*Sx;
	
	G(Daux-1,1) = -0.5*Jxy*Sp.transpose();
	G(Daux-1,2) = -0.5*Jxy*Sp;
	if (Jz!=0.) {G(Daux-1,3) = -Jz*Sz;}
	G(Daux-1,Daux-1).setIdentity();
	
	return G;
}

HeisenbergModel::
HeisenbergModel (int L_input, double Jxy_input, double Jz_input, double hz_input, double hx_input, bool CALC_SQUARE)
:MpoQ<2,1> (L_input, HeisenbergModel::qloc, {0}, HeisenbergModel::maglabel, "", HeisenbergModel::halve),
Jxy(Jxy_input), Jz(Jz_input), hz(hz_input), hx(hx_input)
{
	if (Jz==numeric_limits<double>::infinity()) {Jz=Jxy;} // default: Jxy=Jz
	this->label = create_label(frac(1,2),Jz,Jxy);
	
	this->Daux = (Jz==0.)? 4 : 5;
	this->N_sv = this->Daux;
	
	SuperMatrix<2> G = Generator<2>(Sz,Sp,Sx, Jxy,Jz,hz,hx);
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

// doesn't work that way if hx!=0
MpoQ<2,1> HeisenbergModel::
Hsq()
{
	SuperMatrix<2> W = Generator<2>(Sz,Sp,Sx, Jxy,Jz,hz,hx);
	MpoQ<2,1> Mout(this->N_sites, tensor_product(W,W), HeisenbergModel::qloc, {0}, HeisenbergModel::maglabel, "HeisenbergModel(S=1/2)^2", HeisenbergModel::halve);
	return Mout;
}

string HeisenbergModel::
halve (qarray<1> qnum)
{
	stringstream ss;
	ss << "(";
	rational<int> m = rational<int>(qnum[0],2);
	if (m.numerator()==0) {ss << 0;}
	else if (m.denominator()==1) {ss << m.numerator();}
	else {ss << m;}
	ss << ")";
	return ss.str();
}

}

#endif

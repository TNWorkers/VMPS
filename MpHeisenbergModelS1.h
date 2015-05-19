#ifndef STRAWBERRY_HEISENBERGMODEL_SPINONE
#define STRAWBERRY_HEISENBERGMODEL_SPINONE

#include "MpoQ.h"
#include "MpHeisenbergModel.h"

namespace VMPS
{

/**MPO representation of
\f$
H = - J_{xy} \sum_{<ij>} \left(S^x_iS^x_j+S^y_iS^y_j\right) - J_z \sum_{<ij>} S^z_iS^z_j - B_z \sum_i S^z_i - B_x \sum_i S^x_i 
\f$.
\note S=1
\note \f$J<0\f$ : antiferromagnetic*/
class HeisenbergModelS1 : public MpoQ<1,double>
{
public:
	
	/**
	\param L_input : chain length
	\param Jxy_input : \f$J_{xy}\f$, default \f$J_{xy}=-1\f$
	\param Jz_input : \f$J_z\f$, default \f$J_{xy}=J_z\f$ (Heisenberg, otherwise XXZ)
	\param Bz_input : external field in z-direction
	\param Bx_input : external field in x-direction
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	*/
	HeisenbergModelS1 (int L_input,
	                   double Jxy_input=-1., double Jz_input=numeric_limits<double>::infinity(), 
	                   double Bz_input=0., double Bx_input=0.,
	                   bool CALC_SQUARE=true);
	
	/**
	\f$S^z = \left(
	\begin{array}{ccc}
	1 & 0 & 0 \\
	0 & 0 & 0 \\
	0 & 0 & -1 \\
	\end{array}
	\right)\f$
	*/
	static const Eigen::Matrix<double,3,3,RowMajor> Sz;
	
	/**
	\f$S^+ = \sqrt{2}\left(
	\begin{array}{ccc}
	0 & 1 & 0 \\
	0 & 0 & 1 \\
	0 & 0 & 0 \\
	\end{array}
	\right)\f$
	*/
	static const Eigen::Matrix<double,3,3,RowMajor> Sp;
	
	/**
	\f$S^x = \frac{1}{\sqrt{2}} \left(
	\begin{array}{ccc}
	0 & 1 & 0 \\
	1 & 0 & 1 \\
	0 & 1 & 0 \\
	\end{array}
	\right)\f$
	*/
	static const Eigen::Matrix<double,3,3,RowMajor> Sx;
	
	/**local basis: \f$\{ \left|\uparrow\right>, \left|0\right>, \left|\downarrow\right> \}\f$*/
	static const std::array<qarray<1>,3> qloc;
	
	MpoQ<1> Hsq();
	
	///@{
	/**Typedef for convenient reference (no need to specify \p Nq, \p Scalar all the time).*/
	typedef MpsQ<1,double>                           StateXd;
	typedef MpsQ<1,complex<double> >                 StateXcd;
	typedef DmrgSolverQ<1,HeisenbergModelS1>         Solver;
	typedef MpsQCompressor<1,double,double>          CompressorXd;
	typedef MpsQCompressor<1,complex<double>,double> CompressorXcd;
	typedef MpoQ<1>                                  Operator;
	///@}
	
private:
	
	double Jxy, Jz;
	double Bz, Bx;
};

static const double SzS1_data[] = 
{1., 0.,  0., 
 0., 0.,  0.,
 0., 0., -1.};
 
static const double SpS1_data[] = 
{0., M_SQRT2, 0., 
 0., 0.,      M_SQRT2,
 0., 0.,      0.};
 
static const double SxS1_data[] = 
{0.,        M_SQRT1_2, 0., 
 M_SQRT1_2, 0.,        M_SQRT1_2,
 0.,        M_SQRT1_2, 0.};

const Eigen::Matrix<double,3,3,RowMajor> HeisenbergModelS1::Sz(SzS1_data);
const Eigen::Matrix<double,3,3,RowMajor> HeisenbergModelS1::Sp(SpS1_data);
const Eigen::Matrix<double,3,3,RowMajor> HeisenbergModelS1::Sx(SxS1_data);

const std::array<qarray<1>,3> HeisenbergModelS1::qloc {qarray<1>{+2}, qarray<1>{0}, qarray<1>{-2}}; // Q=2*M

HeisenbergModelS1::
HeisenbergModelS1 (int L_input, double Jxy_input, double Jz_input, double Bz_input, double Bx_input, bool CALC_SQUARE)
:MpoQ<1> (L_input, vector<qarray<1> >(begin(HeisenbergModelS1::qloc),end(HeisenbergModelS1::qloc)), {0}, HeisenbergModel::maglabel, ""),
Jxy(Jxy_input), Jz(Jz_input), Bz(Bz_input), Bx(Bx_input)
{
	if (Jz==numeric_limits<double>::infinity()) {Jz=Jxy;} // default: Jxy=Jz
	this->label = HeisenbergModel::create_label(1,Jz,Jxy);
	
	SuperMatrix<double> G = HeisenbergModel::Generator<3>(Sz,Sp,Sx, Jxy,Jz,Bz,Bx);
	this->construct(G, this->W, this->Gvec);
	
	this->Daux = (Jz==0.)? 4 : 5;
	this->N_sv = this->Daux;
	
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

// does it work if Bx!=0 ?
MpoQ<1> HeisenbergModelS1::
Hsq()
{
	SuperMatrix<double> W = HeisenbergModel::Generator<3>(Sz,Sp,Sx, Jxy,Jz,Bz,Bx);
	MpoQ<1> Mout(this->N_sites, tensor_product(W,W),  
	             vector<qarray<1> >(begin(HeisenbergModelS1::qloc),end(HeisenbergModelS1::qloc)), 
	             {0}, HeisenbergModel::maglabel, "HeisenbergModel(S=1) H^2");
	return Mout;
}

}

#endif

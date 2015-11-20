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
H = -J_{xy} \sum_{<ij>} \left(S^x_iS^x_j+S^y_iS^y_j\right) - J_z \sum_{<ij>} S^z_iS^z_j -J' \sum_{<<ij>>} \left(\mathbf{S_i}\mathbf{S_j}\right) - B_z \sum_i S^z_i
\f$.
\param D : \f$D=2S+1\f$ where \f$S\f$ is the spin
\note \f$J<0\f$ : antiferromagnetic*/
class HeisenbergModel : public MpoQ<1,double>
{
public:
	
	/**
	\param L_input : chain length
	\param Jxy_input : \f$J_{xy}\f$, default \f$J_{xy}=-1\f$
	\param Jz_input : \f$J_z\f$, default \f$J_{xy}=J_z\f$ (Heisenberg, otherwise XXZ)
	\param Bz_input : external field in z-direction
	\param D_input : \f$2S+1\f$
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	*/
	HeisenbergModel (int L_input, double Jxy_input=-1., double Jz_input=numeric_limits<double>::infinity(), double Bz_input=0., size_t D_input=2, 
	                 size_t Ly_input=1, bool CALC_SQUARE=true);
	
	/**
	\param L_input : chain length
	\param Jlist : list of next-/second-nearerst neighbour exchange interactions
	\param Bz_input : external field in z-direction
	\param D_input : \f$2S+1\f$
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	*/
	HeisenbergModel (int Lx_input, array<double,2> Jlist, double Bz_input=0., size_t D_input=2, bool CALC_SQUARE=true);
	
	/**Creates the MPO generator matrix for the Heisenberg model (of any spin (\f$D=2S+1\f$))
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
	static SuperMatrix<double> Generator (double Jxy, double Jz, double Bz, double Bx, size_t D=2);
	
//	SuperMatrix<double> GeneratorJ12 (double J, double Jprime, double Bz);
	
	static void set_operators (LocalTermsXd &Olocal, TightTermsXd &Otight, NextnTermsXd &Onextn, 
	                           double Jxy, double Jz, double Bz, double Bx, size_t D=2, double Jprime=0.,
	                           size_t N_legs=1);
	
	//---label stuff---
	///@{
	/**Creates a label for this MpoQ to have a nice output.
	\param D : \f$2S+1\f$
	\param Jz : \f$J_z\f$
	\param Jxy : \f$J_{xy}\f$
	\param Jprime : \f$J'\f$
	\param Bz : \f$B_{z}\f$
	\param Bx : \f$B_{x}\f$ (when called by GrandHeisenbergModel, otherwise 0)*/
	static string create_label (size_t D, double Jxy, double Jz, double Jprime, double Bz, double Bx)
	{
		auto S = frac(D-1,2);
		stringstream ss;
		if      (Jz == Jxy) {ss << "Heisenberg(S=" << S << ",J=" << Jz;}
		else if (Jxy == 0.) {ss << "Ising(S=" << S << ",J=" << Jz;}
		else if (Jz == 0.)  {ss << "XX(S=" << S << ",J=" << Jxy;}
		else                {ss << "XXZ(S=" << S << ",Jxy=" << Jxy << ",Jz=" << Jz;}
		if (Bz != 0.) {ss << ",Bz=" << Bz;}
		if (Bx != 0.) {ss << ",Bx=" << Bx;}
		if (Jprime != 0.) {ss << ",J'=" << Jprime;}
		ss << ")";
		return ss.str();
	}
	
	/**local basis: \f$\{ \left|\uparrow\right>, \left|\downarrow\right> \}\f$*/
	static const vector<qarray<1> > qloc (size_t D=2)
	{
		vector<qarray<1> > vout;
		int Sx2 = static_cast<int>(D-1);
		for (int M=Sx2; M>=-Sx2; M-=2)
		{
			vout.push_back(qarray<1>{M});
		}
		return vout;
	};
	
	/**Makes half-integers in the output.*/
	static string halve (qarray<1> qnum);
	
	/**Labels the conserved quantum number as "M".*/
	static const std::array<string,1> maglabel;
	///@}
	
	MpoQ<1> Hsq (size_t D=2);
	
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
	
	MpoQ<1> Sz (size_t locx, size_t locy=0);
	MpoQ<1> SzSz (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	
private:
	
	double Jxy=-1., Jz=-1., Bz=0.;
	double Jprime=0.;
	size_t D=2;
	
	size_t N_legs = 1;
};

const std::array<string,1> HeisenbergModel::maglabel{"M"};

//template<> const std::array<qarray<1>,2> HeisenbergModel<2>::qloc = {qarray<1>{+1}, qarray<1>{-1}};
//template<> const std::array<qarray<1>,3> HeisenbergModel<3>::qloc = {qarray<1>{+2}, qarray<1>{0}, qarray<1>{-2}};

//SuperMatrix<double> HeisenbergModel::
//Generator (double Jxy, double Jz, double Bz, double Bx, size_t D)
//{
//	SuperMatrix<double> G;
//	size_t Daux = calc_Daux(Jxy,Jz);
//	G.setMatrix(Daux,D);
//	G.setZero();
//	
//	// left column
//	G(0,0).setIdentity();
//	if (Jxy != 0.)
//	{
//		G(1,0) = SpinBase::Scomp(SP,D);
//		G(2,0) = SpinBase::Scomp(SM,D);
//		if (Jz!=0.) {G(3,0) = SpinBase::Scomp(SZ,D);}
//	}
//	else
//	{
//		G(1,0) = SpinBase::Scomp(SZ,D);
//	}
//	
//	// corner element
//	G(Daux-1,0) = -Bz*SpinBase::Scomp(SZ,D) -Bx*SpinBase::Scomp(SX,D);
//	
//	// last row
//	if (Jxy != 0.)
//	{
//		G(Daux-1,1) = -0.5*Jxy*SpinBase::Scomp(SM,D);
//		G(Daux-1,2) = -0.5*Jxy*SpinBase::Scomp(SP,D);
//		if (Jz!=0.) {G(Daux-1,3) = -Jz*SpinBase::Scomp(SZ,D);}
//	}
//	else
//	{
//		G(Daux-1,1) = SpinBase::Scomp(SZ,D);
//	}
//	G(Daux-1,Daux-1).setIdentity();
//	
//	return G;
//}

//SuperMatrix<double> HeisenbergModel::
//GeneratorJ12 (double J, double Jprime, double Bz)
//{
//	SuperMatrix<double> G;
//	size_t Daux = 11;
//	G.setMatrix(Daux,D);
//	G.setZero();
//	
//	// left column
//	G(0,0).setIdentity();
//	
//	G(1,0) = SpinBase::Scomp(SP,D);
//	G(2,0) = SpinBase::Scomp(SM,D);
//	G(3,0) = SpinBase::Scomp(SZ,D);
//	
//	G(4,0) = SpinBase::Scomp(SP,D);
//	G(5,0) = SpinBase::Scomp(SM,D);
//	G(6,0) = SpinBase::Scomp(SZ,D);
//	
//	// corner element
//	G(Daux-1,0) = -Bz*SpinBase::Scomp(SZ,D);
//	
//	// last row
//	G(Daux-1,4) = -0.5*J*SpinBase::Scomp(SM,D);
//	G(Daux-1,5) = -0.5*J*SpinBase::Scomp(SP,D);
//	G(Daux-1,6) = -J*SpinBase::Scomp(SZ,D);
//	
//	G(Daux-1,7) = -0.5*Jprime*SpinBase::Scomp(SM,D);
//	G(Daux-1,8) = -0.5*Jprime*SpinBase::Scomp(SP,D);
//	G(Daux-1,9) = -Jprime*SpinBase::Scomp(SZ,D);
//	
//	// id-block for J':
//	G.set_block_to_skewdiag(9,1, 3, Eigen::MatrixXd::Identity(D,D));
//	
//	G(Daux-1,Daux-1).setIdentity();
//	
//	return G;
//}

SparseMatrixXd embed (const SparseMatrixXd &M, size_t locy, size_t N_legs)
{
	assert(locy<N_legs);
	
	size_t D = M.rows();
	size_t Nl = D*locy;
	size_t Nr = D*(N_legs-locy-1);
	
	SparseMatrixXd Il = MatrixXd::Identity(Nl,Nl).sparseView();
	SparseMatrixXd Ir = MatrixXd::Identity(Nr,Nr).sparseView();
	
	// all = 0
	if (Nl == 0 and Nr == 0)
	{
		return M;
	}
	// one != 0
	else if (Nl == 0 and Nr != 0)
	{
		return kroneckerProduct(M,Ir);
	}
	else if (Nl != 0 and Nr == 0)
	{
		return kroneckerProduct(Il,M);
	}
	// all != 0
	else
	{
		return kroneckerProduct(Il,kroneckerProduct(M,Ir));
	}
}

SparseMatrixXd embed (const SparseMatrixXd &M1, size_t locy1, const SparseMatrixXd &M2, size_t locy2, size_t N_legs)
{
	assert(locy1<N_legs and locy2<N_legs);
	assert(M1.rows() == M1.cols() and M2.rows() == M2.cols() and M1.rows() == M2.rows());
	
	size_t D = M1.rows();
	size_t Nl = D*locy1;
	size_t Nr = D*(N_legs-locy2-1);
	size_t Nm = D*N_legs-Nl-Nr-2*D;
	
	SparseMatrixXd Il = MatrixXd::Identity(Nl,Nl).sparseView();
	SparseMatrixXd Im = MatrixXd::Identity(Nm,Nm).sparseView();
	SparseMatrixXd Ir = MatrixXd::Identity(Nr,Nr).sparseView();
	
	// all = 0
	if (Nl == 0 and Nm == 0 and Nr == 0)
	{
		return kroneckerProduct(M1,M2);
	}
	// one != 0
	else if (Nl != 0 and Nm == 0 and Nr == 0)
	{
		return kroneckerProduct(Il,kroneckerProduct(M1,M2));
	}
	else if (Nl == 0 and Nm == 0 and Nr != 0)
	{
		return kroneckerProduct(M1,kroneckerProduct(M2,Ir));
	}
	else if (Nl == 0 and Nm != 0 and Nr == 0)
	{
		return kroneckerProduct(M1,kroneckerProduct(Im,M2));
	}
	// two != 0
	else if (Nl != 0 and Nm != 0 and Nr == 0)
	{
		return kroneckerProduct(Il,kroneckerProduct(M1,kroneckerProduct(Im,M2)));
	}
	else if (Nl != 0 and Nm == 0 and Nr != 0)
	{
		return kroneckerProduct(Il,kroneckerProduct(M1,kroneckerProduct(M2,Ir)));
	}
	else if (Nl == 0 and Nm != 0 and Nr != 0)
	{
		return kroneckerProduct(M1,kroneckerProduct(Im,kroneckerProduct(M2,Ir)));
	}
	// all != 0
	else
	{
		return kroneckerProduct(Il,kroneckerProduct(M1,kroneckerProduct(Im,kroneckerProduct(M2,Ir))));
	}
}

void HeisenbergModel::
set_operators (LocalTermsXd &Olocal, TightTermsXd &Otight, NextnTermsXd &Onextn, double Jxy, double Jz, double Bz, double Bx, size_t D, double Jprime, size_t N_legs)
{
	// interaction along legs
	for (size_t leg=0; leg<N_legs; ++leg)
	{
		if (Jxy != 0.)
		{
			SparseMatrixXd Sp = embed(SpinBase::Scomp(SP,D), leg, N_legs);
			SparseMatrixXd Sm = embed(SpinBase::Scomp(SM,D), leg, N_legs);
			Otight.push_back(make_tuple(-0.5*Jxy, Sp, Sm));
			Otight.push_back(make_tuple(-0.5*Jxy, Sm, Sp));
		}
		if (Jz != 0.)
		{
			SparseMatrixXd Sz = embed(SpinBase::Scomp(SZ,D), leg, N_legs);
			Otight.push_back(make_tuple(-Jz, Sz, Sz));
		}
		
		// local B-terms
		if (Bz != 0.)
		{
			SparseMatrixXd Sz = embed(SpinBase::Scomp(SZ,D), leg, N_legs);
			Olocal.push_back(make_tuple(-Bz, Sz));
		}
		if (Bx != 0.)
		{
			SparseMatrixXd Sx = embed(SpinBase::Scomp(SX,D), leg, N_legs);
			Olocal.push_back(make_tuple(-Bx, Sx));
		}
	}
	
	// interaction along rungs
	for (size_t leg=0; leg<N_legs-1; ++leg)
	{
		if (Jxy != 0.)
		{
			SparseMatrixXd SpSm = embed(SpinBase::Scomp(SP,D), leg, SpinBase::Scomp(SM,D), leg+1, N_legs);
			SparseMatrixXd SmSp = embed(SpinBase::Scomp(SM,D), leg, SpinBase::Scomp(SP,D), leg+1, N_legs);
			Olocal.push_back(make_tuple(-0.5*Jxy, SpSm));
			Olocal.push_back(make_tuple(-0.5*Jxy, SmSp));
		}
		if (Jz != 0.)
		{
			SparseMatrixXd SzSz = embed(SpinBase::Scomp(SZ,D), leg, SpinBase::Scomp(SZ,D), leg+1, N_legs);
			Olocal.push_back(make_tuple(-Jz, SzSz));
		}
	}
	
	if (Jprime != 0.)
	{
		SparseMatrixXd Id = MatrixXd::Identity(D,D).sparseView();
		Onextn.push_back(make_tuple(-0.5*Jprime, SpinBase::Scomp(SP,D), SpinBase::Scomp(SM,D), Id));
		Onextn.push_back(make_tuple(-0.5*Jprime, SpinBase::Scomp(SM,D), SpinBase::Scomp(SP,D), Id));
		Onextn.push_back(make_tuple(-Jprime,     SpinBase::Scomp(SZ,D), SpinBase::Scomp(SZ,D), Id));
	}
}

HeisenbergModel::
HeisenbergModel (int Lx_input, double Jxy_input, double Jz_input, double Bz_input, size_t D_input, size_t Ly_input, bool CALC_SQUARE)
:MpoQ<1> (Lx_input, HeisenbergModel::qloc(D_input), {0}, HeisenbergModel::maglabel, "", HeisenbergModel::halve),
Jxy(Jxy_input), Jz(Jz_input), Bz(Bz_input), D(D_input), N_legs(Ly_input)
{
	if (Jz==numeric_limits<double>::infinity()) {Jz=Jxy;} // default: Jxy=Jz
	assert(Jxy != 0. or Jz != 0.);
	this->label = create_label(D,Jxy,Jz,0,Bz,0);
	
	set_operators(Olocal,Otight,Onextn, Jxy,Jz,Bz,0.,D, N_legs);
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

HeisenbergModel::
HeisenbergModel (int L_input, array<double,2> Jlist, double Bz_input, size_t D_input, bool CALC_SQUARE)
:MpoQ<1> (L_input, HeisenbergModel::qloc(2), {0}, HeisenbergModel::maglabel, "", HeisenbergModel::halve),
Jxy(Jlist[0]), Jz(Jlist[0]), Bz(Bz_input), D(D_input), Jprime(Jlist[1])
{
	this->label = create_label(D,Jxy,Jz,Jprime,Bz,0.);
	
	set_operators(Olocal,Otight,Onextn, Jxy,Jz,Bz,0.,D,Jprime);
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

MpoQ<1> HeisenbergModel::
Hsq (size_t D)
{
	SuperMatrix<double> G = ::Generator(Olocal,Otight,Onextn);
	MpoQ<1> Mout(this->N_sites, tensor_product(G,G), HeisenbergModel::qloc(D), {0}, HeisenbergModel::maglabel, "", HeisenbergModel::halve);
	Mout.label = create_label(D,Jxy,Jz,Jprime,Bz,0.) + "H^2";
	return Mout;
}

string HeisenbergModel::
halve (qarray<1> qnum)
{
	stringstream ss;
	ss << "(";
	boost::rational<int> m = boost::rational<int>(qnum[0],2);
	if      (m.numerator()   == 0) {ss << 0;}
	else if (m.denominator() == 1) {ss << m.numerator();}
	else {ss << m;}
	ss << ")";
	return ss.str();
}

MpoQ<1> HeisenbergModel::
Sz (size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << "Sz(" << locx << "," << locy << ")";
	MpoQ<1> Mout(N_sites, HeisenbergModel::qloc(D), {0}, HeisenbergModel::maglabel, ss.str(), HeisenbergModel::halve);
	Mout.setLocal(locx, embed(SpinBase::Scomp(SZ,D),locy,N_legs));
	return Mout;
}

MpoQ<1> HeisenbergModel::
SzSz (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
{
	assert(locx1<N_sites and locx2<N_sites and locy1<N_legs and locy2<N_legs);
	stringstream ss;
	ss << "Sz(" << locx1 << "," << locy1 << ")" <<  "Sz(" << locx2 << "," << locy2 << ")";
	MpoQ<1> Mout(N_sites, HeisenbergModel::qloc(D), {0}, HeisenbergModel::maglabel, ss.str(), HeisenbergModel::halve);
	Mout.setLocal({locx1, locx2}, {embed(SpinBase::Scomp(SZ,D),locy1,N_legs), embed(SpinBase::Scomp(SZ,D),locy2,N_legs)});
	return Mout;
}

}

#endif

#ifndef STRAWBERRY_HEISENBERGMODEL
#define STRAWBERRY_HEISENBERGMODEL

#include "array"
#include "MpoQ.h"
#include "SpinBase.h"
#include "DmrgExternalQ.h"

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
	
	HeisenbergModel() : MpoQ<1>() {};
	
	/**
	\param Lx_input : chain length
	\param Jxy_input : \f$J_{xy}\f$, default \f$J_{xy}=-1\f$
	\param Jz_input : \f$J_z\f$, default \f$J_{xy}=J_z\f$ (Heisenberg, otherwise XXZ)
	\param Bz_input : external field in z-direction
	\param D_input : \f$2S+1\f$
	\param Ly_input : amount of legs in ladder
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	*/
	HeisenbergModel (int Lx_input, double Jxy_input=-1., double Jz_input=numeric_limits<double>::infinity(), double Bz_input=0., size_t D_input=2, 
	                 size_t Ly_input=1, bool CALC_SQUARE=true);
	
	/**
	\param Lx_input : chain length
	\param Jlist : list containing \f$J\f$ and \f$J'\f$
	\param Bz_input : external field in z-direction
	\param Ly_input : amount of legs in ladder
	\param D_input : \f$2S+1\f$
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	*/
	HeisenbergModel (size_t Lx_input, std::array<double,2> Jlist, double Bz_input=0., size_t Ly_input=1, size_t D_input=2, bool CALC_SQUARE=true);
	
//	/**Creates the MPO generator matrix for the Heisenberg model (of any spin (\f$D=2S+1\f$))
//	\f$G = \left(
//	\begin{array}{ccccc}
//	1 & 0 & 0 & 0 & 0 \\
//	S^+ & 0 & 0 & 0 & 0 \\
//	S^- & 0 & 0 & 0 & 0 \\
//	S^z & 0 & 0 & 0 & 0 \\
//	h_zS^z+h_xS^x+K(S^z)^2 & -\frac{J_{xy}}{2}S^- & -\frac{J_{xy}}{2}/2S^+ & -\frac{J_z}{2}S^z & 1
//	\end{array}
//	\right)\f$.
//	The fourth row and column are missing when \f$J_{xy}=0\f$. Uses the appropriate spin operators for a given \p S.*/
//	static SuperMatrix<double> Generator (double Jxy, double Jz, double Bz, double Bx, size_t D=2);
	static HamiltonianTermsXd set_operators (const SpinBase &B, double Jxy, double Jz, double Bz=0., double Bx=0., 
	                                         double Jprime=0., double JxyIntra=0., double JzIntra=0., double K=0.);
	
	static HamiltonianTermsXd set_operators (const SpinBase &B, const MatrixXd &JxyInter, const MatrixXd &JzInter, 
	                                         const VectorXd &Bz, const VectorXd &Bx, 
	                                         double Jprime=0., double JxyIntra=0., double JzIntra=0., double K=0.);
	
//	HeisenbergModel (size_t Lx_input, double J, double Bz0_input, double K, const vector<std::array<double,3> > &DM_input, 
//	                 size_t D_input=2, bool CALC_SQUARE=true);
	
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
		frac S = frac(D-1,2);
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
	
	/**local basis: \f$\{ \left|\uparrow\right>, \left|\downarrow\right> \}\f$ if D=2 and N_legs=1*/
	static const vector<qarray<1> > qloc (size_t N_legs=1, size_t D=2);
	
	/**Labels the conserved quantum number as "M".*/
	static const std::array<string,1> maglabel;
	///@}
	
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
	
	MpoQ<1> Sz (size_t locx, size_t locy=0) const;
	MpoQ<1> SzSz (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	MpoQ<1,complex<double> > SaPacket (SPINOP_LABEL SOP, complex<double> (*f)(int)) const;
	
protected:
	
	double Jxy=-1., Jz=-1., Bz=0.;
	double Jprime=0.;
	double K=0.;
	size_t D=2;
	
	SpinBase B;
};

const std::array<string,1> HeisenbergModel::maglabel{"M"};

const vector<qarray<1> > HeisenbergModel::
qloc (size_t N_legs, size_t D)
{
	vector<qarray<1> > qss;
	int Sx2 = static_cast<int>(D-1);
	for (int M=Sx2; M>=-Sx2; M-=2)
	{
		qss.push_back(qarray<1>{M});
	}
	
	vector<qarray<1> > vout(pow(D,N_legs));
	
	NestedLoopIterator Nelly(N_legs,D);
	for (Nelly=Nelly.begin(); Nelly!=Nelly.end(); ++Nelly)
	{
		vout[*Nelly] = qss[Nelly(0)];
		
		for (int leg=1; leg<N_legs; ++leg)
		{
			vout[*Nelly][0] += qss[Nelly(leg)][0];
		}
	}
	return vout;
};

HamiltonianTermsXd HeisenbergModel::
set_operators (const SpinBase &B, double Jxy, double Jz, double Bz, double Bx, double Jprime, double JxyIntra, double JzIntra, double K)
{
	MatrixXd JxyInter(B.orbitals(),B.orbitals()); JxyInter.setIdentity(); JxyInter *= Jxy;
	MatrixXd JzInter (B.orbitals(),B.orbitals()); JzInter.setIdentity();  JzInter  *= Jz;
	VectorXd Bzvec(B.orbitals()); Bzvec.setConstant(Bz);
	VectorXd Bxvec(B.orbitals()); Bxvec.setConstant(Bx);
	return set_operators(B, JxyInter, JzInter, Bzvec, Bxvec, Jprime, JxyIntra, JzIntra, K);
}

HamiltonianTermsXd HeisenbergModel::
set_operators (const SpinBase &B, const MatrixXd &JxyInter, const MatrixXd &JzInter, const VectorXd &Bz, const VectorXd &Bx, 
               double Jprime, double JxyIntra, double JzIntra, double K)
{
	assert(B.orbitals() == JxyInter.rows() and 
	       B.orbitals() == JxyInter.cols() and 
	       B.orbitals() == JzInter.rows()  and 
	       B.orbitals() == JzInter.cols());
	
	HamiltonianTermsXd Terms;
	
	for (int leg1=0; leg1<B.orbitals(); ++leg1)
	for (int leg2=0; leg2<B.orbitals(); ++leg2)
	{
		if (JxyInter(leg1,leg2) != 0.)
		{
			Terms.tight.push_back(make_tuple(-0.5*JxyInter(leg1,leg2), B.Scomp(SP,leg1), B.Scomp(SM,leg2)));
			Terms.tight.push_back(make_tuple(-0.5*JxyInter(leg1,leg2), B.Scomp(SM,leg1), B.Scomp(SP,leg2)));
		}
		if (JzInter(leg1,leg2) != 0.)
		{
			Terms.tight.push_back(make_tuple(-JzInter(leg1,leg2), B.Scomp(SZ,leg1), B.Scomp(SZ,leg2)));
		}
	}
	
	if (Jprime != 0.)
	{
		SparseMatrixXd Id = MatrixXd::Identity(B.get_D(),B.get_D()).sparseView();
		Terms.nextn.push_back(make_tuple(-0.5*Jprime, B.Scomp(SP), B.Scomp(SM), Id));
		Terms.nextn.push_back(make_tuple(-0.5*Jprime, B.Scomp(SM), B.Scomp(SP), Id));
		Terms.nextn.push_back(make_tuple(-Jprime,     B.Scomp(SZ), B.Scomp(SZ), Id));
	}
	
	Terms.local.push_back(make_tuple(1., B.HeisenbergHamiltonian(JxyIntra,JzIntra,Bz,Bx,K)));
	
	return Terms;
}

HeisenbergModel::
HeisenbergModel (int Lx_input, double Jxy_input, double Jz_input, double Bz_input, size_t D_input, size_t Ly_input, bool CALC_SQUARE)
:MpoQ<1> (Lx_input, Ly_input, HeisenbergModel::qloc(Ly_input,D_input), {0}, HeisenbergModel::maglabel, "", halve),
Jxy(Jxy_input), Jz(Jz_input), Bz(Bz_input), D(D_input)
{
	if (Jz==numeric_limits<double>::infinity()) {Jz=Jxy;} // default: Jxy=Jz
	assert(Jxy != 0. or Jz != 0.);
	this->label = create_label(D,Jxy,Jz,0,Bz,0);
	
	B = SpinBase(N_legs,D);
	HamiltonianTermsXd Terms = set_operators(B, Jxy,Jz,Bz,0.);
	SuperMatrix<double> G = Generator(Terms);
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

HeisenbergModel::
HeisenbergModel (size_t Lx_input, std::array<double,2> Jlist, double Bz_input, size_t Ly_input, size_t D_input, bool CALC_SQUARE)
:MpoQ<1> (Lx_input, Ly_input, HeisenbergModel::qloc(Ly_input,D_input), {0}, HeisenbergModel::maglabel, "", halve),
Jxy(Jlist[0]), Jz(Jlist[0]), Bz(Bz_input), D(D_input), Jprime(Jlist[1])
{
	this->label = create_label(D,Jxy,Jz,Jprime,Bz,0.);
	
	B = SpinBase(N_legs,D);
	HamiltonianTermsXd Terms;
	if (Ly_input == 1)
	{
		Terms = set_operators(B, Jxy,Jz,Bz,0., Jprime);
	}
	else
	{
		Terms = set_operators(B, Jprime,Jprime,Bz,0., 0., Jxy,Jxy);
		Terms.tight.push_back(make_tuple(-0.5*Jxy, B.Scomp(SP,1), B.Scomp(SM,0)));
		Terms.tight.push_back(make_tuple(-0.5*Jxy, B.Scomp(SM,1), B.Scomp(SP,0)));
		Terms.tight.push_back(make_tuple(-Jz, B.Scomp(SZ,1), B.Scomp(SZ,0)));
	}
	SuperMatrix<double> G = Generator(Terms);
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

//HeisenbergModel::
//HeisenbergModel(size_t Lx_input, double J, double Bz0_input, double K, const vector<std::array<double,3> > &DM_input, size_t D_input, bool CALC_SQUARE)
//:MpoQ<1> (Lx_input, 1, HeisenbergModel::qloc(Ly_input,D_input), {0}, HeisenbergModel::maglabel, "", halve),
//Jxy(J), Jz(J), Bz(Bz0_input), K(K_input), DM(DM_input), D(D_input)
//{
//	this->label = create_label(D,Jxy,Jz,0,Bz,0);
//	
//	B = SpinBase(N_legs,D);
//	
//	HamiltonianTermsXcd Terms_first = set_operators(B, J,Bz,K,DM);
//	HamiltonianTermsXcd Terms_rest  = set_operators(B, J,0.,K,DM);
//	
//	vector<SuperMatrix<double> > G(this->N_sites);
//	vector<SuperMatrix<double> > Gsq;
//	if (CALC_SQUARE == true)
//	{
//		Gsq.resize(this->N_sites);
//	}
//	
//	for (size_t l=0; l<this->N_sites; ++l)
//	{
//		if (l==0)
//		{
//			this->Daux = Terms_first.auxdim();
//			G[l].setRowVector(Daux,F.dim());
//			G[l] = Generator(Terms_first).row(Daux-1);
//			
//			if (CALC_SQUARE == true)
//			{
//				Gsq[l].setRowVector(Daux*Daux,F.dim());
//				Gsq[l] = tensor_product(G[l],G[l]);
//			}
//		}
//		else if (l==this->N_sites-1)
//		{
//			this->Daux = Terms_rest.auxdim();
//			G[l].setColVector(Daux,F.dim());
//			G[l] = Generator(Terms_rest).col(0);
//			
//			if (CALC_SQUARE == true)
//			{
//				Gsq[l].setColVector(Daux*Daux,F.dim());
//				Gsq[l] = tensor_product(G[l],G[l]);
//			}
//		}
//		else
//		{
//			this->Daux = Terms_rest.auxdim();
//			G[l].setMatrix(Daux,F.dim());
//			G[l] = Generator(Terms_rest);
//			
//			if (CALC_SQUARE == true)
//			{
//				Gsq[l].setMatrix(Daux*Daux,F.dim());
//				Gsq[l] = tensor_product(G[l],G[l]);
//			}
//		}
//	}
//	
//	this->construct(G, this->W, this->Gvec);
//	
//	if (CALC_SQUARE == true)
//	{
//		this->construct(Gsq, this->Wsq, this->GvecSq);
//		this->GOT_SQUARE = true;
//	}
//	else
//	{
//		this->GOT_SQUARE = false;
//	}
//}

MpoQ<1> HeisenbergModel::
Sz (size_t locx, size_t locy) const
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << "Sz(" << locx << "," << locy << ")";
	MpoQ<1> Mout(N_sites, N_legs, HeisenbergModel::qloc(N_legs,D), {0}, HeisenbergModel::maglabel, ss.str(), halve);
	Mout.setLocal(locx, B.Scomp(SZ,locy));
	return Mout;
}

MpoQ<1> HeisenbergModel::
SzSz (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	assert(locx1<N_sites and locx2<N_sites and locy1<N_legs and locy2<N_legs);
	stringstream ss;
	ss << "Sz(" << locx1 << "," << locy1 << ")" <<  "Sz(" << locx2 << "," << locy2 << ")";
	MpoQ<1> Mout(N_sites, N_legs, HeisenbergModel::qloc(N_legs,D), {0}, HeisenbergModel::maglabel, ss.str(), halve);
	Mout.setLocal({locx1, locx2}, {B.Scomp(SZ,locy1), B.Scomp(SZ,locy2)});
	return Mout;
}

MpoQ<1,complex<double> > HeisenbergModel::
SaPacket (SPINOP_LABEL SOP, complex<double> (*f)(int)) const
{
	assert(SOP==SP or SOP==SM or SOP==SZ);
	stringstream ss;
	ss << SOP << "Packet";
	qarray<1> DeltaM;
	if      (SOP==SP) {DeltaM={+2};}
	else if (SOP==SM) {DeltaM={-2};}
	else              {DeltaM={ 0};}
	MpoQ<1,complex<double> > Mout(N_sites, N_legs, HeisenbergModel::qloc(N_legs,D), DeltaM, HeisenbergModel::maglabel, ss.str(), halve);
	Mout.setLocalSum(B.Scomp(SOP), f);
	return Mout;
}

}

#endif

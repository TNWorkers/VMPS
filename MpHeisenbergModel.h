#ifndef STRAWBERRY_HEISENBERGMODEL
#define STRAWBERRY_HEISENBERGMODEL

#include "symmetry/U1.h"
#include "array"
#include "MpoQ.h"
#include "SpinBase.h"
#include "DmrgExternalQ.h"
#include "ParamHandler.h"

namespace VMPS
{

/**MPO representation of 
\f$
H = -J_{xy} \sum_{<ij>} \left(S^x_iS^x_j+S^y_iS^y_j\right) - J_z \sum_{<ij>} S^z_iS^z_j -J' \sum_{<<ij>>} \left(\mathbf{S_i}\mathbf{S_j}\right) - B_z \sum_i S^z_i
\f$.
\param D : \f$D=2S+1\f$ where \f$S\f$ is the spin
\note \f$J<0\f$ : antiferromagnetic*/
class HeisenbergModel : public MpoQ<Sym::U1<double>,double>
{
typedef Sym::U1<double> Symmetry;
typedef Symmetry::qType qType;
typedef SiteOperator<Symmetry,SparseMatrix<double> > OperatorType;
public:
	
	HeisenbergModel() : MpoQ<Symmetry>() {};
	
	HeisenbergModel (size_t Lx_input, initializer_list<Param> params, size_t D_input=2, size_t Ly_input=1, bool CALC_SQUARE=true);
	
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
	                 size_t Ly_input=1, bool CALC_SQUARE=false);
	
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
	template<typename Symmetry_>
	static HamiltonianTermsXd<Symmetry_> set_operators (const SpinBase<Symmetry_> &B, double Jxy, double Jz, double Bz=0., double Bx=0., 
	                                         double Jprime=0., double JxyIntra=0., double JzIntra=0., double K=0.);
	
	template<typename Symmetry_>
	static HamiltonianTermsXd<Symmetry_> set_operators (const SpinBase<Symmetry_> &B, const MatrixXd &JxyInter, const MatrixXd &JzInter, 
	                                         const VectorXd &Bz, const VectorXd &Bx, 
	                                         double Jprime=0., double JxyIntra=0., double JzIntra=0., double K=0.);
	
	template<typename Symmetry_>
	static HamiltonianTermsXd<Symmetry_> set_operators (const SpinBase<Symmetry_> &B, const ParamHandler &P);
	
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

	/**Operator Quantum numbers: \f$\{ \left|0\right>, \left|+2\right>, \left|-2\right> \}\f$ */
	static const vector<qarray<1> > qOp ();

	/**Labels the conserved quantum number as "M".*/
	static const std::array<string,1> maglabel;
	///@}
	
	///@{
	/**Typedef for convenient reference (no need to specify \p Symmetry, \p Scalar all the time).*/
	typedef MpsQ<Symmetry,double>                           StateXd;
	typedef MpsQ<Symmetry,complex<double> >                 StateXcd;
	typedef DmrgSolverQ<Symmetry,HeisenbergModel,double>    Solver;
	typedef MpsQCompressor<Symmetry,double,double>          CompressorXd;
	typedef MpsQCompressor<Symmetry,complex<double>,double> CompressorXcd;
	typedef MpoQ<Symmetry>                                  Operator;
	///@}
	
	/**Calculates the necessary auxiliary dimension, detecting when \p Jxy or \p Jz are zero.*/
	static size_t calc_Daux (double Jxy, double Jz)
	{
		size_t res = 2;
		res += (Jxy!=0.)? 2 : 0;
		res += (Jz !=0.)? 1 : 0;
		return res;
	}
	
	MpoQ<Symmetry> Sz (size_t locx, size_t locy=0) const;
	MpoQ<Symmetry> SzSz (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	
protected:
	
	double Jxy=0, Jz=0;
	double Jxyprime=0, Jzprime=0;
	MatrixXd JxyInter, JzInter;
	double Bz=0, K=0;
	size_t D=2;
	
	std::map<string,std::any> defaults = 
	{
		{"J",0.}, {"Jxy",0.}, {"Jz",0.}, {"D",2ul}, {"Jprime",0.}, {"Jxyprime",0.}, {"Jzprime",0.}, {"Bz",0.}, {"K",0.}
	};
	
	SpinBase<Symmetry> B;
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

const vector<qarray<1> > HeisenbergModel::
qOp ()
{
	vector<qarray<1> > vout;
	vout.push_back({0});
	vout.push_back({+2});
	vout.push_back({-2});
	return vout;
}

template<typename Symmetry_>
HamiltonianTermsXd<Symmetry_> HeisenbergModel::
set_operators (const SpinBase<Symmetry_> &B, double Jxy, double Jz, double Bz, double Bx, double Jprime, double JxyIntra, double JzIntra, double K)
{
	JxyIntra = Jxy; JzIntra = Jz;
	MatrixXd JxyInter(B.orbitals(),B.orbitals()); JxyInter.setIdentity(); JxyInter *= Jxy;
	MatrixXd JzInter (B.orbitals(),B.orbitals()); JzInter.setIdentity();  JzInter  *= Jz;
	VectorXd Bzvec(B.orbitals()); Bzvec.setConstant(Bz);
	VectorXd Bxvec(B.orbitals()); Bxvec.setConstant(Bx);
	return set_operators(B, JxyInter, JzInter, Bzvec, Bxvec, Jprime, JxyIntra, JzIntra, K);
}

template<typename Symmetry_>
HamiltonianTermsXd<Symmetry_> HeisenbergModel::
set_operators (const SpinBase<Symmetry_> &B, const MatrixXd &JxyInter, const MatrixXd &JzInter, const VectorXd &Bz, const VectorXd &Bx, 
               double Jprime, double JxyIntra, double JzIntra, double K)
{
	assert(B.orbitals() == JxyInter.rows() and 
	       B.orbitals() == JxyInter.cols() and 
	       B.orbitals() == JzInter.rows()  and 
	       B.orbitals() == JzInter.cols());
	
	HamiltonianTermsXd<Symmetry_> Terms;
	
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
		SiteOperator<Symmetry_,SparseMatrix<double> > Id(Matrix<double,Dynamic,Dynamic>::Identity(B.get_D(),B.get_D()).sparseView(),Symmetry_::qvacuum());
		Terms.nextn.push_back(make_tuple(-0.5*Jprime, B.Scomp(SP), B.Scomp(SM), Id));
		Terms.nextn.push_back(make_tuple(-0.5*Jprime, B.Scomp(SM), B.Scomp(SP), Id));
		Terms.nextn.push_back(make_tuple(-Jprime,     B.Scomp(SZ), B.Scomp(SZ), Id));
	}
	
	Terms.local.push_back(make_tuple(1., B.HeisenbergHamiltonian(JxyIntra,JzIntra,Bz,Bx,K)));
	
	return Terms;
}

template<typename Symmetry_>
HamiltonianTermsXd<Symmetry_> HeisenbergModel::
set_operators (const SpinBase<Symmetry_> &B, const ParamHandler &P)
{
	HamiltonianTermsXd<Symmetry_> Terms;
	
	frac S = frac(B.get_D()-1,2);
	stringstream ss;
	IOFormat CommaInitFmt(StreamPrecision, DontAlignCols, ",", ",", "", "", "{", "}");
	
	// J-terms
	
	double J   = P.get_default<double>("J");
	double Jxy = P.get_default<double>("Jxy");
	double Jz  = P.get_default<double>("Jz");
	
	MatrixXd Jpara  (B.orbitals(),B.orbitals()); Jpara.setZero();
	MatrixXd Jxypara(B.orbitals(),B.orbitals()); Jxypara.setZero();
	MatrixXd Jzpara (B.orbitals(),B.orbitals()); Jzpara.setZero();
	
	if (P.HAS("J"))
	{
		J = P.get<double>("J");
		Jxypara.diagonal().setConstant(J);
		Jzpara.diagonal().setConstant(J);
		ss << "Heisenberg(S=" << S << ",J=" << J;
	}
	else if (P.HAS("Jxy") or P.HAS("Jz"))
	{
		if (P.HAS("Jxy"))
		{
			Jxy = P.get<double>("Jxy");
			Jxypara.diagonal().setConstant(Jxy);
		}
		if (P.HAS("Jz"))
		{
			Jz = P.get<double>("Jz");
			Jzpara.diagonal().setConstant(Jz);
		}
		
		if      (Jxy == 0.) {ss << "Ising(S=" << S << ",J=" << Jz;}
		else if (Jz  == 0.) {ss << "XX(S="    << S << ",J=" << Jxy;}
		else                {ss << "XXZ(S="   << S << ",Jxy=" << Jxy << ",Jz=" << Jz;}
	}
	else if (P.HAS("Jpara"))
	{
		assert(B.orbitals() == Jpara.rows() and 
		       B.orbitals() == Jpara.cols());
		Jpara = P.get<MatrixXd>("Jpara");
		ss << "Heisenberg(S=" << S << ",J∥=" << Jpara.format(CommaInitFmt);
	}
	else if (P.HAS("Jxypara") or P.HAS("Jzpara"))
	{
		if (P.HAS("Jxypara"))
		{
			Jxypara = P.get<MatrixXd>("Jxypara");
		}
		if (P.HAS("Jzpara"))
		{
			Jzpara = P.get<MatrixXd>("Jzpara");
		}
		
		if      (Jxypara.norm() == 0.) {ss << "Ising(S=" << S << ",J∥=" << Jzpara.format(CommaInitFmt);}
		else if (Jzpara.norm()  == 0.) {ss << "XX(S="    << S << ",J∥=" << Jxypara.format(CommaInitFmt);}
		else                           {ss << "XXZ(S="   << S << ",Jxy∥=" << Jxypara.format(CommaInitFmt) << ",Jz=" << Jzpara.format(CommaInitFmt);}
	}
	else
	{
		ss << "JustLocal(";
	}
	
	for (int i=0; i<B.orbitals(); ++i)
	for (int j=0; j<B.orbitals(); ++j)
	{
		if (Jxypara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(-0.5*Jxypara(i,j), B.Scomp(SP,i), B.Scomp(SM,j)));
			Terms.tight.push_back(make_tuple(-0.5*Jxypara(i,j), B.Scomp(SM,i), B.Scomp(SP,j)));
		}
		if (Jzpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(-Jzpara(i,j), B.Scomp(SZ,i), B.Scomp(SZ,j)));
		}
	}
	
	// J'-terms
	
	if (P.HAS("Jprime") or P.HAS("Jxyprime") or P.HAS("Jzprime"))
	{
		assert(B.orbitals() == 1 and "Cannot interpret Ly>1 and J'!=0");
		
		double Jprime, Jxyprime, Jzprime;
		
		if (P.HAS("Jprime"))
		{
			Jprime = P.get<double>("Jprime");
			Jxyprime = Jprime;
			Jzprime  = Jprime;
			ss << ",J'" << Jprime;
		}
		else
		{
			if (P.HAS("Jxyprime"))
			{
				Jxyprime = P.get<double>("Jxyprime");
				ss << ",Jxy'" << Jxyprime;
			}
			if (P.HAS("Jxzprime"))
			{
				Jzprime = P.get<double>("Jzprime");
				ss << ",Jz'=" << Jzprime;
			}
		}
		
		SiteOperator<Symmetry_,SparseMatrix<double> > Id(Matrix<double,Dynamic,Dynamic>::Identity(B.get_D(),B.get_D()).sparseView(),Symmetry_::qvacuum());
		Terms.nextn.push_back(make_tuple(-0.5*Jxyprime, B.Scomp(SP), B.Scomp(SM), Id));
		Terms.nextn.push_back(make_tuple(-0.5*Jxyprime, B.Scomp(SM), B.Scomp(SP), Id));
		Terms.nextn.push_back(make_tuple(-Jzprime,     B.Scomp(SZ), B.Scomp(SZ), Id));
	}
	
	// local terms
	
	double Jperp   = P.get_default<double>("Jperp");
	double Jxyperp = P.get_default<double>("Jxyperp");
	double Jzperp  = P.get_default<double>("Jzperp");
	
	if (P.HAS("J"))
	{
		Jxyperp = P.get<double>("J");
		Jzperp  = P.get<double>("J");
	}
	else if (P.HAS("Jperp"))
	{
		Jperp = P.get<double>("Jperp");
		ss << ",J⟂=" << Jperp;
	}
	else
	{
		if (P.HAS("Jxyperp"))
		{
			Jxyperp = P.get<double>("Jxyperp");
			ss << ",Jxy⟂=" << Jxyperp;
		}
		if (P.HAS("Jzperp"))
		{
			Jzperp = P.get<double>("Jzperp");
			ss << ",Jz⟂=" << Jzperp;
		}
		
	}
	
	double Bz = P.get_default<double>("Bz");
	double Bx = P.get_default<double>("Bx");
	double K  = P.get_default<double>("K");
	
	if (P.HAS("Bz"))
	{
		Bz = P.get<double>("Bz");
		ss << ",Bz=" << Bz;
	}
	if (P.HAS("Bx"))
	{
		Bx = P.get<double>("Bx");
		ss << ",Bx=" << Bx;
	}
	if (P.HAS("K"))
	{
		K = P.get<double>("K");
		ss << ",K=" << K;
	}
	
	ss << ")";
	Terms.info = ss.str();
	
	Terms.local.push_back(make_tuple(1., B.HeisenbergHamiltonian(Jxyperp,Jzperp,Bz,Bx,K)));
	
	return Terms;
}

HeisenbergModel::
HeisenbergModel (size_t Lx_input, initializer_list<Param> params, size_t D_input, size_t Ly_input, bool CALC_SQUARE)
:MpoQ<Symmetry> (Lx_input, Ly_input, qarray<Symmetry::Nq>({0}), HeisenbergModel::qOp(), HeisenbergModel::maglabel, "", halve)
{
	ParamHandler P(params,defaults);
	B = SpinBase<Symmetry>(N_legs, P.get<size_t>("D"));
	
	for (size_t l=0; l<N_sites; ++l)
	{
		setLocBasis(B.basis(),l);
	}
	
	HamiltonianTermsXd<Symmetry> Terms = set_operators(B,P);
	this->label = Terms.info;
	SuperMatrix<Symmetry,double> G = Generator(Terms);
	this->Daux = Terms.auxdim();
	
	this->construct(G, this->W, this->Gvec);
	
	if (CALC_SQUARE == true)
	{
		vector<qType> qOpSq_;
		qOpSq_.push_back({0}); qOpSq_.push_back({2}); qOpSq_.push_back({-2}); qOpSq_.push_back({4}); qOpSq_.push_back({-4});
		vector<vector<qType> > qOpSq(this->N_sites,qOpSq_);
		this->setOpBasisSq(qOpSq);
		this->construct(tensor_product(G,G), this->Wsq, this->GvecSq, qOpSq);
		this->GOT_SQUARE = true;
	}
	else
	{
		this->GOT_SQUARE = false;
	}
	this->calc_auxBasis();
}

HeisenbergModel::
HeisenbergModel (int Lx_input, double Jxy_input, double Jz_input, double Bz_input, size_t D_input, size_t Ly_input, bool CALC_SQUARE)
:MpoQ<Symmetry> (Lx_input, Ly_input, HeisenbergModel::qloc(Ly_input,D_input), HeisenbergModel::qOp(), {0}, HeisenbergModel::maglabel, "", halve),
Jxy(Jxy_input), Jz(Jz_input), Bz(Bz_input), D(D_input)
{
	if (Jz==numeric_limits<double>::infinity()) {Jz=Jxy;} // default: Jxy=Jz
	assert(Jxy != 0. or Jz != 0.);
	this->label = create_label(D,Jxy,Jz,0,Bz,0);
	B = SpinBase<Symmetry>(N_legs,D);
	HamiltonianTermsXd<Symmetry> Terms = set_operators(B, Jxy,Jz,Bz,0.);
	SuperMatrix<Symmetry,double> G = Generator(Terms);
	this->Daux = Terms.auxdim();
	
	this->construct(G, this->W, this->Gvec);
	
	if (CALC_SQUARE == true)
	{
		vector<qType> qOpSq_;
		qOpSq_.push_back({0}); qOpSq_.push_back({2}); qOpSq_.push_back({-2}); qOpSq_.push_back({4}); qOpSq_.push_back({-4});
		vector<vector<qType> > qOpSq(this->N_sites,qOpSq_);
		this->setOpBasisSq(qOpSq);
		this->construct(tensor_product(G,G), this->Wsq, this->GvecSq, qOpSq);
		this->GOT_SQUARE = true;
	}
	else
	{
		this->GOT_SQUARE = false;
	}
	this->calc_auxBasis();
}

HeisenbergModel::
HeisenbergModel (size_t Lx_input, std::array<double,2> Jlist, double Bz_input, size_t Ly_input, size_t D_input, bool CALC_SQUARE)
:MpoQ<Symmetry> (Lx_input, Ly_input, HeisenbergModel::qloc(Ly_input,D_input), HeisenbergModel::qOp(), {0}, HeisenbergModel::maglabel, "", halve),
Jxy(Jlist[0]), Jz(Jlist[0]), Bz(Bz_input), D(D_input), Jxyprime(Jlist[1]), Jzprime(Jlist[1])
{
	this->label = create_label(D,Jxy,Jz,Jxyprime,Bz,0.);
	
	B = SpinBase<Symmetry>(N_legs,D);
	HamiltonianTermsXd<Symmetry> Terms;
	if (Ly_input == 1)
	{
		Terms = set_operators(B, Jxy,Jz,Bz,0., Jxyprime);
	}
	else
	{
		Terms = set_operators(B, Jxyprime,Jxyprime,Bz,0., 0., Jxy,Jxy);
		Terms.tight.push_back(make_tuple(-0.5*Jxy, B.Scomp(SP,1), B.Scomp(SM,0)));
		Terms.tight.push_back(make_tuple(-0.5*Jxy, B.Scomp(SM,1), B.Scomp(SP,0)));
		Terms.tight.push_back(make_tuple(-Jz, B.Scomp(SZ,1), B.Scomp(SZ,0)));
	}
	SuperMatrix<Symmetry,double> G = Generator(Terms);
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

MpoQ<Sym::U1<double> > HeisenbergModel::
Sz (size_t locx, size_t locy) const
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << "Sz(" << locx << "," << locy << ")";
//	MpoQ<Symmetry> Mout(N_sites, N_legs, HeisenbergModel::qloc(N_legs,D), {{0}}, {0}, HeisenbergModel::maglabel, ss.str(), halve);
	MpoQ<Symmetry> Mout(N_sites, N_legs, qarray<Symmetry::Nq>({0}), {{0}}, HeisenbergModel::maglabel, ss.str(), halve);
	for (size_t l=0; l<N_sites; ++l)
	{
		Mout.setLocBasis(B.basis(),l);
	}
	Mout.setLocal(locx, B.Scomp(SZ,locy));
	return Mout;
}

MpoQ<Sym::U1<double> > HeisenbergModel::
SzSz (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	assert(locx1<N_sites and locx2<N_sites and locy1<N_legs and locy2<N_legs);
	stringstream ss;
	ss << "Sz(" << locx1 << "," << locy1 << ")" <<  "Sz(" << locx2 << "," << locy2 << ")";
//	MpoQ<Symmetry> Mout(N_sites, N_legs, HeisenbergModel::qloc(N_legs,D), {{0}}, {0}, HeisenbergModel::maglabel, ss.str(), halve);
	MpoQ<Symmetry> Mout(N_sites, N_legs, qarray<Symmetry::Nq>({0}), {{0}}, HeisenbergModel::maglabel, ss.str(), halve);
	for (size_t l=0; l<N_sites; ++l)
	{
		Mout.setLocBasis(B.basis(),l);
	}
	Mout.setLocal({locx1, locx2}, {B.Scomp(SZ,locy1), B.Scomp(SZ,locy2)});
	return Mout;
}

}

#endif

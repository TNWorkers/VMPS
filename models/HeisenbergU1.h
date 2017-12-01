#ifndef STRAWBERRY_HEISENBERGU1
#define STRAWBERRY_HEISENBERGU1

#include <array>

#include "MpoQ.h"
#include "symmetry/U1.h"
#include "SpinBase.h"
#include "DmrgExternalQ.h"
#include "ParamHandler.h"

namespace VMPS
{

/** \class HeisenbergU1
  * \ingroup Models
  *
  * \brief Heisenberg Model
  *
  * MPO representation of
  \f[
  H = -J_{xy} \sum_{<ij>} \left(S^x_iS^x_j+S^y_iS^y_j\right) - J_z \sum_{<ij>} S^z_iS^z_j -J' \sum_{<<ij>>} \left(\mathbf{S_i}\mathbf{S_j}\right)
      - B_z \sum_i S^z_i
  \f]
  *
  \param D : \f$D=2S+1\f$ where \f$S\f$ is the spin
  \note Take use of the \f$S^z\f$ U(1) symmetry.
  \note \f$J<0\f$ is antiferromagnetic
*/

class HeisenbergU1 : public MpoQ<Sym::U1<double>,double>
{
public:
	typedef Sym::U1<double> Symmetry;
private:
	typedef Symmetry::qType qType;
	typedef SiteOperator<Symmetry,SparseMatrix<double> > OperatorType;
public:

	//---constructors---
	///\{
	/**Do nothing.*/
	HeisenbergU1() : MpoQ<Symmetry>() {};

	/**
	   \param Lx_input : chain length
	   \describe_params
	   \param Ly_input : amount of legs in ladder
	   \param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	*/
	HeisenbergU1 (size_t Lx_input, initializer_list<Param> params, size_t Ly_input=1, bool CALC_SQUARE=true);
	///\}

	/**
	   \param B : Base class from which the local operators are received
	   \param P : The parameters
	*/
	template<typename Symmetry_>
	static HamiltonianTermsXd<Symmetry_> set_operators (const SpinBase<Symmetry_> &B, const ParamHandler &P);

	/**Operator Quantum numbers: \f$\{ \left|0\right>, \left|+2\right>, \left|-2\right> \}\f$ */
	static const vector<qarray<1> > qOp ();

	/**Labels the conserved quantum number as "M".*/
	static const std::array<string,1> maglabel;

	///@{
	/**Typedef for convenient reference (no need to specify \p Symmetry, \p Scalar all the time).*/
	typedef MpsQ<Symmetry,double>                           StateXd;
	typedef MpsQ<Symmetry,complex<double> >                 StateXcd;
	typedef DmrgSolverQ<Symmetry,HeisenbergU1,double>       Solver;
	typedef MpsQCompressor<Symmetry,double,double>          CompressorXd;
	typedef MpsQCompressor<Symmetry,complex<double>,double> CompressorXcd;
	typedef MpoQ<Symmetry>                                  Operator;
	///@}

	///@{
	/**Observables.*/	
	MpoQ<Symmetry> Sz (size_t locx, size_t locy=0) const;
	MpoQ<Symmetry> SzSz (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	///@}
	
protected:
		
	const std::map<string,std::any> defaults = 
	{
		{"J",0.}, {"Jxy",0.}, {"Jz",0.},
		{"Jprime",0.}, {"Jxyprime",0.}, {"Jzprime",0.},
		{"Jperp",0.}, {"Jxyperp",0.}, {"Jzperp",0.},
		{"Jpara",0.}, {"Jxypara",0.}, {"Jzpara",0.},
		{"D",2ul}, {"Bz",0.}, {"Bx",0.}, {"K",0.}
	};
	
	SpinBase<Symmetry> B;
};

const std::array<string,1> HeisenbergU1::maglabel{"M"};

const vector<qarray<1> > HeisenbergU1::
qOp ()
{
	vector<qarray<1> > vout;
	vout.push_back({0});
	vout.push_back({+2});
	vout.push_back({-2});
	return vout;
}

HeisenbergU1::
HeisenbergU1 (size_t Lx_input, initializer_list<Param> params, size_t Ly_input, bool CALC_SQUARE)
:MpoQ<Symmetry> (Lx_input, Ly_input, qarray<Symmetry::Nq>({0}), HeisenbergU1::qOp(), HeisenbergU1::maglabel, "", halve)
{
	ParamHandler P(params,defaults);
	B = SpinBase<Symmetry>(N_legs, P.get<size_t>("D"));
	
	for (size_t l=0; l<N_sites; ++l) { setLocBasis(B.basis(),l); }
	
	HamiltonianTermsXd<Symmetry> Terms = set_operators(B,P);
	this->label = Terms.info;
	SuperMatrix<Symmetry,double> G = Generator(Terms);
	this->Daux = Terms.auxdim();
	
	this->construct(G, this->W, this->Gvec, CALC_SQUARE);	
}

MpoQ<Sym::U1<double> > HeisenbergU1::
Sz (size_t locx, size_t locy) const
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << "Sz(" << locx << "," << locy << ")";
	MpoQ<Symmetry> Mout(N_sites, N_legs, qarray<Symmetry::Nq>({0}), {{0}}, HeisenbergU1::maglabel, ss.str(), halve);
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(B.basis(),l); }
	Mout.setLocal(locx, B.Scomp(SZ,locy));
	return Mout;
}

MpoQ<Sym::U1<double> > HeisenbergU1::
SzSz (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	assert(locx1<N_sites and locx2<N_sites and locy1<N_legs and locy2<N_legs);
	stringstream ss;
	ss << "Sz(" << locx1 << "," << locy1 << ")" <<  "Sz(" << locx2 << "," << locy2 << ")";
	MpoQ<Symmetry> Mout(N_sites, N_legs, qarray<Symmetry::Nq>({0}), {{0}}, HeisenbergU1::maglabel, ss.str(), halve);
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(B.basis(),l); }
	Mout.setLocal({locx1, locx2}, {B.Scomp(SZ,locy1), B.Scomp(SZ,locy2)});
	return Mout;
}

template<typename Symmetry_>
HamiltonianTermsXd<Symmetry_> HeisenbergU1::
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

} //end namespace VMPS

#endif

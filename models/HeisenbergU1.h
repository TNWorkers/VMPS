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
  \note The default variable settings can be seen in \p HeisenbergU1::defaults.
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
	
	HeisenbergU1() : MpoQ<Symmetry>() {};
	HeisenbergU1 (size_t Lx_input, vector<Param> params, size_t Ly_input=1);
	
	/**
	   \param B : Base class from which the local operators are received
	   \param P : The parameters
	*/
	template<typename Symmetry_>
	static HamiltonianTermsXd<Symmetry_> set_operators (const SpinBase<Symmetry_> &B, const ParamHandler &P, size_t loc=0);
	
	/**Operator Quantum numbers: \f$\{ Id,S_z:k=\left|0\right>; S_+:k=\left|+2\right>; S_-:k=\left|-2\right>\}\f$ */
	static const vector<qarray<1> > qOp();
	
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
	/**Observables*/
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
		{"D",2ul}, {"Bz",0.}, {"Bx",0.}, {"K",0.},
		{"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}
	};
	
	vector<SpinBase<Symmetry> > B;
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
HeisenbergU1 (size_t Lx_input, vector<Param> params, size_t Ly_input)
:MpoQ<Symmetry> (Lx_input, Ly_input, qarray<Symmetry::Nq>({0}), HeisenbergU1::qOp(), HeisenbergU1::maglabel, "", halve)
{
	ParamHandler P(params,defaults);
	
	size_t Lcell = P.size();
	vector<SuperMatrix<Symmetry,double> > G;
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	B.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		B[l] = SpinBase<Symmetry>(N_legs, P.get<size_t>("D",l%Lcell));
		setLocBasis(B[l].get_basis(),l);
		
		Terms[l] = set_operators(B[l],P,l%Lcell);
		this->Daux = Terms[l].auxdim();
		
		G.push_back(Generator(Terms[l]));
	}

	this->generate_label(Terms[0].name,Terms,Lcell);
	this->construct(G, this->W, this->Gvec, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
}

MpoQ<Sym::U1<double> > HeisenbergU1::
Sz (size_t locx, size_t locy) const
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << "Sz(" << locx << "," << locy << ")";
	
	MpoQ<Symmetry> Mout(N_sites, N_legs, qarray<Symmetry::Nq>({0}), {{0}}, HeisenbergU1::maglabel, ss.str(), halve);
	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(B[l].get_basis(),l);}
	
	Mout.setLocal(locx, B[locx].Scomp(SZ,locy));
	return Mout;
}

MpoQ<Sym::U1<double> > HeisenbergU1::
SzSz (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	assert(locx1<N_sites and locx2<N_sites and locy1<N_legs and locy2<N_legs);
	stringstream ss;
	ss << "Sz(" << locx1 << "," << locy1 << ")" <<  "Sz(" << locx2 << "," << locy2 << ")";
	
	MpoQ<Symmetry> Mout(N_sites, N_legs, qarray<Symmetry::Nq>({0}), {{0}}, HeisenbergU1::maglabel, ss.str(), halve);
	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(B[l].get_basis(),l);}
	
	Mout.setLocal({locx1, locx2}, {B[locx1].Scomp(SZ,locy1), B[locx2].Scomp(SZ,locy2)});
	return Mout;
}
	
template<typename Symmetry_>
HamiltonianTermsXd<Symmetry_> HeisenbergU1::
set_operators (const SpinBase<Symmetry_> &B, const ParamHandler &P, size_t loc)
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
	
	if (P.HAS("J",loc))
	{
		J = P.get<double>("J",loc);
		Jxypara.diagonal().setConstant(J);
		Jzpara.diagonal().setConstant(J);
		Terms.name = "Heisenberg";
		ss << "S=" << print_frac_nice(S) << ",J=" << J;
	}
	else if (P.HAS("Jxy",loc) or P.HAS("Jz",loc))
	{
		if (P.HAS("Jxy",loc))
		{
			Jxy = P.get<double>("Jxy",loc);
			Jxypara.diagonal().setConstant(Jxy);
		}
		if (P.HAS("Jz",loc))
		{
			Jz = P.get<double>("Jz",loc);
			Jzpara.diagonal().setConstant(Jz);
		}
		
		if      (Jxy == 0.) {Terms.name = "Ising"; ss << "S=" << print_frac_nice(S) << ",J=" << Jz;}
		else if (Jz  == 0.) {Terms.name = "XX"; ss << "S="    << print_frac_nice(S) << ",J=" << Jxy;}
		else                {Terms.name = "XXZ"; ss << "S="   << print_frac_nice(S) << ",Jxy=" << Jxy << ",Jz=" << Jz;}
	}
	else if (P.HAS("Jpara",loc))
	{
		assert(B.orbitals() == Jpara.rows() and 
		       B.orbitals() == Jpara.cols());
		Jpara = P.get<MatrixXd>("Jpara",loc);
		Jxypara = Jpara;
		Jzpara = Jpara;
		Terms.name = "Heisenberg";
		ss << "S=" << print_frac_nice(S) << ",J∥=" << Jpara.format(CommaInitFmt);
	}
	else if (P.HAS("Jxypara",loc) or P.HAS("Jzpara",loc))
	{
		if (P.HAS("Jxypara",loc))
		{
			Jxypara = P.get<MatrixXd>("Jxypara",loc);
		}
		if (P.HAS("Jzpara",loc))
		{
			Jzpara = P.get<MatrixXd>("Jzpara",loc);
		}
		
		if      (Jxypara.norm() == 0.) {Terms.name = "Ising"; ss << "S=" << print_frac_nice(S) << ",J∥=" << Jzpara.format(CommaInitFmt);}
		else if (Jzpara.norm()  == 0.) {Terms.name = "XX"; ss << "S="    << print_frac_nice(S) << ",J∥=" << Jxypara.format(CommaInitFmt);}
		else                           {Terms.name = "XXZ"; ss << "S="   << print_frac_nice(S) <<
																",Jxy∥=" << Jxypara.format(CommaInitFmt) << ",Jz=" << Jzpara.format(CommaInitFmt);}
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
	double Jprime   = P.get_default<double>("Jprime");
	double Jxyprime = P.get_default<double>("Jxyprime");
	double Jzprime  = P.get_default<double>("Jzprime");
	
	if (P.HAS("Jprime",loc) or P.HAS("Jxyprime",loc) or P.HAS("Jzprime",loc))
	{
		assert((B.orbitals() == 1 or (Jprime == 0 and Jxyprime == 0 and Jzprime == 0)) and "Cannot interpret Ly>1 and J'!=0");
		// assert(B.orbitals() == 1 and "Cannot interpret Ly>1 and J'!=0");
		
		if (P.HAS("Jprime",loc))
		{
			Jprime = P.get<double>("Jprime",loc);
			Jxyprime = Jprime;
			Jzprime  = Jprime;
			ss << ",J'=" << Jprime;
		}
		else
		{
			if (P.HAS("Jxyprime",loc))
			{
				Jxyprime = P.get<double>("Jxyprime",loc);
				ss << ",Jxy'=" << Jxyprime;
			}
			if (P.HAS("Jzprime",loc))
			{
				Jzprime = P.get<double>("Jzprime",loc);
				ss << ",Jz'=" << Jzprime;
			}
		}
		if(Jxyprime != 0)
		{
			Terms.nextn.push_back(make_tuple(-0.5*Jxyprime, B.Scomp(SP), B.Scomp(SM), B.Id()));
			Terms.nextn.push_back(make_tuple(-0.5*Jxyprime, B.Scomp(SM), B.Scomp(SP), B.Id()));
		}
		if(Jzprime != 0)
		{
			Terms.nextn.push_back(make_tuple(-Jzprime,     B.Scomp(SZ), B.Scomp(SZ), B.Id()));
		}
	}
	
	// local terms
	
	double Jperp   = P.get_default<double>("Jperp");
	double Jxyperp = P.get_default<double>("Jxyperp");
	double Jzperp  = P.get_default<double>("Jzperp");
	
	if (P.HAS("J",loc))
	{
		Jxyperp = P.get<double>("J",loc);
		Jzperp  = P.get<double>("J",loc);
	}
	else if (P.HAS("Jperp",loc))
	{
		Jperp = P.get<double>("Jperp",loc);
		Jxyperp = Jperp;
		Jzperp  = Jperp;
		ss << ",J⟂=" << Jperp;
	}
	else
	{
		if (P.HAS("Jxyperp",loc))
		{
			Jxyperp = P.get<double>("Jxyperp",loc);
			ss << ",Jxy⟂=" << Jxyperp;
		}
		if (P.HAS("Jzperp",loc))
		{
			Jzperp = P.get<double>("Jzperp",loc);
			ss << ",Jz⟂=" << Jzperp;
		}
		
	}
	
	double Bz = P.get_default<double>("Bz");
	double Bx = P.get_default<double>("Bx");
	double K  = P.get_default<double>("K");
	
	if (P.HAS("Bz",loc))
	{
		Bz = P.get<double>("Bz",loc);
		ss << ",Bz=" << Bz;
	}
	if (P.HAS("Bx",loc))
	{
		Bx = P.get<double>("Bx",loc);
		ss << ",Bx=" << Bx;
	}
	if (P.HAS("K",loc))
	{
		K = P.get<double>("K",loc);
		ss << ",K=" << K;
	}
	
	Terms.info = ss.str();
	Terms.local.push_back(make_tuple(1., B.HeisenbergHamiltonian(Jxyperp,Jzperp,Bz,Bx,K, P.get<bool>("CYLINDER"))));
	
	return Terms;
}

} //end namespace VMPS

#endif

#ifndef STRAWBERRY_HEISENBERGU1XXZ
#define STRAWBERRY_HEISENBERGU1XXZ

#include <array>

#include "MpoQ.h"
#include "symmetry/U1.h"
#include "SpinBase.h"
#include "DmrgExternalQ.h"
#include "ParamHandler.h"

namespace VMPS
{

class HeisenbergU1XXZ : public MpoQ<Sym::U1<double>,double>
{
public:
	typedef Sym::U1<double> Symmetry;
private:
	typedef Symmetry::qType qType;
	typedef SiteOperator<Symmetry,SparseMatrix<double> > OperatorType;
public:
	
	HeisenbergU1XXZ() : MpoQ<Symmetry>() {};
	HeisenbergU1XXZ (variant<size_t,std::array<size_t,2> > L, vector<Param> params);
	
	/**
	   \param B : Base class from which the local operators are received
	   \param P : The parameters
	*/
	template<typename Symmetry_>
	static HamiltonianTermsXd<Symmetry_> set_operators (const SpinBase<Symmetry_> &B, const ParamHandler &P, size_t loc=0);
	
	///@{
	/**Typedef for convenient reference (no need to specify \p Symmetry, \p Scalar all the time).*/
	typedef MpsQ<Symmetry,double>                           StateXd;
	typedef MpsQ<Symmetry,complex<double> >                 StateXcd;
	typedef DmrgSolverQ<Symmetry,HeisenbergU1XXZ,double>    Solver;
	typedef MpsQCompressor<Symmetry,double,double>          CompressorXd;
	typedef MpsQCompressor<Symmetry,complex<double>,double> CompressorXcd;
	typedef MpoQ<Symmetry>                                  Operator;
	///@}
	
	///@{
	/**Observables*/
	MpoQ<Symmetry> Sz (size_t locx, size_t locy=0) const;
	MpoQ<Symmetry> SzSz (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	///@}
	
	/**Validates whether a given total quantum number \p qnum is a possible target quantum number for an MpsQ.
	\returns \p true if valid, \p false if not*/
	bool validate (qarray<1> qnum) const;
	
protected:
	
	vector<SpinBase<Symmetry> > B;
};

HeisenbergU1XXZ::
HeisenbergU1XXZ (variant<size_t,std::array<size_t,2> > L, vector<Param> params)
:MpoQ<Symmetry> (holds_alternative<size_t>(L)? get<0>(L):get<1>(L)[0], 
                 holds_alternative<size_t>(L)? 1        :get<1>(L)[1], 
                 qarray<Symmetry::Nq>({0}), HeisenbergU1XXZ::qOp(), HeisenbergU1XXZ::maglabel, "", halve)
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

template<typename Symmetry_>
HamiltonianTermsXd<Symmetry_> HeisenbergU1::
set_operators (const SpinBase<Symmetry_> &B, const ParamHandler &P, size_t loc)
{
	HamiltonianTermsXd<Symmetry_> Terms;
	
	// J-terms
	
	auto [Jxy,Jxypara,Jxylabel] = P.fill_array2d<double>("Jxy","Jxypara",B.orbitals(),loc);
	Terms.info.push_back(Jlabel);
	
	auto [Jz,Jzpara,Jzlabel] = P.fill_array2d<double>("Jz","Jzpara",B.orbitals(),loc);
	Terms.info.push_back(Jlabel);
	
	for (int i=0; i<B.orbitals(); ++i)
	for (int j=0; j<B.orbitals(); ++j)
	{
		if (Jxypara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(-0.5*Jxypara(i,j), B.Scomp(SP,i), B.Scomp(SM,j)));
			Terms.tight.push_back(make_tuple(-0.5*Jxypara(i,j), B.Scomp(SM,i), B.Scomp(SP,j)));
			Terms.tight.push_back(make_tuple(-Jxypara(i,j),     B.Scomp(SZ,i), B.Scomp(SZ,j)));
		}
		
		if (Jzpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(-0.5*Jzpara(i,j), B.Scomp(SP,i), B.Scomp(SM,j)));
			Terms.tight.push_back(make_tuple(-0.5*Jzpara(i,j), B.Scomp(SM,i), B.Scomp(SP,j)));
			Terms.tight.push_back(make_tuple(-Jzpara(i,j),     B.Scomp(SZ,i), B.Scomp(SZ,j)));
		}
	}
	
	// local terms
	
	double Jxyperp = P.get_default<double>("Jxyperp");
	
	if (P.HAS("Jxy",loc))
	{
		Jxyperp = P.get<double>("Jxy",loc);
	}
	else if (P.HAS("Jxyperp",loc))
	{
		Jperp = P.get<double>("Jxyperp",loc);
		stringstream ss; ss << "Jxy⟂=" << Jperp; Terms.info.push_back(ss.str());
	}
	
	double Jzperp = P.get_default<double>("Jzperp");
	
	if (P.HAS("Jz",loc))
	{
		Jxyperp = P.get<double>("Jz",loc);
	}
	else if (P.HAS("Jzperp",loc))
	{
		Jperp = P.get<double>("Jzperp",loc);
		stringstream ss; ss << "Jz⟂=" << Jperp; Terms.info.push_back(ss.str());
	}
	
	auto [Bz,Bzorb,Bzlabel] = P.fill_array1d<double>("Bz","Bzorb",B.orbitals(),loc);
	Terms.info.push_back(Bzlabel);
	
	auto [Bx,Bxorb,Bxlabel] = P.fill_array1d<double>("Bx","Bxorb",B.orbitals(),loc);
	Terms.info.push_back(Bxlabel);
	
	auto [K,Korb,Klabel] = P.fill_array1d<double>("K","Korb",B.orbitals(),loc);
	Terms.info.push_back(Klabel);
	
	Terms.local.push_back(make_tuple(1., B.HeisenbergHamiltonian(Jxyperp,Jzperp,Bzorb,Bxorb,Korb,0., P.get<bool>("CYLINDER"))));
	
	if (P.HAS("Jxy",loc) or P.HAS("Jxypara") or P.HAS("Jxyperp"))
	{
		Terms.name = "XXZ";
	}
	else
	{
		Terms.name = "Ising";
	}
	
	return Terms;
}

} //end namespace VMPS

#endif

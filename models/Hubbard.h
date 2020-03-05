#ifndef VANILLA_GRANDHUBBARDMODEL
#define VANILLA_GRANDHUBBARDMODEL

#include "symmetry/U0.h"
#include "HubbardU1xU1.h"
#include "BetheAnsatzIntegrals.h" // from TOOLS, depends on gsl

namespace VMPS
{
/**
 * \class Hubbard
 * \ingroup Hubbard
 * \brief Hubbard model without any symmetries.
 * MPO representation of the Hubbard model corresponding to HubbardU1xU1, but without symmetries and an additional possibility of adding
 * \f[
 * 	-B_x \sum_{i} \sigma^x_i
 * \f]
 * with
 * \f[
 * 	\sigma^x_i = \frac{1}{2} \left(\sigma^+_i+\sigma^-_i\right)
 * \f]
 * but is mainly needed for VUMPS.
 * \note The default variable settings can be seen in \p Hubbard::defaults.
 */
class Hubbard : public Mpo<Sym::U0,double>, public HubbardObservables<Sym::U0>, public ParamReturner
{
public:
	typedef Sym::U0 Symmetry;
	MAKE_TYPEDEFS(Hubbard)
	
	Hubbard() : Mpo() {};
	Hubbard (const size_t &L, const vector<Param> &params);
	
	static qarray<0> singlet (int N) {return qarray<0>{};};
	
	template<typename Symmetry_>
	//static void add_operators (HamiltonianTermsXd<Symmetry_> &Terms, const vector<FermionBase<Symmetry_> > &F, const ParamHandler &P, size_t loc=0);
	static void add_operators (const std::vector<FermionBase<Symmetry_>> &F, const ParamHandler &P, HamiltonianTermsXd<Symmetry_> &Terms);
	
	static const std::map<string,std::any> defaults;
	
	static refEnergy ref (const vector<Param> &params, double L=numeric_limits<double>::infinity());
};

const std::map<string,std::any> Hubbard::defaults = 
{
	{"t",1.}, {"tPrime",0.}, {"tRung",1.},
	{"mu",0.}, {"t0",0.}, 
	{"U",0.}, {"Uph",0.},
	{"V",0.}, {"Vrung",0.}, 
	{"Vxy",0.}, {"Vz",0.},
	{"Bz",0.}, {"Bx",0.}, 
	{"J",0.}, {"Jrung",0.},
	{"J3site",0.},
	{"Delta",0.},
	{"X",0.}, {"Xperp",0.},
	{"F",0.}, {"Fperp",0.},
	{"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}
};

Hubbard::
Hubbard (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, PROP::HAMILTONIAN),
 HubbardObservables(L,params,Hubbard::defaults),
 ParamReturner()
{
	ParamHandler P(params,Hubbard::defaults);
	
	size_t Lcell = P.size();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(F[l].get_basis(),l);
	}
	
	HamiltonianTermsXd<Symmetry> Terms(N_sites, P.get<bool>("OPEN_BC"));
	HubbardU1xU1::set_operators(F,P,Terms);
	add_operators(F,P,Terms);
	
	this->construct_from_Terms(Terms, Lcell, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
	this->precalc_TwoSiteData();
}

template<typename Symmetry_>
void Hubbard::
add_operators (const std::vector<FermionBase<Symmetry_>> &F, const ParamHandler &P, HamiltonianTermsXd<Symmetry_> &Terms)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = Terms.size();
	
	for(std::size_t loc=0; loc<N_sites; ++loc)
	{
		std::size_t orbitals = F[loc].orbitals();
		
		param1d Bx = P.fill_array1d<double>("Bx", "Bxorb", orbitals, loc%Lcell);
		Terms.save_label(loc, Bx.label);
		
		// Can also implement superconductivity terms c*c+cdag*cdag here
		
		ArrayXd  U_array  = F[loc].ZeroField();
		ArrayXd  Uph_array  = F[loc].ZeroField();
		ArrayXd  E_array  = F[loc].ZeroField();
		ArrayXd  Bz_array = F[loc].ZeroField();
		ArrayXXd tperp_array = F[loc].ZeroHopping();
		ArrayXXd Vperp_array = F[loc].ZeroHopping();
		ArrayXXd Jperp_array = F[loc].ZeroHopping();
		
		Terms.push_local(loc, 1., F[loc].template HubbardHamiltonian<double>(U_array, Uph_array, E_array, Bz_array, Bx.a, tperp_array, Vperp_array, Jperp_array));
		
		// Fermion parity breaking term c+cdag
		param1d Fval = P.fill_array1d<double>("F", "Forb", orbitals, loc%Lcell);
		Terms.save_label(loc, Fval.label);
		Terms.push_local(loc, 1., F[loc].template FermionParityBreaking<double>(Fval.a));
	}
}


refEnergy Hubbard::
ref (const vector<Param> &params, double L)
{
	ParamHandler P(params,{{"t",1.},{"U",0.},{"n",1.},{"Ly",1ul},{"tRung",1.},{"tPrime",0.},
	                       {"t0",0.},{"V",0.},{"Bz",0.},{"Bx",0.},{"J",0.},{"J3site",0.}});
	refEnergy out;
	
	size_t Ly = P.get<size_t>("Ly");
	double n = P.get<double>("n");
	double U = P.get<double>("U");
	double t = P.get<double>("t");
	double tRung = P.get<double>("tRung");
	
	// half-filled chain
	if (isinf(L) and Ly == 1ul and n == 1. and P.ARE_ALL_ZERO<double>({"tPrime","t0","V","Bz","Bx","J","J3site"}))
	{
		out.value = BetheAnsatz::e0(U,t);
		out.source = "Elliott H. Lieb, F. Y. Wu, Absence of Mott Transition in an Exact Solution of the Short-Range, One-Band Model in One Dimension, Phys. Rev. Lett. 20, 1445 (1968)";
		out.method = "num. integration with gsl";
	}
	// U=0 ladder
	else if (Ly == 2ul and n == 1. and P.ARE_ALL_ZERO<double>({"U","tPrime","t0","V","Bz","Bx","J","J3site"}))
	{
		if (t/tRung <= 0.5) {out.value = -tRung;}
		else
		{
			if (isinf(L)) {out.value = -tRung-2.*M_1_PI*tRung*(sqrt(pow(2.*t/tRung,2)-1.)-acos(0.5*tRung/t));}
		}
		out.source = "Zheng Weihong, J. Oitmaa, C. J. Hamer, R. J. Bursill, Numerical studies of the two-leg Hubbard ladder, J. Phys.: Condens. Matter 13 (2001) 433–448";
		out.method = "analytical";
	}
	
	return out;
}

}

#endif

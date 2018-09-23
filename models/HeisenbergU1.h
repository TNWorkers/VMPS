#ifndef STRAWBERRY_HEISENBERGU1
#define STRAWBERRY_HEISENBERGU1

//include <array>

#include "models/HeisenbergObservables.h"
//include "Mpo.h"
#include "symmetry/U1.h"
//include "bases/SpinBase.h"
//include "DmrgExternal.h"
//include "ParamHandler.h" // from HELPERS
//include "symmetry/kind_dummies.h"
#include "ParamReturner.h"

namespace VMPS
{

/** \class HeisenbergU1
  * \ingroup Heisenberg
  *
  * \brief Heisenberg Model
  *
  * MPO representation of
  * \f[
  * H =  J \sum_{<ij>} \left(\mathbf{S_i} \cdot \mathbf{S_j}\right) 
  *     +J' \sum_{<<ij>>} \left(\mathbf{S_i} \cdot \mathbf{S_j}\right)
  *     -B_z \sum_i S^z_i
  *     +K_z \sum_i \left(S^z_i\right)^2
  *     -D_y \sum_{<ij>} \left(\mathbf{S_i} \times \mathbf{S_j}\right)_y
  *     -D_y' \sum_{<<ij>>} \left(\mathbf{S_i} \times \mathbf{S_j}\right)_y
  * \f]
  *
  * \param D : \f$D=2S+1\f$ where \f$S\f$ is the spin
  * \note Makes use of the \f$S^z\f$ U(1) symmetry.
  * \note The default variable settings can be seen in \p HeisenbergU1::defaults.
  * \note \f$J>0\f$ is antiferromagnetic
  * \note Isotropic \f$J\f$ is required here. For XXZ coupling, use VMPS::HeisenbergU1XXZ.
  */
class HeisenbergU1 : public Mpo<Sym::U1<Sym::SpinU1>,double>, public HeisenbergObservables<Sym::U1<Sym::SpinU1> >, public ParamReturner
{
public:
	
	typedef Sym::U1<Sym::SpinU1> Symmetry;
	MAKE_TYPEDEFS(HeisenbergU1)
	
private:
	typedef Symmetry::qType qType;
	typedef SiteOperator<Symmetry,SparseMatrix<double> > OperatorType;
	
public:
	
	///@{
	HeisenbergU1() : Mpo<Symmetry>(), ParamReturner(HeisenbergU1::sweep_defaults) {};
	HeisenbergU1 (const size_t &L);
	HeisenbergU1 (const size_t &L, const vector<Param> &params);
	///@}
	
	/**
	 * \describe_set_operators
	 *
	 * \param B : Base class from which the local operators are received
	 * \param P : The parameters
	 * \param loc : The location in the chain
	 */
	template<typename Symmetry_>
	static HamiltonianTermsXd<Symmetry_> set_operators (const vector<SpinBase<Symmetry_> > &B, const ParamHandler &P, size_t loc=0);
	
	/**
	 * Validates whether a given total quantum number \p qnum is a possible target quantum number for an Mps.
	 * \returns \p true if valid, \p false if not
	 */
	bool validate (qarray<1> qnum) const;
	
	static const std::map<string,std::any> defaults;
	static const std::map<string,std::any> sweep_defaults;
};

const std::map<string,std::any> HeisenbergU1::defaults = 
{
	{"J",1.}, {"Jprime",0.}, {"Jrung",1.},
	{"Bz",0.}, {"Kz",0.},
	{"D",2ul}, {"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}
};

const std::map<string,std::any> HeisenbergU1::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.e-11}, {"lim_alpha",10ul}, {"eps_svd",1.e-7},
	{"Dincr_abs", 4ul}, {"Dincr_per", 2ul}, {"Dincr_rel", 1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",20ul}, {"min_halfsweeps",4ul},
	{"Dinit",5ul}, {"Qinit",10ul}, {"Dlimit",100ul},
	{"tol_eigval",1e-7}, {"tol_state",1e-6},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_2SITE}
};

HeisenbergU1::
HeisenbergU1 (const size_t &L)
:Mpo<Symmetry> (L, qarray<Symmetry::Nq>({0}), "", PROP::HERMITIAN, PROP::NON_UNITARY, PROP::HAMILTONIAN),
 HeisenbergObservables(L),
 ParamReturner(HeisenbergU1::sweep_defaults)
{}

HeisenbergU1::
HeisenbergU1 (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, qarray<Symmetry::Nq>({0}), "", PROP::HERMITIAN, PROP::NON_UNITARY, PROP::HAMILTONIAN),
 HeisenbergObservables(L,params,HeisenbergU1::defaults),
 ParamReturner(HeisenbergU1::sweep_defaults)
{
	ParamHandler P(params,defaults);
	
	size_t Lcell = P.size();
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(B[l].get_basis(),l);
	}
	
	for (size_t l=0; l<N_sites; ++l)
	{
		Terms[l] = set_operators(B,P,l%Lcell);
		
		stringstream ss;
		ss << "Ly=" << P.get<size_t>("Ly",l%Lcell);
		Terms[l].info.push_back(ss.str());
	}
	
	this->construct_from_Terms(Terms, Lcell, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
	this->precalc_TwoSiteData();
}

bool HeisenbergU1::
validate (qarray<1> qnum) const
{
	frac Smax(0,1);
	frac q_in(qnum[0],2);
	for (size_t l=0; l<N_sites; ++l) { Smax+=frac(B[l].get_D()-1,2); }
	if (Smax.denominator()==q_in.denominator() and q_in <= Smax) {return true;}
	else {return false;}
}

template<typename Symmetry_>
HamiltonianTermsXd<Symmetry_> HeisenbergU1::
set_operators (const vector<SpinBase<Symmetry_> > &B, const ParamHandler &P, size_t loc)
{
	HamiltonianTermsXd<Symmetry_> Terms;
	
	auto save_label = [&Terms] (string label)
	{
		if (label!="") {Terms.info.push_back(label);}
	};
	
	stringstream ss;
	ss << "S=" << print_frac_nice(frac(P.get<size_t>("D",loc)-1,2));
	save_label(ss.str());
	
	size_t lp1 = (loc+1)%B.size();
	
	// J terms
	
	auto [J,Jpara,Jlabel] = P.fill_array2d<double>("J","Jpara",{{B[loc].orbitals(),B[lp1].orbitals()}},loc);
	save_label(Jlabel);
	
	for (int i=0; i<B[loc].orbitals(); ++i)
	for (int j=0; j<B[lp1].orbitals(); ++j)
	{
		if (Jpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(0.5*Jpara(i,j), B[loc].Scomp(SP,i), B[loc].Scomp(SM,i)));
			Terms.tight.push_back(make_tuple(0.5*Jpara(i,j), B[loc].Scomp(SM,i), B[loc].Scomp(SP,i)));
			Terms.tight.push_back(make_tuple(    Jpara(i,j), B[loc].Scomp(SZ,i), B[loc].Scomp(SZ,i)));
		}
	}
	
	// J' terms
	
	param0d Jprime = P.fill_array0d<double>("Jprime","Jprime",loc);
	save_label(Jprime.label);
	
	if (Jprime.x != 0.)
	{
		assert(B[loc].orbitals() == 1 and "Cannot do a ladder with J' terms!");
		
		Terms.nextn.push_back(make_tuple(0.5*Jprime.x, B[loc].Scomp(SP), B[loc].Scomp(SM), B[loc].Id()));
		Terms.nextn.push_back(make_tuple(0.5*Jprime.x, B[loc].Scomp(SM), B[loc].Scomp(SP), B[loc].Id()));
		Terms.nextn.push_back(make_tuple(    Jprime.x, B[loc].Scomp(SZ), B[loc].Scomp(SZ), B[loc].Id()));
	}
	
	// local terms
	
	auto [Jrung,Jperp,Jperplabel] = P.fill_array2d<double>("Jrung","J","Jperp",B[loc].orbitals(),loc,P.get<bool>("CYLINDER"));
	save_label(Jperplabel);
	
	auto [Bz,Bzorb,Bzlabel] = P.fill_array1d<double>("Bz","Bzorb",B[loc].orbitals(),loc);
	save_label(Bzlabel);
	
	auto [Kz,Kzorb,Kzlabel] = P.fill_array1d<double>("Kz","Kzorb",B[loc].orbitals(),loc);
	save_label(Kzlabel);
	
	Terms.name = "Heisenberg";
	
	ArrayXd Bxorb   = B[loc].ZeroField();
	ArrayXd Kxorb   = B[loc].ZeroField();
	ArrayXXd Dyperp = B[loc].ZeroHopping();
	
	Terms.local.push_back(make_tuple(1., B[loc].HeisenbergHamiltonian(Jperp,Jperp,Bzorb,Bxorb,Kzorb,Kxorb,Dyperp)));
	
	return Terms;
}

} //end namespace VMPS

#endif

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
#include "Geometry2D.h" // from TOOLS

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
  *	 +J' \sum_{<<ij>>} \left(\mathbf{S_i} \cdot \mathbf{S_j}\right)
  *	 -B_z \sum_i S^z_i
  *	 +K_z \sum_i \left(S^z_i\right)^2
  *	 -D_y \sum_{<ij>} \left(\mathbf{S_i} \times \mathbf{S_j}\right)_y
  *	 -D_y' \sum_{<<ij>>} \left(\mathbf{S_i} \times \mathbf{S_j}\right)_y
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
	
	static qarray<1> singlet() {return qarray<1>{0};};
	
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
	 * \param Terms : \p HamiltonianTerms instance
	 */
	template<typename Symmetry_>
	//static HamiltonianTermsXd<Symmetry_> set_operators (const vector<SpinBase<Symmetry_> > &B, const ParamHandler &P, size_t loc=0);
	static void set_operators(const std::vector<SpinBase<Symmetry_>> &B, const ParamHandler &P, HamiltonianTermsXd<Symmetry_> &Terms);
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
	{"Dinit",8ul}, {"Qinit",7ul}, {"Dlimit",100ul},
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
	HamiltonianTermsXd<Symmetry> Terms(N_sites, P.get<bool>("OPEN_BC"));
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(B[l].get_basis(),l);
	}
	
	set_operators(B,P,Terms);
	
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
void HeisenbergU1::
set_operators (const std::vector<SpinBase<Symmetry_>> &B, const ParamHandler &P, HamiltonianTermsXd<Symmetry_> &Terms)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = Terms.size();
	Terms.set_name("Heisenberg");
	
	for (std::size_t loc=0; loc<N_sites; ++loc)
	{
		size_t lp1 = (loc+1)%N_sites;
		size_t lp2 = (loc+2)%N_sites;
		
		std::size_t orbitals = B[loc].orbitals();
		std::size_t next_orbitals = B[lp1].orbitals();
		std::size_t nextn_orbitals = B[(loc+2)%N_sites].orbitals();
		
		stringstream ss1, ss2;
		ss1 << "S=" << print_frac_nice(frac(P.get<size_t>("D",loc%Lcell)-1,2));
		ss2 << "Ly=" << P.get<size_t>("Ly",loc%Lcell);
		Terms.save_label(loc, ss1.str());
		Terms.save_label(loc, ss2.str());
		
		// Local terms: B, K and J⟂
		
		param1d Bz = P.fill_array1d<double>("Bz", "Bzorb", orbitals, loc%Lcell);
		param1d Kz = P.fill_array1d<double>("Kz", "Kzorb", orbitals, loc%Lcell);
		param2d Jperp = P.fill_array2d<double>("Jrung", "J", "Jperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		
		Terms.save_label(loc, Bz.label);
		Terms.save_label(loc, Kz.label);
		Terms.save_label(loc, Jperp.label);
		
		Eigen::ArrayXd Bx_array = B[loc].ZeroField();
		Eigen::ArrayXd mu_array = B[loc].ZeroField();
		Eigen::ArrayXd Kx_array = B[loc].ZeroField();
		Eigen::ArrayXXd Dyperp_array = B[loc].ZeroHopping();
		
		Terms.push_local(loc, 1., B[loc].HeisenbergHamiltonian(Jperp.a, Jperp.a, Bz.a, Bx_array, mu_array, Kz.a, Kx_array, Dyperp_array));
		
		if (P.HAS("Jfull"))
		{
			ArrayXXd Full = P.get<Eigen::ArrayXXd>("Jfull");
			vector<vector<std::pair<size_t,double> > > R = Geometry2D::rangeFormat(Full);
			
			if (P.get<bool>("OPEN_BC")) {assert(R.size() ==   N_sites and "Use an (N_sites)x(N_sites) hopping matrix for open BC!");}
			else                        {assert(R.size() >= 2*N_sites and "Use at least a (2*N_sites)x(N_sites) hopping matrix for infinite BC!");}
			
			for (size_t h=0; h<R[loc].size(); ++h)
			{
				size_t range = R[loc][h].first;
				double value = R[loc][h].second;
				
				size_t Ntrans = (range == 0)? 0:range-1;
				vector<SiteOperator<Symmetry_,double> > TransOps(Ntrans);
				for (size_t i=0; i<Ntrans; ++i)
				{
					TransOps[i] = B[(loc+i+1)%N_sites].Id();
				}
				
				if (range != 0)
				{
					auto SP_loc = B[loc].Scomp(SP);
					auto SM_loc = B[loc].Scomp(SM);
					auto SZ_loc = B[loc].Scomp(SZ);
					
					auto SP_hop = B[(loc+range)%N_sites].Scomp(SP);
					auto SM_hop = B[(loc+range)%N_sites].Scomp(SM);
					auto SZ_hop = B[(loc+range)%N_sites].Scomp(SZ);
					
					Terms.push(range, loc, 0.5*value, SP_loc, TransOps, SM_hop);
					Terms.push(range, loc, 0.5*value, SM_loc, TransOps, SP_hop);
					Terms.push(range, loc, 0.5*value, SZ_loc, TransOps, SZ_hop);
				}
			}
			
			stringstream ss;
			ss << "Jᵢⱼ(" << Geometry2D::hoppingInfo(Full) << ")";
			Terms.save_label(loc, ss.str());
			continue;
		}
		else
		{
			// Nearest-neighbour terms: J
			
			param2d Jpara = P.fill_array2d<double>("J", "Jpara", {orbitals, next_orbitals}, loc%Lcell);
			Terms.save_label(loc, Jpara.label);
			
			if (loc < N_sites-1 or !P.get<bool>("OPEN_BC"))
			{
				for (std::size_t alfa=0; alfa < orbitals; ++alfa)
				for (std::size_t beta=0; beta < next_orbitals; ++beta)
				{
					Terms.push_tight(loc, 0.5*Jpara(alfa,beta), B[loc].Scomp(SP,alfa), B[lp1].Scomp(SM,beta));
					Terms.push_tight(loc, 0.5*Jpara(alfa,beta), B[loc].Scomp(SM,alfa), B[lp1].Scomp(SP,beta));
					Terms.push_tight(loc,     Jpara(alfa,beta), B[loc].Scomp(SZ,alfa), B[lp1].Scomp(SZ,beta));
				}
			}
			
			// Next-nearest-neighbour terms: J
			
			param2d Jprime = P.fill_array2d<double>("Jprime", "Jprime_array", {orbitals, nextn_orbitals}, loc%Lcell);
			Terms.save_label(loc, Jprime.label);
			
			if (loc < N_sites-2 or !P.get<bool>("OPEN_BC"))
			{
				for (std::size_t alfa=0; alfa < orbitals; ++alfa)
				for (std::size_t beta=0; beta < nextn_orbitals; ++beta)
				{
					Terms.push_nextn(loc, 0.5*Jprime(alfa,beta), B[loc].Scomp(SP,alfa), B[lp1].Id(), B[lp2].Scomp(SM,beta));
					Terms.push_nextn(loc, 0.5*Jprime(alfa,beta), B[loc].Scomp(SM,alfa), B[lp1].Id(), B[lp2].Scomp(SP,beta));
					Terms.push_nextn(loc,     Jprime(alfa,beta), B[loc].Scomp(SZ,alfa), B[lp1].Id(), B[lp2].Scomp(SZ,beta));
				}
			}
		}
	}
}

} //end namespace VMPS

#endif

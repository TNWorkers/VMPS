#ifndef STRAWBERRY_HEISENBERGU1XXZ
#define STRAWBERRY_HEISENBERGU1XXZ

#include "models/HeisenbergU1.h"

namespace VMPS
{
	
/** \class HeisenbergU1XXZ
  * \ingroup Heisenberg
  *
  * \brief Heisenberg Model with XXZ-coupling
  *
  * MPO representation of
  \f[
  H =  J_{xy} \sum_{<ij>} \left(S^x_iS^x_j+S^y_iS^y_j\right) + J_z \sum_{<ij>} S^z_iS^z_j 
      +J'_{xy} \sum_{<<ij>>} \left(S^x_iS^x_j+S^y_iS^y_j\right) + J'_z \sum_{<<ij>>} S^z_iS^z_j 
      -B_z \sum_i S^z_i
      +K_z \sum_i \left(S^z_i\right)^2
      -D_y \sum_{<ij>} \left(\mathbf{S_i} \times \mathbf{S_j}\right)_y
      -D_y' \sum_{<<ij>>} \left(\mathbf{S_i} \times \mathbf{S_j}\right)_y
  \f]
  *
  \param D : \f$D=2S+1\f$ where \f$S\f$ is the spin
  \note Makes use of the \f$S^z\f$ U(1) symmetry.
  \note The default variable settings can be seen in \p HeisenbergU1XXZ::defaults.
  \note \f$J>0\f$ is antiferromagnetic.
*/
class HeisenbergU1XXZ : public HeisenbergU1
{
public:
	typedef Sym::U1<Sym::SpinU1> Symmetry;
	MAKE_TYPEDEFS(HeisenbergU1XXZ)
	
	static qarray<1> singlet() {return qarray<1>{0};};
	
public:
	
	HeisenbergU1XXZ() : HeisenbergU1() {};
	HeisenbergU1XXZ (const size_t &L, const vector<Param> &params);
	
	template<typename Symmetry_>
	static void add_operators(const std::vector<SpinBase<Symmetry_>> &B, const ParamHandler &P, HamiltonianTermsXd<Symmetry_> &Terms);
	
	static const std::map<string,std::any> defaults;
};

const std::map<string,std::any> HeisenbergU1XXZ::defaults = 
{
	{"Jxy",1.}, {"Jxyprime",0.}, {"Jxyrung",1.},
	{"Jz",0.}, {"Jzprime",0.}, {"Jzrung",0.},
	
	{"Dy",0.}, {"Dyprime",0.}, {"Dyrung",0.},
	{"Bz",0.}, {"Kz",0.},
	{"D",2ul}, {"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}, 
	
	// for consistency during inheritance (should not be set for XXZ!):
	{"J",0.}, {"Jprime",0.}
};

HeisenbergU1XXZ::
HeisenbergU1XXZ (const size_t &L, const vector<Param> &params)
:HeisenbergU1(L)
{
	ParamHandler P(params,HeisenbergU1XXZ::defaults);
	
	size_t Lcell = P.size();
	HamiltonianTermsXd<Symmetry> Terms(N_sites, P.get<bool>("OPEN_BC"));
	B.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		
		B[l] = SpinBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), P.get<size_t>("D",l%Lcell));
		setLocBasis(B[l].get_basis(),l);
	}
	
	set_operators(B,P,Terms);
	add_operators(B,P,Terms);
	
	this->construct_from_Terms(Terms, Lcell, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
	this->precalc_TwoSiteData();
}

/*template<typename Symmetry_>
void HeisenbergU1XXZ::
add_operators (HamiltonianTermsXd<Symmetry_> &Terms, const vector<SpinBase<Symmetry_> > &B, const ParamHandler &P, size_t loc)
{
	auto save_label = [&Terms] (string label)
	{
		if (label!="") {Terms.info.push_back(label);}
	};
	
	size_t lp1 = (loc+1)%B.size();
	
	// Jxy/Jz terms
	
	auto [Jxy,Jxypara,Jxylabel] = P.fill_array2d<double>("Jxy","Jxypara",{{B[loc].orbitals(),B[lp1].orbitals()}},loc);
	save_label(Jxylabel);
	
	auto [Jz,Jzpara,Jzlabel] = P.fill_array2d<double>("Jz","Jzpara",{{B[loc].orbitals(),B[lp1].orbitals()}},loc);
	save_label(Jzlabel);
	
	for (int i=0; i<B[loc].orbitals(); ++i)
	for (int j=0; j<B[lp1].orbitals(); ++j)
	{
		if (Jxypara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(0.5*Jxypara(i,j), B[loc].Scomp(SP,i), B[loc].Scomp(SM,j)));
			Terms.tight.push_back(make_tuple(0.5*Jxypara(i,j), B[loc].Scomp(SM,i), B[loc].Scomp(SP,j)));
		}
		
		if (Jzpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(Jzpara(i,j), B[loc].Scomp(SZ,i), B[loc].Scomp(SZ,j)));
		}
	}
	
	// Jxy'/Jz' terms
	
	param0d Jxyprime = P.fill_array0d<double>("Jxyprime","Jxyprime",loc);
	save_label(Jxyprime.label);
	
	if (Jxyprime.x != 0.)
	{
		assert(B[loc].orbitals() == 1 and "Cannot do a ladder with Jxy' terms!");
		
		Terms.nextn.push_back(make_tuple(0.5*Jxyprime.x, B[loc].Scomp(SP), B[loc].Scomp(SM), B[loc].Id()));
		Terms.nextn.push_back(make_tuple(0.5*Jxyprime.x, B[loc].Scomp(SM), B[loc].Scomp(SP), B[loc].Id()));
	}
	
	param0d Jzprime = P.fill_array0d<double>("Jzprime","Jzprime",loc);
	save_label(Jzprime.label);
	
	if (Jzprime.x != 0.)
	{
		assert(B[loc].orbitals() == 1 and "Cannot do a ladder with Jz' terms!");
		
		Terms.nextn.push_back(make_tuple(Jzprime.x, B[loc].Scomp(SZ), B[loc].Scomp(SZ), B[loc].Id()));
	}
	
	// local terms
	
//	param0d Jxyperp = P.fill_array0d<double>("Jxy","Jxyperp",loc);
//	save_label(Jxyperp.label);
//	
//	param0d Jzperp = P.fill_array0d<double>("Jz","Jzperp",loc);
//	save_label(Jzperp.label);
	
	auto [Jxy_,Jxyperp,Jxyperplabel] = P.fill_array2d<double>("Jxyrung","Jxy","Jxyperp",B[loc].orbitals(),loc,P.get<bool>("CYLINDER"));
	save_label(Jxyperplabel);
	
	auto [Jz_,Jzperp,Jzperplabel] = P.fill_array2d<double>("Jzrung","Jz","Jzperp",B[loc].orbitals(),loc,P.get<bool>("CYLINDER"));
	save_label(Jzperplabel);
	
	ArrayXd Bzorb   = B[loc].ZeroField();
	ArrayXd Bxorb   = B[loc].ZeroField();
	ArrayXd Kzorb   = B[loc].ZeroField();
	ArrayXd Kxorb   = B[loc].ZeroField();
	ArrayXXd Dyperp = B[loc].ZeroHopping();
	
	Terms.local.push_back(make_tuple(1., B[loc].HeisenbergHamiltonian(Jxyperp,Jzperp,Bzorb,Bxorb,Kzorb,Kxorb,Dyperp)));
	
	Terms.name = (P.HAS_ANY_OF({"Jxy","Jxypara","Jxyperp"},loc))? "XXZ":"Ising";
}*/

template<typename Symmetry_>
void HeisenbergU1XXZ::
add_operators (const std::vector<SpinBase<Symmetry_>> &B, const ParamHandler &P, HamiltonianTermsXd<Symmetry_> &Terms)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = Terms.size();
	if (P.HAS_ANY_OF({"Jxy", "Jxypara", "Jxyperp"}))
	{
		Terms.set_name("XXZ");
	}
	else
	{
		Terms.set_name("Ising");
	}
	
	for (std::size_t loc=0; loc<N_sites; ++loc)
	{
		size_t lp1 = (loc+1)%N_sites;
		size_t lp2 = (loc+2)%N_sites;
		
		std::size_t orbitals       = B[loc].orbitals();
		std::size_t next_orbitals  = B[lp1].orbitals();
		std::size_t nextn_orbitals = B[lp2].orbitals();
		
//		stringstream ss1, ss2;
//		ss2 << "Ly=" << P.get<size_t>("Ly",loc%Lcell);
//		Terms.save_label(loc, ss2.str());
		
		// Local terms: J⟂
		
		param2d Jxyperp = P.fill_array2d<double>("Jxyrung", "Jxy", "Jxyperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Jzperp  = P.fill_array2d<double>("Jzrung",  "Jz",  "Jzperp",  orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		
		Terms.save_label(loc, Jxyperp.label);
		Terms.save_label(loc, Jzperp.label);
		
		ArrayXd Bz_array      = B[loc].ZeroField();
		ArrayXd Bx_array      = B[loc].ZeroField();
		ArrayXd mu_array      = B[loc].ZeroField();
		ArrayXd Kz_array      = B[loc].ZeroField();
		ArrayXd Kx_array      = B[loc].ZeroField();
		ArrayXXd Dyperp_array = B[loc].ZeroHopping();
		
		Terms.push_local(loc, 1., B[loc].HeisenbergHamiltonian(Jxyperp.a, Jzperp.a, Bz_array, Bx_array, mu_array, Kz_array, Kx_array, Dyperp_array));
		
		// Nearest-neighbour terms: J
		
		param2d Jxypara = P.fill_array2d<double>("Jxy", "Jxypara", {orbitals, next_orbitals}, loc%Lcell);
		param2d Jzpara  = P.fill_array2d<double>("Jz",  "Jzpara",  {orbitals, next_orbitals}, loc%Lcell);
		
		Terms.save_label(loc, Jxypara.label);
		Terms.save_label(loc, Jzpara.label);
		
//		if (!P.HAS("Jxy"))
//		{
//			Terms.save_label(loc, "def.Jxy=1.");
//		}
		
		if (loc < N_sites-1 or !P.get<bool>("OPEN_BC"))
		{
			for (std::size_t alfa=0; alfa < orbitals; ++alfa)
			for (std::size_t beta=0; beta < next_orbitals; ++beta)
			{
				Terms.push_tight(loc, 0.5*Jxypara(alfa,beta), B[loc].Scomp(SP,alfa), B[lp1].Scomp(SM,beta));
				Terms.push_tight(loc, 0.5*Jxypara(alfa,beta), B[loc].Scomp(SM,alfa), B[lp1].Scomp(SP,beta));
				Terms.push_tight(loc,      Jzpara(alfa,beta), B[loc].Scomp(SZ,alfa), B[lp1].Scomp(SZ,beta));
			}
		}
		
		// Next-nearest-neighbour terms: J
		
		param2d Jxyprime = P.fill_array2d<double>("Jxyprime", "Jxyprime_array", {orbitals, nextn_orbitals}, loc%Lcell);
		param2d Jzprime  = P.fill_array2d<double>("Jzprime",  "Jzprime_array",  {orbitals, nextn_orbitals}, loc%Lcell);
		
		Terms.save_label(loc, Jxyprime.label);
		Terms.save_label(loc, Jzprime.label);
		
		if (loc < N_sites-2 or !P.get<bool>("OPEN_BC"))
		{
			for (std::size_t alfa=0; alfa < orbitals; ++alfa)
			for (std::size_t beta=0; beta < nextn_orbitals; ++beta)
			{
				Terms.push_nextn(loc, 0.5*Jxyprime(alfa,beta), B[loc].Scomp(SP,alfa), B[lp1].Id(), B[lp2].Scomp(SM,beta));
				Terms.push_nextn(loc, 0.5*Jxyprime(alfa,beta), B[loc].Scomp(SM,alfa), B[lp1].Id(), B[lp2].Scomp(SP,beta));
				Terms.push_nextn(loc,      Jzprime(alfa,beta), B[loc].Scomp(SZ,alfa), B[lp1].Id(), B[lp2].Scomp(SZ,beta));
			}
		}
	}
}

} //end namespace VMPS

#endif

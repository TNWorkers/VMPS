#ifndef STRAWBERRY_HUBBARDMODEL
#define STRAWBERRY_HUBBARDMODEL

//include "bases/FermionBase.h"
#include "symmetry/S1xS2.h"
#include "symmetry/U1.h"
//include "Mpo.h"
//include "ParamHandler.h" // from HELPERS
#include "models/HubbardObservables.h"
#include "ParamReturner.h"

namespace VMPS
{

/**
 * \class HubbardU1xU1
 * \ingroup Hubbard
 * \brief Hubbard model with U(1) symmetries.
 * MPO representation of the Hubbard model
 * \f[
 * 	H = -t \sum_{<ij>\sigma} \left( c^\dagger_{i\sigma}c_{j\sigma} + h.c. \right)
 * 	    -t^{\prime} \sum_{<<ij>>\sigma} \left( c^\dagger_{i\sigma}c_{j\sigma} +h.c. \right)
 * 	    +\sum_i \left(t_{0,i}-\mu\right) n_i
 * 	    +U \sum_i n_{i\uparrow} n_{i\downarrow}
 * 	    +V \sum_{<ij>} n_{i} n_{j}
 * 	    -B_z \sum_{i} \left(n_{i\uparrow}-n_{i\downarrow}\right)
 * 	    +H_{tJ}
 * 	    +H_{3-site}
 * \f]
 * with
 * \f[
 * H_{tJ} = +J \sum_{<ij>} (\mathbf{S}_{i} \mathbf{S}_{j} - \frac{1}{4} n_in_j)
 * \f]
 * \note: The term before \f$n_i n_j\f$ is not set and has to be adjusted with \p V
 * \f[
 * H_{3-site} = -\frac{J}{4} \sum_{<ijk>\sigma} (c^\dagger_{i\sigma} n_{j,-\sigma} c_{k\sigma} - c^\dagger_{i\sigma} S^{-\sigma}_j c_{k,-\sigma} + h.c.) \
 * \f]
 * \note Makes use of the U(1) particle conservation symmetry for both spin components separatly.
 *       You can change this by choosing another symmetry class. For example, to use the magnetization and the particle number use:
 * \code{.cpp}
 *     Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::U1<Sym::ChargeU1> >
 * \endcode
 * \note The default variable settings can be seen in \p HubbardU1xU1::defaults.
 * \note If the NNN-hopping is positive, the ground state energy is lowered.
 * \warning \f$J>0\f$ is antiferromagnetic
 */
class HubbardU1xU1 : public Mpo<Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::U1<Sym::ChargeU1> >,double>,
                     public HubbardObservables<Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::U1<Sym::ChargeU1> > >, 
                     public ParamReturner
{
public:
	
	typedef Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::U1<Sym::ChargeU1> > Symmetry;
	MAKE_TYPEDEFS(HubbardU1xU1)
	
	///@{
	HubbardU1xU1() : Mpo(){};
	HubbardU1xU1 (const size_t &L, const vector<Param> &params);
	///@}
	
	static qarray<2> singlet (int N) {return qarray<2>{0,N};};
	
	template<typename Symmetry_> 
	//static HamiltonianTermsXd<Symmetry_> set_operators (const vector<FermionBase<Symmetry_> > &F, const ParamHandler &P, size_t loc=0);
    static void set_operators(const std::vector<FermionBase<Symmetry_>> &F, const ParamHandler &P, HamiltonianTermsXd<Symmetry_> &Terms);
    
	/**Default parameters.*/
	static const std::map<string,std::any> defaults;
};

const std::map<string,std::any> HubbardU1xU1::defaults = 
{
	{"t",1.}, {"tPrime",0.}, {"tRung",1.},
	{"mu",0.}, {"t0",0.}, 
	{"U",0.}, {"V",0.}, {"Vrung",0.}, 
	{"Bz",0.}, 
	{"J",0.}, {"Jperp",0.}, {"J3site",0.},
	{"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}
};

HubbardU1xU1::
HubbardU1xU1 (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, PROP::HAMILTONIAN),
 HubbardObservables(L,params,HubbardU1xU1::defaults),
 ParamReturner()
{
	/*ParamHandler P(params,HubbardU1xU1::defaults);
	
	size_t Lcell = P.size();
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(F[l].get_basis(),l);
	}
	
	for (size_t l=0; l<N_sites; ++l)
	{
		Terms[l] = set_operators(F,P,l%Lcell);
		
		stringstream ss;
		ss << "Ly=" << P.get<size_t>("Ly",l%Lcell);
		Terms[l].info.push_back(ss.str());
	}
	
	this->construct_from_Terms(Terms, Lcell, false, P.get<bool>("OPEN_BC"));
	this->precalc_TwoSiteData();*/
    
    ParamHandler P(params, HubbardU1xU1::defaults);
    
    size_t Lcell = P.size();
    HamiltonianTermsXd<Symmetry> Terms(N_sites, P.get<bool>("OPEN_BC"));
    
    for (size_t l=0; l<N_sites; ++l)
    {
        N_phys += P.get<size_t>("Ly",l%Lcell);
        setLocBasis(F[l].get_basis(),l);
    }
    
    set_operators(F,P,Terms);
    // cout << Terms.print_info() << endl;
    
    this->construct_from_Terms(Terms, Lcell, false, P.get<bool>("OPEN_BC"));
    this->precalc_TwoSiteData();
}

/*template<typename Symmetry_>
HamiltonianTermsXd<Symmetry_> HubbardU1xU1::
set_operators (const vector<FermionBase<Symmetry_> > &F, const ParamHandler &P, size_t loc)
{
	HamiltonianTermsXd<Symmetry_> Terms;
	
	auto save_label = [&Terms] (string label)
	{
		if (label!="") {Terms.info.push_back(label);}
	};
	
	size_t lp1 = (loc+1)%F.size();
	
	// NN terms
	
	auto [t,tPara,tlabel] = P.fill_array2d<double>("t","tPara",{{F[loc].orbitals(),F[lp1].orbitals()}},loc);
	save_label(tlabel);
	
	auto [V,Vpara,Vlabel] = P.fill_array2d<double>("V","Vpara",{{F[loc].orbitals(),F[lp1].orbitals()}},loc);
	save_label(Vlabel);
	
	auto [J,Jpara,Jlabel] = P.fill_array2d<double>("J","Jpara",{{F[loc].orbitals(),F[lp1].orbitals()}},loc);
	save_label(Jlabel);
	
	for (int i=0; i<F[loc].orbitals(); ++i)
	for (int j=0; j<F[lp1].orbitals(); ++j)
	{
		if (tPara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(-tPara(i,j), F[loc].cdag(UP,i)  * F[loc].sign(), F[loc].c(UP,j)));
			Terms.tight.push_back(make_tuple(-tPara(i,j), F[loc].cdag(DN,i)  * F[loc].sign(), F[loc].c(DN,j)));
			Terms.tight.push_back(make_tuple(-tPara(i,j), -1.*F[loc].c(UP,i) * F[loc].sign(), F[loc].cdag(UP,j)));
			Terms.tight.push_back(make_tuple(-tPara(i,j), -1.*F[loc].c(DN,i) * F[loc].sign(), F[loc].cdag(DN,j)));
		}
		
		if (Vpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(Vpara(i,j), F[loc].n(i), F[loc].n(i)));
		}
		
		if (Jpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(0.5*Jpara(i,j), F[loc].Sp(i), F[loc].Sm(i)));
			Terms.tight.push_back(make_tuple(0.5*Jpara(i,j), F[loc].Sm(i), F[loc].Sp(i)));
			Terms.tight.push_back(make_tuple(Jpara(i,j),     F[loc].Sz(i), F[loc].Sz(i)));
		}
	}
	
	// NNN terms
	
	param0d tPrime = P.fill_array0d<double>("tPrime","tPrime",loc);
	save_label(tPrime.label);
	
	if (tPrime.x != 0.)
	{
		assert(F[loc].orbitals() == 1 and "Cannot do a ladder with t'!");
		
		Terms.nextn.push_back(make_tuple(-tPrime.x, F[loc].cdag(UP)  * F[loc].sign(), F[loc].c(UP),    F[loc].sign()));
		Terms.nextn.push_back(make_tuple(-tPrime.x, F[loc].cdag(DN)  * F[loc].sign(), F[loc].c(DN),    F[loc].sign()));
		Terms.nextn.push_back(make_tuple(-tPrime.x, -1.*F[loc].c(UP) * F[loc].sign(), F[loc].cdag(UP), F[loc].sign()));
		Terms.nextn.push_back(make_tuple(-tPrime.x, -1.*F[loc].c(DN) * F[loc].sign(), F[loc].cdag(DN), F[loc].sign()));
	}
	
	param0d J3site = P.fill_array0d<double>("J3site","J3site",loc);
	save_label(J3site.label);
	
	if (J3site.x != 0.)
	{
		lout << "Warning! J3site has to be tested against ED!" << endl;
		
		assert(F[loc].orbitals() == 1 and "Cannot do a ladder with 3-site J terms!");
		
		// old and probably wrong:
		
//		// three-site terms without spinflip
//		Terms.nextn.push_back(make_tuple(-0.25*J3site.x, F.cdag(UP), F.sign()*F.c(UP),    F.n(DN)*F.sign()));
//		Terms.nextn.push_back(make_tuple(-0.25*J3site.x, F.cdag(DN), F.sign()*F.c(DN),    F.n(UP)*F.sign()));
//		Terms.nextn.push_back(make_tuple(+0.25*J3site.x, F.c(UP),    F.sign()*F.cdag(UP), F.n(DN)*F.sign()));
//		Terms.nextn.push_back(make_tuple(+0.25*J3site.x, F.c(DN),    F.sign()*F.cdag(DN), F.n(UP)*F.sign()));
//		
//		// three-site terms with spinflip
//		Terms.nextn.push_back(make_tuple(+0.25*J3site.x, F.cdag(DN), F.sign()*F.c(UP),    F.Sp()*F.sign()));
//		Terms.nextn.push_back(make_tuple(+0.25*J3site.x, F.cdag(UP), F.sign()*F.c(DN),    F.Sm()*F.sign()));
//		Terms.nextn.push_back(make_tuple(-0.25*J3site.x, F.c(DN),    F.sign()*F.cdag(UP), F.Sm()*F.sign()));
//		Terms.nextn.push_back(make_tuple(-0.25*J3site.x, F.c(UP),    F.sign()*F.cdag(DN), F.Sp()*F.sign()));
		
		// new:
		
		// three-site terms without spinflip
		Terms.nextn.push_back(make_tuple(-0.25*J3site.x, F[loc].cdag(UP)  * F[loc].sign(), F[loc].c(UP),    F[loc].n(DN)*F[loc].sign()));
		Terms.nextn.push_back(make_tuple(-0.25*J3site.x, F[loc].cdag(DN)  * F[loc].sign(), F[loc].c(DN),    F[loc].n(UP)*F[loc].sign()));
		Terms.nextn.push_back(make_tuple(-0.25*J3site.x, -1.*F[loc].c(UP) * F[loc].sign(), F[loc].cdag(UP), F[loc].n(DN)*F[loc].sign()));
		Terms.nextn.push_back(make_tuple(-0.25*J3site.x, -1.*F[loc].c(DN) * F[loc].sign(), F[loc].cdag(DN), F[loc].n(UP)*F[loc].sign()));
		
		// three-site terms with spinflip
		Terms.nextn.push_back(make_tuple(+0.25*J3site.x, F[loc].cdag(DN)  * F[loc].sign(), F[loc].c(UP),    F[loc].Sp()*F[loc].sign()));
		Terms.nextn.push_back(make_tuple(+0.25*J3site.x, F[loc].cdag(UP)  * F[loc].sign(), F[loc].c(DN),    F[loc].Sm()*F[loc].sign()));
		Terms.nextn.push_back(make_tuple(+0.25*J3site.x, -1.*F[loc].c(DN) * F[loc].sign(), F[loc].cdag(UP), F[loc].Sm()*F[loc].sign()));
		Terms.nextn.push_back(make_tuple(+0.25*J3site.x, -1.*F[loc].c(UP) * F[loc].sign(), F[loc].cdag(DN), F[loc].Sp()*F[loc].sign()));
	}
	
	// local terms
	
	// Hubbard-U
	auto [U,Uorb,Ulabel] = P.fill_array1d<double>("U","Uorb",F[loc].orbitals(),loc);
	save_label(Ulabel);
	
	// t0
	auto [t0,t0orb,t0label] = P.fill_array1d<double>("t0","t0orb",F[loc].orbitals(),loc);
	save_label(t0label);
	
	// μ
	auto [mu,muorb,mulabel] = P.fill_array1d<double>("mu","muorb",F[loc].orbitals(),loc);
	save_label(mulabel);
	
	// Bz
	auto [Bz,Bzorb,Bzlabel] = P.fill_array1d<double>("Bz","Bzorb",F[loc].orbitals(),loc);
	save_label(Bzlabel);
	
	// t⟂
	auto [tRung,tPerp,tPerplabel] = P.fill_array2d<double>("tRung","t","tPerp",F[loc].orbitals(),loc,P.get<bool>("CYLINDER"));
	save_label(tPerplabel);
	
	// V⟂
	auto [Vrung,Vperp,Vperplabel] = P.fill_array2d<double>("Vrung","V","Vperp",F[loc].orbitals(),loc,P.get<bool>("CYLINDER"));
	save_label(Vperplabel);
	
	// J⟂
	auto [Jrung,Jperp,Jperplabel] = P.fill_array2d<double>("Jrung","J","Jperp",F[loc].orbitals(),loc,P.get<bool>("CYLINDER"));
	save_label(Jperplabel);
	
	if (isfinite(Uorb.sum()))
	{
		Terms.name = "Hubbard";
	}
	else
	{
		Terms.name = (P.HAS_ANY_OF({"J","J3site"}))? "t-J":"U=∞-Hubbard";
	}
	
	ArrayXd Bxorb = F[loc].ZeroField();
	
	Terms.local.push_back(make_tuple(1., F[loc].template HubbardHamiltonian<double>(Uorb,t0orb-muorb,Bzorb,Bxorb,tPerp,Vperp,Jperp)));
	
	return Terms;
}*/

template<typename Symmetry_>
void HubbardU1xU1::
set_operators (const std::vector<FermionBase<Symmetry_>> &F, const ParamHandler &P, HamiltonianTermsXd<Symmetry_> &Terms)
{
    std::size_t Lcell = P.size();
    std::size_t N_sites = Terms.size();
    bool U_infinite = true;
    
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        std::size_t orbitals = F[loc].orbitals();
        std::size_t next_orbitals = F[(loc+1)%N_sites].orbitals();
        std::size_t nextn_orbitals = F[(loc+2)%N_sites].orbitals();
        
        stringstream ss;
        ss << "Ly=" << P.get<size_t>("Ly",loc%Lcell);
        Terms.save_label(loc, ss.str());
        
        
        // Local terms: U, t0, μ, Bz, t⟂, V⟂, J⟂
        
        param1d U = P.fill_array1d<double>("U", "Uorb", orbitals, loc%Lcell);
        param1d t0 = P.fill_array1d<double>("t0", "t0orb", orbitals, loc%Lcell);
        param1d mu = P.fill_array1d<double>("mu", "muorb", orbitals, loc%Lcell);
        param1d Bz = P.fill_array1d<double>("Bz", "Bzorb", orbitals, loc%Lcell);
        param2d tperp = P.fill_array2d<double>("tRung", "t", "tPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
        param2d Vperp = P.fill_array2d<double>("Vrung", "V", "Vperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
        param2d Jperp = P.fill_array2d<double>("Jrung", "J", "Jperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
        
        Terms.save_label(loc, U.label);
        Terms.save_label(loc, t0.label);
        Terms.save_label(loc, mu.label);
        Terms.save_label(loc, Bz.label);
        Terms.save_label(loc, tperp.label);
        Terms.save_label(loc, Vperp.label);
        Terms.save_label(loc, Jperp.label);
        
        ArrayXd Bx_array = F[loc].ZeroField();
        
        Terms.push_local(loc, 1., F[loc].template HubbardHamiltonian<double>(U.a, t0.a - mu.a, Bz.a, Bx_array, tperp.a, Vperp.a, Jperp.a));
        
        if (isfinite(U.a.sum()))
        {
            U_infinite = false;
        }
    
        // Nearest-neighbour terms: t, V, J
    
        param2d tpara = P.fill_array2d<double>("t", "tPara", {orbitals, next_orbitals}, loc%Lcell);
        param2d Vpara = P.fill_array2d<double>("V", "Vpara", {orbitals, next_orbitals}, loc%Lcell);
        param2d Jpara = P.fill_array2d<double>("J", "Jpara", {orbitals, next_orbitals}, loc%Lcell);
        
        Terms.save_label(loc, tpara.label);
        Terms.save_label(loc, Vpara.label);
        Terms.save_label(loc, Jpara.label);
        
        if(loc < N_sites-1 || !P.get<bool>("OPEN_BC"))
        {
            for (std::size_t alpha=0; alpha<orbitals; ++alpha)
            {
                for (std::size_t beta=0; beta<next_orbitals; ++beta)
                {
                    Terms.push_tight(loc, -tpara.a(alpha, beta), F[loc].cdag(UP, alpha)*F[loc].sign(), F[(loc+1)%N_sites].c(UP, beta));
                    Terms.push_tight(loc, -tpara.a(alpha, beta), F[loc].cdag(DN, alpha)*F[loc].sign(), F[(loc+1)%N_sites].c(DN, beta));
                    Terms.push_tight(loc, +tpara.a(alpha, beta), F[loc].c(UP, alpha)*F[loc].sign(), F[(loc+1)%N_sites].cdag(UP, beta));
                    Terms.push_tight(loc, +tpara.a(alpha, beta), F[loc].c(DN, alpha)*F[loc].sign(), F[(loc+1)%N_sites].cdag(DN, beta));
                    
                    Terms.push_tight(loc, Vpara.a(alpha, beta), F[loc].n(alpha), F[(loc+1)%N_sites].n(beta));
                    
                    Terms.push_tight(loc, 0.5*Jpara.a(alpha, beta), F[loc].Sp(alpha), F[(loc+1)%N_sites].Sm(beta));
                    Terms.push_tight(loc, 0.5*Jpara.a(alpha, beta), F[loc].Sm(alpha), F[(loc+1)%N_sites].Sp(beta));
                    Terms.push_tight(loc, Jpara.a(alpha, beta), F[loc].Sz(alpha), F[(loc+1)%N_sites].Sz(beta));
                }
            }
        }
    
        // Next-nearest-neighbour terms: t'
    
        param2d tprime = P.fill_array2d<double>("tPrime", "tPrime_array", {orbitals, nextn_orbitals}, loc%Lcell);
        Terms.save_label(loc, tprime.label);
        
        if(loc < N_sites-2 || !P.get<bool>("OPEN_BC"))
        {
            for (std::size_t alpha=0; alpha<orbitals; ++alpha)
            {
                for (std::size_t beta=0; beta<nextn_orbitals; ++beta)
                {
                    Terms.push_nextn(loc, -tprime.a(alpha, beta), F[loc].cdag(UP, alpha)*F[loc].sign(), F[(loc+1)%N_sites].sign(), F[(loc+2)%N_sites].c(UP, beta));
                    Terms.push_nextn(loc, -tprime.a(alpha, beta), F[loc].cdag(DN, alpha)*F[loc].sign(), F[(loc+1)%N_sites].sign(), F[(loc+2)%N_sites].c(DN, beta));
                    Terms.push_nextn(loc, -tprime.a(alpha, beta), -1.*F[loc].c(UP, alpha)*F[loc].sign(), F[(loc+1)%N_sites].sign(), F[(loc+2)%N_sites].cdag(UP, beta));
                    Terms.push_nextn(loc, -tprime.a(alpha, beta), -1.*F[loc].c(DN, alpha)*F[loc].sign(), F[(loc+1)%N_sites].sign(), F[(loc+2)%N_sites].cdag(DN, beta));
                }
            }
        }
    
       /* param0d J3site = P.fill_array0d<double>("J3site","J3site",loc);
        save_label(J3site.label);
        
        if (J3site.x != 0.)
        {
            lout << "Warning! J3site has to be tested against ED!" << endl;
            
            assert(F[loc].orbitals() == 1 and "Cannot do a ladder with 3-site J terms!");
            
            // three-site terms without spinflip
            Terms.nextn.push_back(make_tuple(-0.25*J3site.x, F[loc].cdag(UP)  * F[loc].sign(), F[loc].c(UP),    F[loc].n(DN)*F[loc].sign()));
            Terms.nextn.push_back(make_tuple(-0.25*J3site.x, F[loc].cdag(DN)  * F[loc].sign(), F[loc].c(DN),    F[loc].n(UP)*F[loc].sign()));
            Terms.nextn.push_back(make_tuple(-0.25*J3site.x, -1.*F[loc].c(UP) * F[loc].sign(), F[loc].cdag(UP), F[loc].n(DN)*F[loc].sign()));
            Terms.nextn.push_back(make_tuple(-0.25*J3site.x, -1.*F[loc].c(DN) * F[loc].sign(), F[loc].cdag(DN), F[loc].n(UP)*F[loc].sign()));
            
            // three-site terms with spinflip
            Terms.nextn.push_back(make_tuple(+0.25*J3site.x, F[loc].cdag(DN)  * F[loc].sign(), F[loc].c(UP),    F[loc].Sp()*F[loc].sign()));
            Terms.nextn.push_back(make_tuple(+0.25*J3site.x, F[loc].cdag(UP)  * F[loc].sign(), F[loc].c(DN),    F[loc].Sm()*F[loc].sign()));
            Terms.nextn.push_back(make_tuple(+0.25*J3site.x, -1.*F[loc].c(DN) * F[loc].sign(), F[loc].cdag(UP), F[loc].Sm()*F[loc].sign()));
            Terms.nextn.push_back(make_tuple(+0.25*J3site.x, -1.*F[loc].c(UP) * F[loc].sign(), F[loc].cdag(DN), F[loc].Sp()*F[loc].sign()));
        }*/
    }
    
    if (!U_infinite)
    {
        Terms.set_name("Hubbard");
    }
    else if (P.HAS_ANY_OF({"J", "J3site"}))
    {
        Terms.set_name("t-J");
    }
    else
    {
        Terms.set_name("U=∞-Hubbard");
    }
}
    
////Mpo<Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::U1<Sym::ChargeU1> >,complex<double> > HubbardU1xU1::
////doublonPacket (complex<double> (*f)(int))
////{
////	stringstream ss;
////	ss << "doublonPacket";
////	
////	Mpo<Symmetry,complex<double> > Mout(N_sites, qarray<Symmetry::Nq>({-1,-1}), HubbardU1xU1::Nlabel, ss.str());
////	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F.get_basis(),l);}
////	
////	Mout.setLocalSum(F.c(UP)*F.c(DN), f);
////	return Mout;
////}

////Mpo<Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::U1<Sym::ChargeU1> >,complex<double> > HubbardU1xU1::
////electronPacket (complex<double> (*f)(int))
////{
////	assert(N_legs==1);
////	stringstream ss;
////	ss << "electronPacket";
////	
////	qarray<2> qdiff = {+1,0};
////	
////	vector<SuperMatrix<Symmetry,complex<double> > > M(N_sites);
////	M[0].setRowVector(2,F.dim());
//////	M[0](0,0) = f(0) * F.cdag(UP);
////	M[0](0,0).data = f(0) * F.cdag(UP).data; M[0](0,0).Q = F.cdag(UP).Q;
////	M[0](0,1) = F.Id();
////	
////	for (size_t l=1; l<N_sites-1; ++l)
////	{
////		M[l].setMatrix(2,F.dim());
//////		M[l](0,0) = complex<double>(1.,0.) * F.sign();
////		M[l](0,0).data = complex<double>(1.,0.) * F.sign().data; M[l](0,0).Q = F.sign().Q;
//////		M[l](1,0) = f(l) * F.cdag(UP);
////		M[l](1,0).data = f(l) * F.cdag(UP).data; M[l](1,0).Q = F.cdag(UP).Q;
////		M[l](0,1).setZero();
////		M[l](1,1) = F.Id();
////	}
////	
////	M[N_sites-1].setColVector(2,F.dim());
//////	M[N_sites-1](0,0) = complex<double>(1.,0.) * F.sign();
////	M[N_sites-1](0,0).data = complex<double>(1.,0.) * F.sign().data; M[N_sites-1](0,0).Q = F.sign().Q;
//////	M[N_sites-1](1,0) = f(N_sites-1) * F.cdag(UP);
////	M[N_sites-1](1,0).data = f(N_sites-1) * F.cdag(UP).data; M[N_sites-1](1,0).Q = F.cdag(UP).Q;
////	
////	Mpo<Symmetry,complex<double> > Mout(N_sites, M, qarray<Symmetry::Nq>(qdiff), HubbardU1xU1::Nlabel, ss.str());
////	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F.get_basis(),l);}
////	return Mout;
////}

////Mpo<Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::U1<Sym::ChargeU1> >,complex<double> > HubbardU1xU1::
////holePacket (complex<double> (*f)(int))
////{
////	assert(N_legs==1);
////	stringstream ss;
////	ss << "holePacket";
////	
////	qarray<2> qdiff = {-1,0};
////	
////	vector<SuperMatrix<Symmetry,complex<double> > > M(N_sites);
////	M[0].setRowVector(2,F.dim());
////	M[0](0,0) = f(0) * F.c(UP);
////	M[0](0,1) = F.Id();
////	
////	for (size_t l=1; l<N_sites-1; ++l)
////	{
////		M[l].setMatrix(2,F.dim());
////		M[l](0,0) = complex<double>(1.,0.) * F.sign();
////		M[l](1,0) = f(l) * F.c(UP);
////		M[l](0,1).setZero();
////		M[l](1,1) = F.Id();
////	}
////	
////	M[N_sites-1].setColVector(2,F.dim());
////	M[N_sites-1](0,0) = complex<double>(1.,0.) * F.sign();
////	M[N_sites-1](1,0) = f(N_sites-1) * F.c(UP);
////	
////	Mpo<Symmetry,complex<double> > Mout(N_sites, M, qarray<Symmetry::Nq>(qdiff), HubbardU1xU1::Nlabel, ss.str());
////	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F.get_basis(),l);}
////	return Mout;
////}

////Mpo<Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::U1<Sym::ChargeU1> > > HubbardU1xU1::
////triplon (SPIN_INDEX sigma, size_t locx, size_t locy)
////{
////	assert(locx<N_sites and locy<F[locx].dim());
////	stringstream ss;
////	ss << "triplon(" << locx << ")" << "c(" << locx+1 << ",σ=" << sigma << ")";
////	
////	qarray<2> qdiff;
////	(sigma==UP) ? qdiff = {-2,-1} : qdiff = {-1,-2};
////	
////	vector<SuperMatrix<Symmetry,double> > M(N_sites);
////	for (size_t l=0; l<locx; ++l)
////	{
////		M[l].setMatrix(1,F[l].dim());
////		M[l](0,0) = F[l].sign();
////	}
////	// c(locx,UP)*c(locx,DN)
////	M[locx].setMatrix(1,F[locx].dim());
////	M[locx](0,0) = F[locx].c(UP,locy)*F[locx].c(DN,locy);
////	// c(locx+1,UP|DN)
////	M[locx+1].setMatrix(1,F[locx+1].dim());
////	M[locx+1](0,0) = (sigma==UP)? F[locx+1].c(UP,locy) : F[locx+1].c(DN,locy);
////	for (size_t l=locx+2; l<N_sites; ++l)
////	{
////		M[l].setMatrix(1,F[l].dim());
////		M[l](0,0) = F[l].Id();
////	}
////	
////	Mpo<Symmetry> Mout(N_sites, M, qarray<Symmetry::Nq>(qdiff), HubbardU1xU1::Nlabel, ss.str());
////	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
////	return Mout;
////}

////Mpo<Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::U1<Sym::ChargeU1> > > HubbardU1xU1::
////antitriplon (SPIN_INDEX sigma, size_t locx, size_t locy)
////{
////	assert(locx<N_sites and locy<F[locx].dim());
////	stringstream ss;
////	ss << "antitriplon(" << locx << ")" << "c(" << locx+1 << ",σ=" << sigma << ")";
////	
////	qarray<2> qdiff;
////	(sigma==UP) ? qdiff = {+2,+1} : qdiff = {+1,+2};
////	
////	vector<SuperMatrix<Symmetry,double> > M(N_sites);
////	for (size_t l=0; l<locx; ++l)
////	{
////		M[l].setMatrix(1,F[l].dim());
////		M[l](0,0) = F[l].sign();
////	}
////	// c†(locx,DN)*c†(locx,UP)
////	M[locx].setMatrix(1,F[locx].dim());
////	M[locx](0,0) = F[locx].cdag(DN,locy)*F[locx].cdag(UP,locy);
////	// c†(locx+1,UP|DN)
////	M[locx+1].setMatrix(1,F[locx+1].dim());
////	M[locx+1](0,0) = (sigma==UP)? F[locx+1].cdag(UP,locy) : F[locx+1].cdag(DN,locy);
////	for (size_t l=locx+2; l<N_sites; ++l)
////	{
////		M[l].setMatrix(1,F[l].dim());
////		M[l](0,0) = F[l].Id();
////	}
////	
////	Mpo<Symmetry> Mout(N_sites, M, qarray<Symmetry::Nq>(qdiff), HubbardU1xU1::Nlabel, ss.str());
////	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
////	return Mout;
////}

////Mpo<Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::U1<Sym::ChargeU1> > > HubbardU1xU1::
////quadruplon (size_t locx, size_t locy)
////{
////	assert(locx<N_sites and locy<F[locx].dim());
////	stringstream ss;
////	ss << "Auger(" << locx << ")" << "Auger(" << locx+1 << ")";
////	
////	vector<SuperMatrix<Symmetry,double> > M(N_sites);
////	for (size_t l=0; l<locx; ++l)
////	{
////		M[l].setMatrix(1,F[l].dim());
////		M[l](0,0) = F[l].Id();
////	}
////	// c(loc,UP)*c(loc,DN)
////	M[locx].setMatrix(1,F[locx].dim());
////	M[locx](0,0) = F[locx].c(UP,locy)*F[locx].c(DN,locy);
////	// c(loc+1,UP)*c(loc+1,DN)
////	M[locx+1].setMatrix(1,F[locx+1].dim());
////	M[locx+1](0,0) = F[locx+1].c(UP,locy)*F[locx+1].c(DN,locy);
////	for (size_t l=locx+2; l<N_sites; ++l)
////	{
////		M[l].setMatrix(1,4);
////		M[l](0,0) = F[l].Id();
////	}
////	
////	Mpo<Symmetry> Mout(N_sites, M, qarray<Symmetry::Nq>({-2,-2}), HubbardU1xU1::Nlabel, ss.str());
////	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
////	return Mout;
////}

} // end namespace VMPS::models

#endif

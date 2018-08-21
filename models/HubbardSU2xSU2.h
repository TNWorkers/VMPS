#ifndef HUBBARDMODELSU2XSU2_H_
#define HUBBARDMODELSU2XSU2_H_

#include "symmetry/S1xS2.h"
#include "symmetry/SU2.h"
#include "bases/FermionBaseSU2xSU2.h"
#include "Mpo.h"
//include "DmrgExternal.h"
//include "ParamHandler.h"
#include "ParamReturner.h"

namespace VMPS
{

/** 
 * \class HubbardSU2xSU2
 * \ingroup Hubbard
 *
 * \brief Hubbard Model
 *
 * MPO representation of 
 * 
 * \f[
 * H = -t \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} 
 *     +U \sum_i \left[\left(n_{i\uparrow}-\frac{1}{2}\right)\left(n_{i\downarrow}-\frac{1}{2}\right) -\frac{1}{4}\right]
 *     +V \sum_{<ij>} \mathbf{T}_i \mathbf{T}_j
 *     +J \sum_{<ij>} \mathbf{S}_i \mathbf{S}_j
 * \f]
 * with \f$T^+_i = (-1)^i c^{\dagger}_{i\uparrow} c^{\dagger}_{i\downarrow}\f$, \f$Q^-_i = (T^+_i)^{\dagger}\f$, \f$T^z_i = 0.5(n_{i}-1)\f$
 *
 * \note Makes use of the spin-SU(2) symmetry and the charge-SU(2) symmetry.
 * \warning Bipartite hopping structure is mandatory (particle-hole symmetry)!
 * \warning \f$J>0\f$ is antiferromagnetic
 */
class HubbardSU2xSU2 : public Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > ,double>, public ParamReturner
{
public:
	typedef Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > Symmetry;
	MAKE_TYPEDEFS(HubbardSU2xSU2)
	
private:
	
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	
	typedef SiteOperatorQ<Symmetry,MatrixType> OperatorType;
	
public:
	
	HubbardSU2xSU2() : Mpo(){};
	HubbardSU2xSU2 (const size_t &L, const vector<Param> &params);
	
	static HamiltonianTermsXd<Symmetry> set_operators (const vector<FermionBase<Symmetry> > &F, const ParamHandler &P, size_t loc=0);
	
	Mpo<Symmetry> c (size_t locx, size_t locy=0, double factor=sqrt(2.));
	Mpo<Symmetry> cdag (size_t locx, size_t locy=0, double factor=sqrt(2.));
	
	Mpo<Symmetry> cdagc (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	Mpo<Symmetry> nh (size_t locx, size_t locy=0);
	Mpo<Symmetry> ns (size_t locx, size_t locy=0);
	
	// MpoQ<Symmetry> S (size_t locx, size_t locy=0);
	// MpoQ<Symmetry> Sdag (size_t locx, size_t locy=0);
	// MpoQ<Symmetry> SSdag (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	
	// MpoQ<Symmetry> T (size_t locx, size_t locy=0);
	// MpoQ<Symmetry> Tdag (size_t locx, size_t locy=0);
	// MpoQ<Symmetry> TTdag (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	
	static const map<string,any> defaults;
	static const map<string,any> sweep_defaults;
	
protected:
	
	vector<FermionBase<Symmetry> > F;
};

const map<string,any> HubbardSU2xSU2::defaults = 
{
	{"t",1.}, {"tRung",1.},
	{"U",0.},
	{"V",0.}, {"Vrung",0.},
	{"J",0.}, {"Jrung",0.},
	{"CALC_SQUARE",false}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}
};

const map<string,any> HubbardSU2xSU2::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1e-11}, {"eps_svd",1e-7},
	{"Dincr_abs", 2ul}, {"Dincr_per", 2ul}, {"Dincr_rel", 1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",20ul}, {"min_halfsweeps",6ul},
	{"Dinit",4ul}, {"Qinit",10ul}, {"Dlimit",500ul},
	{"tol_eigval",1e-6}, {"tol_state",1e-5},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_2SITE}
};

HubbardSU2xSU2::
HubbardSU2xSU2 (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, qarray<Symmetry::Nq>({1,1}), "", PROP::HERMITIAN, PROP::NON_UNITARY, PROP::HAMILTONIAN),
 ParamReturner(HubbardSU2xSU2::sweep_defaults)
{
	ParamHandler P(params,defaults);
	
	size_t Lcell = P.size();
	assert(Lcell > 1 and "You need to set a unit cell with at least Lcell=2 for the charge-SU(2) symmetry!");
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	F.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		
		F[l] = (l%2 == 0) ? FermionBase<Symmetry>(P.get<size_t>("Ly",l%Lcell),SUB_LATTICE::A):
		                    FermionBase<Symmetry>(P.get<size_t>("Ly",l%Lcell),SUB_LATTICE::B);
		setLocBasis(F[l].get_basis().qloc(),l);
	}
	
	for (size_t l=0; l<N_sites; ++l)
	{
		Terms[l] = set_operators(F,P,l%Lcell);
		
		stringstream ss;
		ss << "Ly=" << P.get<size_t>("Ly",l%Lcell);
		Terms[l].info.push_back(ss.str());
	}
	
	this->construct_from_Terms(Terms, Lcell, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
	this->precalc_TwoSiteData();
}

HamiltonianTermsXd<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > HubbardSU2xSU2::
set_operators (const vector<FermionBase<Symmetry> > &F, const ParamHandler &P, size_t loc)
{
	HamiltonianTermsXd<Symmetry> Terms;
	
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
			// Only works with both times loc: strange??
			auto cdagF = OperatorType::prod(F[loc].cdag(i), F[loc].sign(), {2,2});
			Terms.tight.push_back(make_tuple(-tPara(i,j)*sqrt(2.)*sqrt(2.), cdagF.plain<double>(), F[loc].c(i).plain<double>()));
		}
		
		// Warning: Needs testing! Especially if correct with [loc] both times.
		
		if (Vpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(Vpara(i,j)*sqrt(3.), F[loc].Tdag(i).plain<double>(), F[loc].T(i).plain<double>()));
		}
		
		if (Jpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(Jpara(i,j)*sqrt(3.), F[loc].Sdag(i).plain<double>(), F[loc].S(i).plain<double>()));
		}
	}
	
	// local terms
	
	// Hubbard-U
	auto [U,Uorb,Ulabel] = P.fill_array1d<double>("U","Uorb",F[loc].orbitals(),loc);
	save_label(Ulabel);
	
	// t⟂
	auto [tRung,tPerp,tPerplabel] = P.fill_array2d<double>("tRung","t","tPerp",F[loc].orbitals(),loc,P.get<bool>("CYLINDER"));
	save_label(tPerplabel);
	
	// V⟂
	auto [Vrung,Vperp,Vperplabel] = P.fill_array2d<double>("Vrung","V","Vperp",F[loc].orbitals(),loc,P.get<bool>("CYLINDER"));
	save_label(Vperplabel);
	
	// J⟂
	auto [Jrung,Jperp,Jperplabel] = P.fill_array2d<double>("Jrung","J","Jperp",F[loc].orbitals(),loc,P.get<bool>("CYLINDER"));
	save_label(Jperplabel);
	
	Terms.local.push_back(make_tuple(1., F[loc].HubbardHamiltonian(Uorb,tPerp,Vperp,Jperp).plain<double>()));
	Terms.name = "Hubbard SU(2)⊗SU(2)";
	
	return Terms;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > HubbardSU2xSU2::
c (size_t locx, size_t locy, double factor)
{
	assert(locx<N_sites and locy<F[locx].dim());
	stringstream ss;
	ss << "c(" << locx << "," << locy << ")";
	
	Mpo<Symmetry> Mout(N_sites, {2,2}, ss.str());
	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	Mout.setLocal(locx, factor*F[locx].c(locy).plain<double>(), F[0].sign().plain<double>());
	return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > HubbardSU2xSU2::
cdag (size_t locx, size_t locy, double factor)
{
	assert(locx<N_sites and locy<F[locx].dim());
	stringstream ss;
	ss << "c†(" << locx << "," << locy << ")";
	
	Mpo<Symmetry> Mout(N_sites, {2,2}, ss.str());
	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	Mout.setLocal(locx, factor*F[locx].cdag(locy).plain<double>(), F[0].sign().plain<double>());
	return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > HubbardSU2xSU2::
cdagc (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
{
	assert(locx1<this->N_sites and locx2<this->N_sites);
	stringstream ss;
	ss << "c†(" << locx1 << "," << locy1 << ")" << "c(" << locx2 << "," << locy2 << ")";
	
	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
	for (size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F[l].get_basis().qloc(),l); }
	
	auto cdag = F[locx1].cdag(locy1);
	auto c    = F[locx2].c   (locy2);
	
	if (locx1 == locx2)
	{
		//The diagonal element is actually 2*unity by the symmetry. But we may leave this as a check.
		Mout.setLocal(locx1, sqrt(2.) * sqrt(2.) * OperatorType::prod(cdag,c,Symmetry::qvacuum()).plain<double>());
	}
	else if (locx1<locx2)
	{
		Mout.setLocal({locx1, locx2}, {sqrt(2.) * sqrt(2.) * OperatorType::prod(cdag, F[locx1].sign(), {2,2}).plain<double>(), 
		                               c.plain<double>()}, 
		                               F[0].sign().plain<double>());
	}
	else if (locx1>locx2)
	{
		Mout.setLocal({locx2, locx1}, {sqrt(2.) * sqrt(2.) * OperatorType::prod(c, F[locx2].sign(), {2,2}).plain<double>(), 
		                               -1. * cdag.plain<double>()}, 
		                               F[0].sign().plain<double>());
	}
	return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > HubbardSU2xSU2::
nh (size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<F[locx].dim());
	stringstream ss;
	ss << "holon_occ(" << locx << "," << locy << ")";
	
	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
	for (size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F[l].get_basis().qloc(),l); }
	
	Mout.setLocal(locx, F[locx].nh(locy).plain<double>());
	return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > HubbardSU2xSU2::
ns (size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<F[locx].dim());
	stringstream ss;
	ss << "spinon_occ(" << locx << "," << locy << ")";
	
	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
	for (size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F[l].get_basis().qloc(),l); }
	
	Mout.setLocal(locx, F[locx].ns(locy).plain<double>());
	return Mout;
}

} //end namespace VMPS
#endif

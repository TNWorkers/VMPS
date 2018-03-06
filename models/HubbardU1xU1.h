#ifndef STRAWBERRY_HUBBARDMODEL
#define STRAWBERRY_HUBBARDMODEL

#include "Mpo.h"
#include "bases/FermionBase.h"
#include "ParamHandler.h" // from HELPERS
#include "models/HubbardObservables.h"
#include "symmetry/S1xS2.h"

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
 * 	H_{tJ} = +J \sum_{<ij>} (\mathbf{S}_{i} \mathbf{S}_{j} - \frac{1}{4} n_in_j)
 * \f]
 * \f[
 * 	H_{3-site} = -\frac{J}{4} \sum_{<ijk>\sigma} (c^\dagger_{i\sigma} n_{j,-\sigma} c_{k\sigma} - c^\dagger_{i\sigma} S^{-\sigma}_j c_{k,-\sigma} + h.c.) \
 * \f]
 * \note Makes use of the U(1) particle conservation symmetry for both spin components seperatly.
 *       You can change this by chossing another symmetry class. To use for example the magnetization and the particle number use:
 * \code{.cpp}
 *     Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::U1<Sym::ChargeU1> >
 * \endcode
 * \note The default variable settings can be seen in \p HubbardU1xU1::defaults.
 * \note If the NNN-hopping is positive, the ground state energy is lowered.
 * \warning \f$J>0\f$ is antiferromagnetic
 */
class HubbardU1xU1 : public Mpo<Sym::S1xS2<Sym::U1<Sym::ChargeUp>,Sym::U1<Sym::ChargeDn> >,double>,
					 public HubbardObservables<Sym::S1xS2<Sym::U1<Sym::ChargeUp>,Sym::U1<Sym::ChargeDn> > >
{
typedef Sym::S1xS2<Sym::U1<Sym::ChargeUp>,Sym::U1<Sym::ChargeDn> > Symmetry;

public:
	
	///@{
	HubbardU1xU1() : Mpo(){};
	HubbardU1xU1 (const size_t &L, const vector<Param> &params);
	///@}
	
	template<typename Symmetry_> 
	static HamiltonianTermsXd<Symmetry_> set_operators (const FermionBase<Symmetry_> &F, const ParamHandler &P, size_t loc=0);
	
	qarray<2> singlet (int N) const {return qarray<2>{N/2,N/2};};
		
	/**Default parameters.*/
	static const std::map<string,std::any> defaults;
};

const std::map<string,std::any> HubbardU1xU1::defaults = 
{
	{"t",1.}, {"tPerp",0.}, {"tPrime",0.}, 
	{"mu",0.}, {"t0",0.}, 
	{"U",0.}, {"V",0.}, {"Vperp",0.}, 
	{"Bz",0.}, 
	{"J",0.}, {"Jperp",0.}, {"J3site",0.},
	{"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}
};

HubbardU1xU1::
HubbardU1xU1 (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", true),
 HubbardObservables(L,params,HubbardU1xU1::defaults)
{
	ParamHandler P(params,HubbardU1xU1::defaults);
	
	size_t Lcell = P.size();
	vector<SuperMatrix<Symmetry,double> > G;
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		
		setLocBasis(F[l].get_basis(),l);
		
		Terms[l] = set_operators(F[l],P,l%Lcell);
		this->Daux = Terms[l].auxdim();
		
		G.push_back(Generator(Terms[l])); // boost::multi_array has stupid assignment
		setOpBasis(G[l].calc_qOp(),l);
	}
	
	this->generate_label(Terms[0].name,Terms,Lcell);
	this->construct(G, this->W, this->Gvec, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
	this->Terms = Terms;
}

template<typename Symmetry_>
HamiltonianTermsXd<Symmetry_> HubbardU1xU1::
set_operators (const FermionBase<Symmetry_> &F, const ParamHandler &P, size_t loc)
{
	HamiltonianTermsXd<Symmetry_> Terms;
	
	auto save_label = [&Terms] (string label)
	{
		if (label!="") {Terms.info.push_back(label);}
	};
	
	// NN terms
	
	auto [t,tPara,tlabel] = P.fill_array2d<double>("t","tPara",F.orbitals(),loc);
	save_label(tlabel);
	
	auto [V,Vpara,Vlabel] = P.fill_array2d<double>("V","Vpara",F.orbitals(),loc);
	save_label(Vlabel);
	
	auto [J,Jpara,Jlabel] = P.fill_array2d<double>("J","Jpara",F.orbitals(),loc);
	save_label(Jlabel);
	
	for (int i=0; i<F.orbitals(); ++i)
	for (int j=0; j<F.orbitals(); ++j)
	{
		if (tPara(i,j) != 0.)
		{
			// wrong:
//			Terms.tight.push_back(make_tuple(-tPara(i,j), F.cdag(UP,i), F.sign() * F.c(UP,j)));
//			Terms.tight.push_back(make_tuple(-tPara(i,j), F.cdag(DN,i), F.sign() * F.c(DN,j)));
//			Terms.tight.push_back(make_tuple(+tPara(i,j), F.c(UP,i),    F.sign() * F.cdag(UP,j)));
//			Terms.tight.push_back(make_tuple(+tPara(i,j), F.c(DN,i),    F.sign() * F.cdag(DN,j)));
			
			// correct:
			Terms.tight.push_back(make_tuple(-tPara(i,j), F.cdag(UP,i)  * F.sign(), F.c(UP,j)));
			Terms.tight.push_back(make_tuple(-tPara(i,j), F.cdag(DN,i)  * F.sign(), F.c(DN,j)));
			Terms.tight.push_back(make_tuple(-tPara(i,j), -1.*F.c(UP,i) * F.sign(), F.cdag(UP,j)));
			Terms.tight.push_back(make_tuple(-tPara(i,j), -1.*F.c(DN,i) * F.sign(), F.cdag(DN,j)));
		}
		
		if (Vpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(Vpara(i,j), F.n(i), F.n(j)));
		}
		
		if (Jpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(0.5*Jpara(i,j), F.Sp(i), F.Sm(i)));
			Terms.tight.push_back(make_tuple(0.5*Jpara(i,j), F.Sm(i), F.Sp(i)));
			Terms.tight.push_back(make_tuple(Jpara(i,j),     F.Sz(i), F.Sz(i)));
		}
	}
	
	// NNN terms
	
	param0d tPrime = P.fill_array0d<double>("tPrime","tPrime",loc);
	save_label(tPrime.label);
	
	if (tPrime.x != 0.)
	{
		assert(F.orbitals() == 1 and "Cannot do a ladder with t'!");
		
		Terms.nextn.push_back(make_tuple(-tPrime.x, F.cdag(UP)  * F.sign(), F.c(UP),    F.sign()));
		Terms.nextn.push_back(make_tuple(-tPrime.x, F.cdag(DN)  * F.sign(), F.c(DN),    F.sign()));
		Terms.nextn.push_back(make_tuple(-tPrime.x, -1.*F.c(UP) * F.sign(), F.cdag(UP), F.sign()));
		Terms.nextn.push_back(make_tuple(-tPrime.x, -1.*F.c(DN) * F.sign(), F.cdag(DN), F.sign()));
	}
	
	param0d J3site = P.fill_array0d<double>("J3site","J3site",loc);
	save_label(J3site.label);
	
	if (J3site.x != 0.)
	{
		lout << "Warning! J3site has to be tested against ED!" << endl;
		
		assert(F.orbitals() == 1 and "Cannot do a ladder with 3-site J terms!");
		
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
		Terms.nextn.push_back(make_tuple(-0.25*J3site.x, F.cdag(UP)  * F.sign(), F.c(UP),    F.n(DN)*F.sign()));
		Terms.nextn.push_back(make_tuple(-0.25*J3site.x, F.cdag(DN)  * F.sign(), F.c(DN),    F.n(UP)*F.sign()));
		Terms.nextn.push_back(make_tuple(-0.25*J3site.x, -1.*F.c(UP) * F.sign(), F.cdag(UP), F.n(DN)*F.sign()));
		Terms.nextn.push_back(make_tuple(-0.25*J3site.x, -1.*F.c(DN) * F.sign(), F.cdag(DN), F.n(UP)*F.sign()));
		
		// three-site terms with spinflip
		Terms.nextn.push_back(make_tuple(+0.25*J3site.x, F.cdag(DN)  * F.sign(), F.c(UP),    F.Sp()*F.sign()));
		Terms.nextn.push_back(make_tuple(+0.25*J3site.x, F.cdag(UP)  * F.sign(), F.c(DN),    F.Sm()*F.sign()));
		Terms.nextn.push_back(make_tuple(+0.25*J3site.x, -1.*F.c(DN) * F.sign(), F.cdag(UP), F.Sm()*F.sign()));
		Terms.nextn.push_back(make_tuple(+0.25*J3site.x, -1.*F.c(UP) * F.sign(), F.cdag(DN), F.Sp()*F.sign()));
	}
	
	// local terms
	
	// Hubbard-U
	auto [U,Uorb,Ulabel] = P.fill_array1d<double>("U","Uorb",F.orbitals(),loc);
	save_label(Ulabel);
	
	// t0
	auto [t0,t0orb,t0label] = P.fill_array1d<double>("t0","t0orb",F.orbitals(),loc);
	save_label(t0label);
	
	// μ
	auto [mu,muorb,mulabel] = P.fill_array1d<double>("mu","muorb",F.orbitals(),loc);
	save_label(mulabel);
	
	// Bz
	auto [Bz,Bzorb,Bzlabel] = P.fill_array1d<double>("Bz","Bzorb",F.orbitals(),loc);
	save_label(Bzlabel);
	
	// t⟂
	param0d tPerp = P.fill_array0d<double>("t","tPerp",loc);
	save_label(tPerp.label);
	
	// V⟂
	param0d Vperp = P.fill_array0d<double>("V","Vperp",loc);
	save_label(Vperp.label);
	
	// J⟂
	param0d Jperp = P.fill_array0d<double>("J","Jperp",loc);
	save_label(Jperp.label);
	
	if (isfinite(Uorb.sum()))
	{
		Terms.name = "Hubbard";
	}
	else
	{
		Terms.name = (P.HAS_ANY_OF({"J","J3site"}))? "t-J":"U=∞-Hubbard";
	}
	
	Terms.local.push_back(make_tuple(1., F.HubbardHamiltonian(Uorb,t0orb-muorb,Bzorb,F.ZeroField(),tPerp.x,Vperp.x,Jperp.x, P.get<bool>("CYLINDER"))));
	
	return Terms;
}

//Mpo<Sym::S1xS2<Sym::U1<Sym::ChargeUp>,Sym::U1<Sym::ChargeDn> > > HubbardU1xU1::
//cc (size_t locx, size_t locy)
//{
//	assert(locx<N_sites and locy<F[locx].dim());
//	stringstream ss;
//	ss << "c(" << locx << "," << locy << "," << UP << ")"
//	   << "c(" << locx << "," << locy << "," << DN << ")";
//	
//	Mpo<Symmetry> Mout(N_sites, qarray<Symmetry::Nq>({-1,-1}), HubbardU1xU1::Nlabel, ss.str());
//	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
//	
//	Mout.setLocal(locx, F[locx].c(UP,locy) * F[locx].c(DN,locy));
//	return Mout;
//}

//Mpo<Sym::S1xS2<Sym::U1<Sym::ChargeUp>,Sym::U1<Sym::ChargeDn> > > HubbardU1xU1::
//eta()
//{
//	assert(N_sites == N_phys);
//	stringstream ss;
//	ss << "eta";
//	
//	Mpo<Symmetry> Mout(N_sites, qarray<Symmetry::Nq>({-1,-1}), HubbardU1xU1::Nlabel, ss.str());
//	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
//	
//	Mout.setLocalSum(F[0].c(UP)*F[0].c(DN), stagger);
//	return Mout;
//}

//Mpo<Sym::S1xS2<Sym::U1<Sym::ChargeUp>,Sym::U1<Sym::ChargeDn> > > HubbardU1xU1::
//cdagcdag (size_t locx, size_t locy)
//{
//	assert(locx<N_sites and locy<F[locx].dim());
//	stringstream ss;
//	ss << "c†(" << locx << "," << locy << "," << DN << ")"
//	   << "c†(" << locx << "," << locy << "," << UP << ")";
//	
//	Mpo<Symmetry> Mout(N_sites, qarray<Symmetry::Nq>({+1,+1}), HubbardU1xU1::Nlabel, ss.str());
//	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
//	
//	Mout.setLocal(locx, F[locx].cdag(DN,locy) * F[locx].cdag(UP,locy));
//	return Mout;
//}

//Mpo<Sym::S1xS2<Sym::U1<Sym::ChargeUp>,Sym::U1<Sym::ChargeDn> > > HubbardU1xU1::
//c (SPIN_INDEX sigma, size_t locx, size_t locy)
//{
//	assert(locx<N_sites and locy<F[locx].dim());
//	stringstream ss;
//	ss << "c(" << locx << "," << locy << "," << sigma << ")";
//	
//	qarray<2> qdiff;
//	(sigma==UP) ? qdiff={-1,0} : qdiff={0,-1};
//	
//	Mpo<Symmetry> Mout(N_sites, qdiff, HubbardU1xU1::Nlabel, ss.str());
//	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
//	
//	Mout.setLocal(locx, F[locx].c(sigma,locy), F[0].sign());
//	
//	return Mout;
//}

//Mpo<Sym::S1xS2<Sym::U1<Sym::ChargeUp>,Sym::U1<Sym::ChargeDn> > > HubbardU1xU1::
//cdag (SPIN_INDEX sigma, size_t locx, size_t locy)
//{
//	assert(locx<N_sites and locy<F[locx].dim());
//	stringstream ss;
//	ss << "c†(" << locx << "," << locy << "," << sigma << ")";
//	
//	qarray<2> qdiff; 
//	(sigma==UP) ? qdiff={+1,0} : qdiff={0,+1};
//	
//	Mpo<Symmetry> Mout(N_sites, qdiff, HubbardU1xU1::Nlabel, ss.str());
//	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
//	
//	Mout.setLocal(locx, F[locx].cdag(sigma,locy), F[0].sign());
//	return Mout;
//}

//Mpo<Sym::S1xS2<Sym::U1<Sym::ChargeUp>,Sym::U1<Sym::ChargeDn> > > HubbardU1xU1::
//cdagc (SPIN_INDEX sigma, size_t locx1, size_t locx2, size_t locy1, size_t locy2)
//{
//	assert(locx1<N_sites and locx2<N_sites);
//	stringstream ss;
//	ss << "c†(" << locx1 << "," << locy1 << "," << sigma << ")" 
//	   << "c (" << locx2 << "," << locy2 << "," << sigma << ")";
//	
//	auto cdag = F[locx1].cdag(sigma,locy1);
//	auto c    = F[locx2].c   (sigma,locy2);
//	
//	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), HubbardU1xU1::Nlabel, ss.str());
//	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
//	
//	if (locx1 == locx2)
//	{
//		Mout.setLocal(locx1, cdag*c);
//	}
//	else if (locx1<locx2)
//	{
//		Mout.setLocal({locx1, locx2}, {cdag*F[locx1].sign(), c}, F[0].sign());
//	}
//	else if (locx1>locx2)
//	{
//		Mout.setLocal({locx2, locx1}, {c*F[locx2].sign(), -1.*cdag}, F[0].sign());
//	}
//	
//	return Mout;
//}

//Mpo<Sym::S1xS2<Sym::U1<Sym::ChargeUp>,Sym::U1<Sym::ChargeDn> > > HubbardU1xU1::
//d (size_t locx, size_t locy)
//{
//	assert(locx<N_sites and locy<F[locx].dim());
//	stringstream ss;
//	ss << "double_occ(" << locx << "," << locy << ")";
//	
//	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), HubbardU1xU1::Nlabel, ss.str());
//	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
//	
//	Mout.setLocal(locx, F[locx].d(locy));
//	return Mout;
//}

//Mpo<Sym::S1xS2<Sym::U1<Sym::ChargeUp>,Sym::U1<Sym::ChargeDn> > > HubbardU1xU1::
//dtot()
//{
//	stringstream ss;
//	ss << "double_occ_total";
//	
//	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), HubbardU1xU1::Nlabel, ss.str());
//	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
//	
//	Mout.setLocalSum(F[0].d());
//	return Mout;
//}

//Mpo<Sym::S1xS2<Sym::U1<Sym::ChargeUp>,Sym::U1<Sym::ChargeDn> > > HubbardU1xU1::
//n (SPIN_INDEX sigma, size_t locx, size_t locy)
//{
//	assert(locx<N_sites and locy<F[locx].dim());
//	stringstream ss;
//	ss << "n(" << locx << "," << locy << "," << sigma << ")";
//	
//	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), HubbardU1xU1::Nlabel, ss.str());
//	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
//	
//	Mout.setLocal(locx, F[locx].n(sigma,locy));
//	return Mout;
//}

//Mpo<Sym::S1xS2<Sym::U1<Sym::ChargeUp>,Sym::U1<Sym::ChargeDn> > > HubbardU1xU1::
//nn (SPIN_INDEX sigma1, size_t locx1, SPIN_INDEX sigma2, size_t locx2, size_t locy1, size_t locy2)
//{
//	assert(locx1<N_sites and locx2<N_sites and locy1<F[locx1].dim() and locy2<F[locx2].dim());
//	stringstream ss;
//	ss << "n(" << locx1 << "," << locy1 << ")"
//	   << "n(" << locx2 << "," << locy2 << ")";
//	
//	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), HubbardU1xU1::Nlabel, ss.str());
//	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
//	
//	Mout.setLocal({locx1,locx2}, {F[locx1].n(sigma1,locy1),F[locx2].n(sigma2,locy2)});
//	return Mout;
//}

//Mpo<Sym::S1xS2<Sym::U1<Sym::ChargeUp>,Sym::U1<Sym::ChargeDn> > > HubbardU1xU1::
//Sz (size_t locx, size_t locy)
//{
//	assert(locx<N_sites and locy<F[locx].dim());
//	stringstream ss;
//	ss << "Sz(" << locx << "," << locy << ")";
//	
//	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), HubbardU1xU1::Nlabel, ss.str());
//	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
//	
//	Mout.setLocal(locx, F[locx].Sz(locy));
//	return Mout;
//}

//Mpo<Sym::S1xS2<Sym::U1<Sym::ChargeUp>,Sym::U1<Sym::ChargeDn> > > HubbardU1xU1::
//SzSz (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
//{
//	assert(locx1 < N_sites and locx2 < N_sites and locy1<F[locx1].dim() and locy2<F[locx2].dim());
//	stringstream ss;
//	ss << "Sz(" << locx1 << "," << locy1 << ")" 
//	   << "Sz(" << locx2 << "," << locy2 << ")";
//	
//	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), HubbardU1xU1::Nlabel, ss.str());
//	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
//	
//	Mout.setLocal({locx1,locx2}, {F[locx1].Sz(locy1),F[locx2].Sz(locy2)});
//	return Mout;
//}

//Mpo<Sym::S1xS2<Sym::U1<Sym::ChargeUp>,Sym::U1<Sym::ChargeDn> > > HubbardU1xU1::
//SaSa (size_t locx1, SPINOP_LABEL SOP1, size_t locx2, SPINOP_LABEL SOP2, size_t locy1, size_t locy2)
//{
//	assert(locx1<N_sites and locx2<N_sites and locy1<F[locx1].dim() and locy2<F[locx2].dim());
//	stringstream ss;
//	ss << SOP1 << "(" << locx1 << "," << locy1 << ")" 
//	   << SOP2 << "(" << locx2 << "," << locy2 << ")";
//	
//	Mpo<Symmetry> Mout(N_sites, qarray<Symmetry::Nq>(F[locx1].getQ(SOP1)+F[locx2].getQ(SOP2)), HubbardU1xU1::Nlabel, ss.str());
//	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
//	
//	Mout.setLocal({locx1,locx2}, {F[locx1].Scomp(SOP1,locy1),F[locx2].Scomp(SOP2,locy2)});
//	return Mout;
//}

//Mpo<Sym::S1xS2<Sym::U1<Sym::ChargeUp>,Sym::U1<Sym::ChargeDn> > > HubbardU1xU1::
//s (size_t locx, size_t locy)
//{
//	assert(locx<N_sites and locy<F[locx].dim());
//	stringstream ss;
//	ss << "single_occ(" << locx << "," << locy << ")";
//	
//	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), HubbardU1xU1::Nlabel, ss.str());
//	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
//	
//	Mout.setLocal(locx, F[locx].n(UP,locy)+F[locx].n(DN,locy)-2.*F[locx].d(locy));
//	return Mout;
//}

//Mpo<Sym::S1xS2<Sym::U1<Sym::ChargeUp>,Sym::U1<Sym::ChargeDn> > > HubbardU1xU1::
//hh (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
//{
//	assert(locx1 < N_sites and locx2 < N_sites and locy1<F[locx1].dim() and locy2<F[locx2].dim());
//	stringstream ss;
//	ss << "h(" << locx1 << "," << locy1 << ")"
//	   << "h(" << locx2 << "," << locy2 << ")";
//	
//	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), HubbardU1xU1::Nlabel, ss.str());
//	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
//	
//	Mout.setLocal({locx1,locx2}, {F[locx1].d(locy1)-F[locx1].n(locy1)+F[locx1].Id(),
//	                              F[locx2].d(locy2)-F[locx2].n(locy2)+F[locx2].Id()});
//	return Mout;
//}

////Mpo<Sym::S1xS2<Sym::U1<Sym::ChargeUp>,Sym::U1<Sym::ChargeDn> >,complex<double> > HubbardU1xU1::
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

////Mpo<Sym::S1xS2<Sym::U1<Sym::ChargeUp>,Sym::U1<Sym::ChargeDn> >,complex<double> > HubbardU1xU1::
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

////Mpo<Sym::S1xS2<Sym::U1<Sym::ChargeUp>,Sym::U1<Sym::ChargeDn> >,complex<double> > HubbardU1xU1::
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

////Mpo<Sym::S1xS2<Sym::U1<Sym::ChargeUp>,Sym::U1<Sym::ChargeDn> > > HubbardU1xU1::
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

////Mpo<Sym::S1xS2<Sym::U1<Sym::ChargeUp>,Sym::U1<Sym::ChargeDn> > > HubbardU1xU1::
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

////Mpo<Sym::S1xS2<Sym::U1<Sym::ChargeUp>,Sym::U1<Sym::ChargeDn> > > HubbardU1xU1::
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

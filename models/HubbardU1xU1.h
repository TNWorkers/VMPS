#ifndef STRAWBERRY_HUBBARDMODEL
#define STRAWBERRY_HUBBARDMODEL

#include "bases/FermionBase.h"
#include "symmetry/S1xS2.h"
#include "Mpo.h"
#include "ParamHandler.h" // from HELPERS
#include "models/HubbardObservables.h"

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
class HubbardU1xU1 : public Mpo<Sym::S1xS2<Sym::U1<Sym::ChargeUp>,Sym::U1<Sym::ChargeDn> >,double>,
                     public HubbardObservables<Sym::S1xS2<Sym::U1<Sym::ChargeUp>,Sym::U1<Sym::ChargeDn> > >
{
public:
	
	typedef Sym::S1xS2<Sym::U1<Sym::ChargeUp>,Sym::U1<Sym::ChargeDn> > Symmetry;
	
	///@{
	HubbardU1xU1() : Mpo(){};
	HubbardU1xU1 (const size_t &L, const vector<Param> &params);
	///@}
	
	static qarray<2> singlet (int N) {return qarray<2>{N/2,N/2};};
	
	template<typename Symmetry_> 
	static HamiltonianTermsXd<Symmetry_> set_operators (const vector<FermionBase<Symmetry_> > &F, const ParamHandler &P, size_t loc=0);
		
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
:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", true),
 HubbardObservables(L,params,HubbardU1xU1::defaults)
{
	ParamHandler P(params,HubbardU1xU1::defaults);
	
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
	}
	
	this->construct_from_Terms(Terms, Lcell, false, P.get<bool>("OPEN_BC"));
}

template<typename Symmetry_>
HamiltonianTermsXd<Symmetry_> HubbardU1xU1::
set_operators (const vector<FermionBase<Symmetry_> > &F, const ParamHandler &P, size_t loc)
{
	HamiltonianTermsXd<Symmetry_> Terms;
	
	auto save_label = [&Terms] (string label)
	{
		if (label!="") {Terms.info.push_back(label);}
	};
	
	// NN terms
	
	auto [t,tPara,tlabel] = P.fill_array2d<double>("t","tPara",F[loc].orbitals(),loc);
	save_label(tlabel);
	
	auto [V,Vpara,Vlabel] = P.fill_array2d<double>("V","Vpara",F[loc].orbitals(),loc);
	save_label(Vlabel);
	
	auto [J,Jpara,Jlabel] = P.fill_array2d<double>("J","Jpara",F[loc].orbitals(),loc);
	save_label(Jlabel);
	
	size_t lp1 = (loc+1)%F.size();
	size_t lp2 = (loc+2)%F.size();
	
	for (int i=0; i<F[loc].orbitals(); ++i)
	for (int j=0; j<F[lp1].orbitals(); ++j)
	{
		if (tPara(i,j) != 0.)
		{
			// wrong:
//			Terms.tight.push_back(make_tuple(-tPara(i,j), F.cdag(UP,i), F.sign() * F.c(UP,j)));
//			Terms.tight.push_back(make_tuple(-tPara(i,j), F.cdag(DN,i), F.sign() * F.c(DN,j)));
//			Terms.tight.push_back(make_tuple(+tPara(i,j), F.c(UP,i),    F.sign() * F.cdag(UP,j)));
//			Terms.tight.push_back(make_tuple(+tPara(i,j), F.c(DN,i),    F.sign() * F.cdag(DN,j)));
			
			// correct:
			Terms.tight.push_back(make_tuple(-tPara(i,j), F[loc].cdag(UP,i)  * F[loc].sign(), F[lp1].c(UP,j)));
			Terms.tight.push_back(make_tuple(-tPara(i,j), F[loc].cdag(DN,i)  * F[loc].sign(), F[lp1].c(DN,j)));
			Terms.tight.push_back(make_tuple(-tPara(i,j), -1.*F[loc].c(UP,i) * F[loc].sign(), F[lp1].cdag(UP,j)));
			Terms.tight.push_back(make_tuple(-tPara(i,j), -1.*F[loc].c(DN,i) * F[loc].sign(), F[lp1].cdag(DN,j)));
		}
		
		if (Vpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(Vpara(i,j), F[loc].n(i), F[lp1].n(j)));
		}
		
		if (Jpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(0.5*Jpara(i,j), F[loc].Sp(i), F[lp1].Sm(j)));
			Terms.tight.push_back(make_tuple(0.5*Jpara(i,j), F[loc].Sm(i), F[lp1].Sp(j)));
			Terms.tight.push_back(make_tuple(Jpara(i,j),     F[loc].Sz(i), F[lp1].Sz(j)));
		}
	}
	
	// NNN terms
	
	param0d tPrime = P.fill_array0d<double>("tPrime","tPrime",loc);
	save_label(tPrime.label);
	
	if (tPrime.x != 0.)
	{
		assert(F[loc].orbitals() == 1 and "Cannot do a ladder with t'!");
		
		Terms.nextn.push_back(make_tuple(-tPrime.x, F[loc].cdag(UP)  * F[loc].sign(), F[lp2].c(UP),    F[lp1].sign()));
		Terms.nextn.push_back(make_tuple(-tPrime.x, F[loc].cdag(DN)  * F[loc].sign(), F[lp2].c(DN),    F[lp1].sign()));
		Terms.nextn.push_back(make_tuple(-tPrime.x, -1.*F[loc].c(UP) * F[loc].sign(), F[lp2].cdag(UP), F[lp1].sign()));
		Terms.nextn.push_back(make_tuple(-tPrime.x, -1.*F[loc].c(DN) * F[loc].sign(), F[lp2].cdag(DN), F[lp1].sign()));
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
		Terms.nextn.push_back(make_tuple(-0.25*J3site.x, F[loc].cdag(UP)  * F[loc].sign(), F[lp2].c(UP),    F[lp1].n(DN)*F[lp1].sign()));
		Terms.nextn.push_back(make_tuple(-0.25*J3site.x, F[loc].cdag(DN)  * F[loc].sign(), F[lp2].c(DN),    F[lp1].n(UP)*F[lp1].sign()));
		Terms.nextn.push_back(make_tuple(-0.25*J3site.x, -1.*F[loc].c(UP) * F[loc].sign(), F[lp2].cdag(UP), F[lp1].n(DN)*F[lp1].sign()));
		Terms.nextn.push_back(make_tuple(-0.25*J3site.x, -1.*F[loc].c(DN) * F[loc].sign(), F[lp2].cdag(DN), F[lp1].n(UP)*F[lp1].sign()));
		
		// three-site terms with spinflip
		Terms.nextn.push_back(make_tuple(+0.25*J3site.x, F[loc].cdag(DN)  * F[loc].sign(), F[lp2].c(UP),    F[lp1].Sp()*F[lp1].sign()));
		Terms.nextn.push_back(make_tuple(+0.25*J3site.x, F[loc].cdag(UP)  * F[loc].sign(), F[lp2].c(DN),    F[lp1].Sm()*F[lp1].sign()));
		Terms.nextn.push_back(make_tuple(+0.25*J3site.x, -1.*F[loc].c(DN) * F[loc].sign(), F[lp2].cdag(UP), F[lp1].Sm()*F[lp1].sign()));
		Terms.nextn.push_back(make_tuple(+0.25*J3site.x, -1.*F[loc].c(UP) * F[loc].sign(), F[lp2].cdag(DN), F[lp1].Sp()*F[lp1].sign()));
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
}

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

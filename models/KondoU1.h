#ifndef STRAWBERRY_TRANSVERSEKONDOMODEL
#define STRAWBERRY_TRANSVERSEKONDOMODEL

#include "models/KondoU1xU1.h"
#include "symmetry/U1.h"

namespace VMPS
{
/** \class KondoU1
 * \ingroup Kondo
 *
 * \brief Kondo Model
 *
 * MPO representation of 
 \f[
 H = -t\sum_{<ij>\sigma} \left(c^\dagger_{i\sigma}c_{j\sigma} + h.c.\right)
 - J \sum_{i \in I} \mathbf{S}_i \cdot \mathbf{s}_i
 - \sum_{i \in I} B_i^x S_i^x
 - \sum_{i \in I} B_i^z S_i^z
 \f]
 *
 where further parameters from Hubbard and Heisenberg are possible.
  \param D : \f$D=2S+1\f$ where \f$S\f$ is the spin of the impurity.

 \note Take use of the U(1) particle conservation symmetry.
 \note The \f$S_z\f$ U(1) symmetry is borken due to the field in x-direction.
 \note The default variable settings can be seen in \p KondoU1::defaults.
 \note \f$J<0\f$ is antiferromagnetic
 \note If nnn-hopping is positive, the GS-energy is lowered.
 \note The multi-impurity model can be received, by setting D=1 (S=0) for all sites without an impurity.
*/
class KondoU1 : public MpoQ<Sym::U1<double>,double>
{
public:
	typedef Sym::U1<double> Symmetry;
	
private:
	typedef typename Symmetry::qType qType;
	
public:
	
	///@{
	KondoU1 () : MpoQ(){};
	KondoU1 (const size_t &L, const vector<Param> &params);
	///@}
	
	/**Labels the conserved quantum number as "N".*/
	static const std::array<string,1> Nlabel;
	
	template<typename Symmetry_>
	static void add_operators (HamiltonianTermsXd<Symmetry_> &Terms, const SpinBase<Symmetry_> &B, 
	                           const FermionBase<Symmetry_> &F, const ParamHandler &P, size_t loc=0);

	///@{
	MpoQ<Symmetry> c (SPIN_INDEX sigma, size_t locx, size_t locy=0);
	MpoQ<Symmetry> cdag (SPIN_INDEX sigma, size_t locx, size_t locy=0);
	///@}
	
	///@{
	MpoQ<Symmetry> n (size_t locx, size_t locy=0);
	MpoQ<Symmetry> d (size_t locx, size_t locy=0);
	MpoQ<Symmetry> cdagc (SPIN_INDEX sigma, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	///@}

	///@{
	MpoQ<Symmetry> Simp (size_t locx, SPINOP_LABEL SOP, size_t locy=0);
	MpoQ<Symmetry> Ssub (size_t locx, SPINOP_LABEL SOP, size_t locy=0);
	///@}

	///@{
	MpoQ<Symmetry> SimpSimp (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y=0, size_t loc2y=0);
	MpoQ<Symmetry> SsubSsub (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y=0, size_t loc2y=0);
	MpoQ<Symmetry> SimpSsub (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y=0, size_t loc2y=0);
	///@}

	///@{ not implemented
	MpoQ<Symmetry> SimpSsubSimpSimp (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, 
	                          size_t loc3x, SPINOP_LABEL SOP3, size_t loc4x, SPINOP_LABEL SOP4,
	                          size_t loc1y=0, size_t loc2y=0, size_t loc3y=0, size_t loc4y=0);
	MpoQ<Symmetry> SimpSsubSimpSsub (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, 
	                          size_t loc3x, SPINOP_LABEL SOP3, size_t loc4x, SPINOP_LABEL SOP4,
	                          size_t loc1y=0, size_t loc2y=0, size_t loc3y=0, size_t loc4y=0);	
	///@}

	/**Validates whether a given \p qnum is a valid combination of \p N and \p M for the given model.
	\returns \p true if valid, \p false if not*/
	bool validate (qType qnum) const;

	static const std::map<string,std::any> defaults;
	
protected:
	
	vector<FermionBase<Symmetry> > F;
	vector<SpinBase<Symmetry> > B;
};

const std::map<string,std::any> KondoU1::defaults = 
{
	{"t",1.}, {"tPerp",0.}, {"tPrime",0.},
	{"J",-1.}, 
	{"U",0.}, {"V",0.}, {"Vperp",0.}, 
	{"mu",0.}, {"t0",0.},
	{"Bz",0.}, {"Bx",0.}, {"Bzsub",0.}, {"Bxsub",0.}, {"Kz",0.}, {"Kx",0.},
	{"D",2ul},
	{"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}
};

const std::array<string,1> KondoU1::Nlabel{"N"};

KondoU1::
KondoU1 (const size_t &L, const vector<Param> &params)
:MpoQ<Symmetry> (L, qarray<Symmetry::Nq>({0}), KondoU1::Nlabel, "")
{
	ParamHandler P(params,defaults);
	
	size_t Lcell = P.size();
	vector<SuperMatrix<Symmetry,double> > G;
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	B.resize(N_sites); F.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		
		F[l] = FermionBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), !isfinite(P.get<double>("U",l%Lcell)));
		B[l] = SpinBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), P.get<size_t>("D",l%Lcell), true); //true means N is good quantum number
		setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l);
		
		Terms[l] = KondoU1xU1::set_operators(B[l],F[l],P,l%Lcell);
		add_operators(Terms[l],B[l],F[l],P,l%Lcell);
		this->Daux = Terms[l].auxdim();
		
		G.push_back(Generator(Terms[l]));
		setOpBasis(G[l].calc_qOp(),l);
	}
	
	this->generate_label(Terms[0].name,Terms,Lcell);
	this->construct(G, this->W, this->Gvec, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
}

MpoQ<Sym::U1<double> > KondoU1::
c (SPIN_INDEX sigma, size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<F[locx].dim());
	stringstream ss;
	ss << "c(" << locx << "," << locy << "," << sigma << ")";
	
	qarray<1> qdiff={-1};

	MpoQ<Symmetry> Mout(N_sites, qdiff, KondoU1::Nlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l); }
	
	Mout.setLocal(locx, kroneckerProduct(B[locx].Id(),F[locx].c(sigma,locy)), kroneckerProduct(B[locx].Id(),F[0].sign()));
	
	return Mout;
}

MpoQ<Sym::U1<double> > KondoU1::
cdag (SPIN_INDEX sigma, size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<F[locx].dim());
	stringstream ss;
	ss << "cdag(" << locx << "," << locy << "," << sigma << ")";
	
	qarray<1> qdiff={+1};

	MpoQ<Symmetry> Mout(N_sites, qdiff, KondoU1::Nlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l); }
	
	Mout.setLocal(locx, kroneckerProduct(B[locx].Id(),F[locx].cdag(sigma,locy)), kroneckerProduct(B[locx].Id(),F[0].sign()));
	
	return Mout;
}

MpoQ<Sym::U1<double> > KondoU1::
n (std::size_t locx, std::size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	std::stringstream ss;
	ss << "occ(" << locx << "," << locy << ")";

	MpoQ<Symmetry> Mout(N_sites, Symmetry::qvacuum(), KondoU1::Nlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l); }

	auto n = kroneckerProduct(B[locx].Id(),F[locx].n(locy));
	Mout.setLocal(locx, n);

	return Mout;	
}

MpoQ<Sym::U1<double> > KondoU1::
d (std::size_t locx, std::size_t locy)
{
	assert(locx<N_sites and locy<F[locx].dim());
	stringstream ss;
	ss << "double_occ(" << locx << "," << locy << ")";
	
	MpoQ<Symmetry> Mout(N_sites, Symmetry::qvacuum(), KondoU1::Nlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l); }

	auto d = kroneckerProduct(B[locx].Id(),F[locx].d(locy));
	Mout.setLocal(locx, d);
	return Mout;
}

MpoQ<Sym::U1<double> > KondoU1::
Simp (size_t locx, SPINOP_LABEL Sa, size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << Sa << "_imp(" << locx << "," << locy << ")";

	MpoQ<Symmetry> Mout(N_sites, B[locx].getQ(Sa), KondoU1::Nlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l); }

	Mout.setLocal(locx, kroneckerProduct(B[locx].Scomp(Sa,locy),F[locx].Id()));
	return Mout;
}

MpoQ<Sym::U1<double> > KondoU1::
Ssub (size_t locx, SPINOP_LABEL Sa, size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << Sa << "_sub(" << locx << "," << locy << ")";

	MpoQ<Symmetry> Mout(N_sites, B[locx].getQ(Sa), KondoU1::Nlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l); }

	Mout.setLocal(locx, kroneckerProduct(B[locx].Id(), F[locx].Scomp(Sa,locy)));
	return Mout;
}

MpoQ<Sym::U1<double> > KondoU1::
SimpSimp (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y, size_t loc2y)
{
	assert(loc1x<this->N_sites and loc2x<this->N_sites);
	std::stringstream ss;
	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")";

	MpoQ<Symmetry> Mout(N_sites, Symmetry::qvacuum(), KondoU1::Nlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l); }

	Mout.setLocal({loc1x, loc2x}, {kroneckerProduct(B[loc1x].Scomp(SOP1,loc1y),F[loc1x].Id()), kroneckerProduct(B[loc2x].Scomp(SOP2,loc2y),F[loc2x].Id())});
	return Mout;
}

MpoQ<Sym::U1<double> > KondoU1::
SsubSsub (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y, size_t loc2y)
{
	assert(loc1x<this->N_sites and loc2x<this->N_sites);
	std::stringstream ss;
	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")";

	MpoQ<Symmetry> Mout(N_sites, Symmetry::qvacuum(), KondoU1::Nlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l); }

	Mout.setLocal({loc1x, loc2x}, {kroneckerProduct(B[loc1x].Id(),F[loc1x].Scomp(SOP1,loc1y)), kroneckerProduct(B[loc2x].Id(),F[loc2x].Scomp(SOP2,loc2y))});
	return Mout;
}

MpoQ<Sym::U1<double> > KondoU1::
SimpSsub (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y, size_t loc2y)
{
	assert(loc1x<this->N_sites and loc2x<this->N_sites);
	std::stringstream ss;
	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")";

	MpoQ<Symmetry> Mout(N_sites, Symmetry::qvacuum(), KondoU1::Nlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l); }

	Mout.setLocal({loc1x, loc2x}, {kroneckerProduct(B[loc1x].Scomp(SOP1,loc1y),F[loc1x].Id()), kroneckerProduct(B[loc2x].Id(),F[loc2x].Scomp(SOP2,loc2y))});
	return Mout;
}

bool KondoU1::
validate (qType qnum) const
{
	if (qnum[0]<=2*static_cast<int>(this->N_phys) and qnum[0]>0) {return true;}
	else {return false;}
}

template<typename Symmetry_>
void KondoU1::
add_operators (HamiltonianTermsXd<Symmetry_> &Terms, const SpinBase<Symmetry_> &B, const FermionBase<Symmetry_> &F, const ParamHandler &P, size_t loc)
{
	auto save_label = [&Terms] (string label)
	{
		if (label!="") {Terms.info.push_back(label);}
	};
	
	// Bx substrate
	auto [Bxsub,Bxsuborb,Bxsublabel] = P.fill_array1d<double>("Bxsub","Bxsuborb",F.orbitals(),loc);
	save_label(Bxsublabel);
	
	// Bx impurities
	auto [Bx,Bxorb,Bxlabel] = P.fill_array1d<double>("Bx","Bxorb",F.orbitals(),loc);
	save_label(Bxlabel);
	
	// Kx anisotropy
	auto [Kx,Kxorb,Kxlabel] = P.fill_array1d<double>("Kx","Kxorb",B.orbitals(),loc);
	save_label(Kxlabel);
	
	auto Himp = kroneckerProduct(B.HeisenbergHamiltonian(0.,0.,B.ZeroField(),Bxorb,B.ZeroField(),Kxorb,0.,P.get<bool>("CYLINDER")), F.Id());
	auto Hsub = kroneckerProduct(B.Id(),F.HubbardHamiltonian(F.ZeroField(),F.ZeroField(),F.ZeroField(),Bxsuborb,0.,0.,0., P.get<bool>("CYLINDER")));
	
	Terms.local.push_back(make_tuple(1.,Himp+Hsub));
	
	Terms.name = "Transverse-field Kondo";
}

};

#endif

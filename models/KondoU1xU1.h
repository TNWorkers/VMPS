#ifndef STRAWBERRY_KONDOMODEL
#define STRAWBERRY_KONDOMODEL

#include "ParamHandler.h" // from HELPERS

#include "bases/FermionBase.h"
#include "bases/SpinBase.h"
#include "symmetry/U1xU1.h"

namespace VMPS
{

/** \class KondoU1xU1
  * \ingroup Kondo
  *
  * \brief Kondo Model
  *
  * MPO representation of 
  \f[
  H = - \sum_{<ij>\sigma} \left(c^\dagger_{i\sigma}c_{j\sigma} +h.c.\right)
  - J \sum_{i \in I} \mathbf{S}_i \cdot \mathbf{s}_i - \sum_{i \in I} B_i^z S_i^z
  \f]
  *
   where further parameters from HubbardU1xU1 and HeisenbergU1 are possible.
  \param D : \f$D=2S+1\f$ where \f$S\f$ is the spin of the impurity.

  \note Take use of the \f$S_z\f$ U(1) symmetry and the U(1) particle conservation symmetry.
  \note The default variable settings can be seen in \p KondoU1xU1::defaults.
  \note \f$J<0\f$ is antiferromagnetic
  \note If nnn-hopping is positive, the GS-energy is lowered.
  \note The multi-impurity model can be received, by setting D=1 (S=0) for all sites without an impurity.
*/
class KondoU1xU1 : public MpoQ<Sym::U1xU1<double>,double>
{
public:
	typedef Sym::U1xU1<double> Symmetry;
	
private:
	typedef typename Symmetry::qType qType;
	
public:
	
	///@{
	KondoU1xU1 () : MpoQ(){};
	KondoU1xU1 (const size_t &L, const vector<Param> &params);
	///@}
	
	/**
	   \param B : Base class from which the local spin-operators are received
	   \param F : Base class from which the local fermion-operators are received
	   \param P : The parameters
	*/
	template<typename Symmetry_> 
	static HamiltonianTermsXd<Symmetry_> set_operators (const SpinBase<Symmetry_> &B, const FermionBase<Symmetry_> &F,
	                                                    const ParamHandler &P, size_t loc=0);
	
	///@{
	/**Makes half-integers in the output for the magnetization quantum number.*/
	static string N_halveM (qType qnum);
	
	/**Labels the conserved quantum numbers as "N", "M".*/
	static const std::array<string,2> NMlabel;
	///@}
	
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

	///@{ //not implemented
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

const std::map<string,std::any> KondoU1xU1::defaults =
{
	{"t",1.}, {"tPerp",0.},{"tPrime",0.},
	{"J",-1.}, 
	{"U",0.}, {"V",0.}, {"Vperp",0.}, 
	{"mu",0.}, {"t0",0.},
	{"Bz",0.}, {"Bzsub",0.}, {"Kz",0.},
	{"D",2ul},
	{"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}
};

const std::array<string,2> KondoU1xU1::NMlabel{"N","M"};

KondoU1xU1::
KondoU1xU1 (const size_t &L, const vector<Param> &params)
:MpoQ<Symmetry> (L, qarray<Symmetry::Nq>({0,0}), KondoU1xU1::NMlabel, "")//, KondoU1xU1::N_halveM())
{
	ParamHandler P(params,defaults);
	
	size_t Lcell = P.size();
	vector<SuperMatrix<Symmetry,double> > G;
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	B.resize(N_sites); F.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		
		F[l] = FermionBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), !isfinite(P.get<double>("U",l%Lcell)), true); //true means basis n,m
		B[l] = SpinBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), P.get<size_t>("D",l%Lcell));
		
		setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l);
		
		Terms[l] = set_operators(B[l],F[l],P,l%Lcell);
		this->Daux = Terms[l].auxdim();
		
		G.push_back(Generator(Terms[l]));
		setOpBasis(G[l].calc_qOp(),l);
	}
	
	this->generate_label(Terms[0].name,Terms,Lcell);
	this->construct(G, this->W, this->Gvec, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
}

MpoQ<Sym::U1xU1<double> > KondoU1xU1::
c (SPIN_INDEX sigma, size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<F[locx].dim());
	stringstream ss;
	ss << "c(" << locx << "," << locy << "," << sigma << ")";
	
	qarray<2> qdiff;
	(sigma==UP) ? qdiff={-1,0} : qdiff={0,-1};

	MpoQ<Symmetry> Mout(N_sites, qdiff, KondoU1xU1::NMlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l); }
	
	Mout.setLocal(locx, kroneckerProduct(B[locx].Id(),F[locx].c(sigma,locy)), kroneckerProduct(B[locx].Id(),F[0].sign()));
	
	return Mout;
}

MpoQ<Sym::U1xU1<double> > KondoU1xU1::
cdag (SPIN_INDEX sigma, size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<F[locx].dim());
	stringstream ss;
	ss << "cdag(" << locx << "," << locy << "," << sigma << ")";
	
	qarray<2> qdiff;
	(sigma==UP) ? qdiff={-1,0} : qdiff={0,-1};

	MpoQ<Symmetry> Mout(N_sites, qdiff, KondoU1xU1::NMlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l); }
	
	Mout.setLocal(locx, kroneckerProduct(B[locx].Id(),F[locx].cdag(sigma,locy)), kroneckerProduct(B[locx].Id(),F[0].sign()));
	
	return Mout;
}

MpoQ<Sym::U1xU1<double> > KondoU1xU1::
n (std::size_t locx, std::size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	std::stringstream ss;
	ss << "occ(" << locx << "," << locy << ")";

	MpoQ<Symmetry> Mout(N_sites, Symmetry::qvacuum(), KondoU1xU1::NMlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l); }

	auto n = kroneckerProduct(B[locx].Id(),F[locx].n(locy));
	Mout.setLocal(locx, n);

	return Mout;	
}

MpoQ<Sym::U1xU1<double> > KondoU1xU1::
d (std::size_t locx, std::size_t locy)
{
	assert(locx<N_sites and locy<F[locx].dim());
	stringstream ss;
	ss << "double_occ(" << locx << "," << locy << ")";
	
	MpoQ<Symmetry> Mout(N_sites, Symmetry::qvacuum(), KondoU1xU1::NMlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l); }

	auto d = kroneckerProduct(B[locx].Id(),F[locx].d(locy));
	Mout.setLocal(locx, d);
	return Mout;
}

MpoQ<Sym::U1xU1<double> > KondoU1xU1::
cdagc (SPIN_INDEX sigma, size_t loc1x, size_t loc2x, size_t loc1y, size_t loc2y)
{
	assert(loc1x<N_sites and loc2x<N_sites);
	stringstream ss;
	ss << "c†(" << loc1x << "," << loc1y << "," << sigma << ")" 
	   << "c (" << loc2x << "," << loc2y << "," << sigma << ")";
	
	auto cdag = kroneckerProduct(B[loc1x].Id(),F[loc1x].cdag(sigma,loc1y));
	auto c    = kroneckerProduct(B[loc2x].Id(),F[loc2x].c   (sigma,loc2y));
	auto sign = kroneckerProduct(B[loc2x].Id(),F[loc2x].sign());
	
	MpoQ<Symmetry> Mout(N_sites, Symmetry::qvacuum(), KondoU1xU1::NMlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l); }
	
	if (loc1x == loc2x)
	{
		Mout.setLocal(loc1x, cdag*c);
	}
	else if (loc1x<loc2x)
	{
		Mout.setLocal({loc1x, loc2x}, {cdag*sign, c}, sign);
	}
	else if (loc1x>loc2x)
	{
		Mout.setLocal({loc2x, loc1x}, {c*sign, -1.*cdag}, sign);
	}
	
	return Mout;
}

MpoQ<Sym::U1xU1<double> > KondoU1xU1::
Simp (size_t locx, SPINOP_LABEL SOP, size_t locy)
{
	assert(locx<this->N_sites);
	stringstream ss;
	ss << SOP << "(" << locx << "," << locy << ")";

	MpoQ<Symmetry> Mout(N_sites, B[locx].getQ(SOP), KondoU1xU1::NMlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l); }

	Mout.setLocal(locx, kroneckerProduct(B[locx].Scomp(SOP,locy), F[locx].Id()));
	return Mout;
}

MpoQ<Sym::U1xU1<double> > KondoU1xU1::
Ssub (size_t locx, SPINOP_LABEL SOP, size_t locy)
{
	assert(locx<this->N_sites);
	stringstream ss;
	ss << SOP << "(" << locx << "," << locy << ")";

	MpoQ<Symmetry> Mout(N_sites, B[locx].getQ(SOP), KondoU1xU1::NMlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l); }

	Mout.setLocal(locx, kroneckerProduct(B[locx].Id(), F[locx].Scomp(SOP,locy)));
	return Mout;
}

MpoQ<Sym::U1xU1<double> > KondoU1xU1::
SimpSimp (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y, size_t loc2y)
{
	assert(loc1x<this->N_sites and loc2x<this->N_sites);
	std::stringstream ss;
	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")";

	MpoQ<Symmetry> Mout(N_sites, Symmetry::qvacuum(), KondoU1xU1::NMlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l); }

	Mout.setLocal({loc1x, loc2x}, {kroneckerProduct(B[loc1x].Scomp(SOP1,loc1y),F[loc1x].Id()), kroneckerProduct(B[loc2x].Scomp(SOP2,loc2y),F[loc2y].Id())});
	return Mout;
}

MpoQ<Sym::U1xU1<double> > KondoU1xU1::
SsubSsub (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y, size_t loc2y)
{
	assert(loc1x<this->N_sites and loc2x<this->N_sites);
	std::stringstream ss;
	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")";

	MpoQ<Symmetry> Mout(N_sites, Symmetry::qvacuum(), KondoU1xU1::NMlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l); }

	Mout.setLocal({loc1x, loc2x}, {kroneckerProduct(B[loc1x].Id(),F[loc1x].Scomp(SOP1,loc1y)), kroneckerProduct(B[loc2x].Id(),F[loc2x].Scomp(SOP2,loc2y))});
	return Mout;
}

MpoQ<Sym::U1xU1<double> > KondoU1xU1::
SimpSsub (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y, size_t loc2y)
{
	assert(loc1x<this->N_sites and loc2x<this->N_sites);
	std::stringstream ss;
	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")";

	MpoQ<Symmetry> Mout(N_sites, Symmetry::qvacuum(), KondoU1xU1::NMlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l); }

	Mout.setLocal({loc1x, loc2x}, {kroneckerProduct(B[loc1x].Scomp(SOP1,loc1y),F[loc1x].Id()), kroneckerProduct(B[loc2x].Id(),F[loc2x].Scomp(SOP2,loc2y))});
	return Mout;
}

// MpoQ<Sym::U1xU1<double> > KondoU1xU1::
// SimpSsubSimpSimp (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc3x, SPINOP_LABEL SOP3, size_t loc4x, SPINOP_LABEL SOP4,
//                   size_t loc1y, size_t loc2y, size_t loc3y, size_t loc4y)
// {
// 	assert(loc1x<this->N_sites and loc2x<this->N_sites and loc3x<this->N_sites and loc4x<this->N_sites);
// 	stringstream ss;
// 	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")" <<
// 	      SOP3 << "(" << loc3x << "," << loc3y << ")" << SOP4 << "(" << loc4x << "," << loc4y << ")";
// 	MpoQ<Symmetry> Mout(this->N_sites, this->N_legs, locBasis(), {0,0}, KondoU1xU1::NMlabel, ss.str());
// 	Mout.setLocal({loc1x, loc2x, loc3x, loc4x},
// 				  {kroneckerProduct(B.Scomp(SOP1,loc1y),F.Id()), 
// 				   kroneckerProduct(B.Id(),F.Scomp(SOP2,loc2y)),
// 				   kroneckerProduct(B.Scomp(SOP3,loc3y),F.Id()),
// 				   kroneckerProduct(B.Scomp(SOP4,loc4y),F.Id())});
// 	return Mout;
// }

// MpoQ<Sym::U1xU1<double> > KondoU1xU1::
// SimpSsubSimpSsub (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc3x, SPINOP_LABEL SOP3, size_t loc4x, SPINOP_LABEL SOP4,
//                   size_t loc1y, size_t loc2y, size_t loc3y, size_t loc4y)
// {
// 	assert(loc1x<this->N_sites and loc2x<this->N_sites and loc3x<this->N_sites and loc4x<this->N_sites);
// 	stringstream ss;
// 	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")" <<
// 	      SOP3 << "(" << loc3x << "," << loc3y << ")" << SOP4 << "(" << loc4x << "," << loc4y << ")";
// 	MpoQ<Symmetry> Mout(this->N_sites, this->N_legs, locBasis(), {0,0}, KondoU1xU1::NMlabel, ss.str());
// 	SparseMatrixXd IdSub(F.dim(),F.dim()); IdSub.setIdentity();
// 	SparseMatrixXd IdImp(MpoQ<Symmetry>::qloc[loc2x].size()/F.dim(), MpoQ<Symmetry>::qloc[loc2x].size()/F.dim()); IdImp.setIdentity();
// 	Mout.setLocal({loc1x, loc2x, loc3x, loc4x},
// 				  {kroneckerProduct(B.Scomp(SOP1,loc1y),F.Id()), 
// 				   kroneckerProduct(B.Id(),F.Scomp(SOP2,loc2y)),
// 				   kroneckerProduct(B.Scomp(SOP3,loc3y),F.Id()),
// 				   kroneckerProduct(B.Id(),F.Scomp(SOP4,loc4y))});
// 	return Mout;
// }

bool KondoU1xU1::
validate (qType qnum) const
{
	frac S_elec(qnum[0],2); //electrons have spin 1/2
	frac Smax = S_elec;
	for (size_t l=0; l<N_sites; ++l) { Smax+=static_cast<int>(B[l].orbitals())*frac(B[l].get_D()-1,2); } //add local spins to Smax
	
	frac S_tot(qnum[1],2);
	cout << S_tot << "\t" << Smax << endl;
	if (Smax.denominator()==S_tot.denominator() and S_tot<=Smax and qnum[0]<=2*static_cast<int>(this->N_phys) and qnum[0]>0) {return true;}
	else {return false;}
}

string KondoU1xU1::
N_halveM (qType qnum)
{
	stringstream ss;
	ss << "(" << qnum[0] << ",";
	
	qarray<1> mag;
	mag[0] = qnum[1];
	string halfmag = ::halve(mag);
	halfmag.erase(0,1);
	ss << halfmag;
	
	return ss.str();
}

template<typename Symmetry_>
HamiltonianTermsXd<Symmetry_> KondoU1xU1::
set_operators (const SpinBase<Symmetry_> &B, const FermionBase<Symmetry_> &F, const ParamHandler &P, size_t loc)
{
	HamiltonianTermsXd<Symmetry_> Terms;
	
	frac S = frac(B.get_D()-1,2);
	stringstream Slabel;
	Slabel << "S=" << print_frac_nice(S);
	Terms.info.push_back(Slabel.str());
	
	auto save_label = [&Terms] (string label)
	{
		if (label!="") {Terms.info.push_back(label);}
	};
	
	// NN terms
	
	auto [t,tPara,tlabel] = P.fill_array2d<double>("t","tPara",F.orbitals(),loc);
	save_label(tlabel);
	
	auto [V,Vpara,Vlabel] = P.fill_array2d<double>("V","Vpara",F.orbitals(),loc);
	save_label(Vlabel);
	
	for (int i=0; i<F.orbitals(); ++i)
	for (int j=0; j<F.orbitals(); ++j)
	{
		if (tPara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(-tPara(i,j),
			                                 kroneckerProduct(B.Id(), F.cdag(UP,i) * F.sign()),
			                                 kroneckerProduct(B.Id(), F.c(UP,j))));
			Terms.tight.push_back(make_tuple(-tPara(i,j),
			                                 kroneckerProduct(B.Id(), F.cdag(DN,i) * F.sign()),
			                                 kroneckerProduct(B.Id(), F.c(DN,j))));
			Terms.tight.push_back(make_tuple(-tPara(i,j),
			                                 kroneckerProduct(B.Id(), -1.*F.c(UP,i) * F.sign()),
			                                 kroneckerProduct(B.Id(), F.cdag(UP,j))));
			Terms.tight.push_back(make_tuple(-tPara(i,j),
			                                 kroneckerProduct(B.Id(), -1.*F.c(DN,i) * F.sign()),
			                                 kroneckerProduct(B.Id(), F.cdag(DN,j))));
		}
		
		if (Vpara(i,j) != 0.)
		{
			if (Vpara(i,j) != 0.)
			{
				Terms.tight.push_back(make_tuple(Vpara(i,j), 
				                                 kroneckerProduct(B.Id(),F.n(i)), 
				                                 kroneckerProduct(B.Id(),F.n(j))));
			}
		}
	}
	
	// NNN terms
	
	param0d tPrime = P.fill_array0d<double>("tPrime","tPrime",loc);
	save_label(tPrime.label);
	
	if (tPrime.x!=0)
	{
		assert(F.orbitals() == 1 and "Cannot do a ladder with t' terms!");
		
		Terms.nextn.push_back(make_tuple(-tPrime.x,
		                                 kroneckerProduct(B.Id(),F.cdag(UP,0)),
		                                 kroneckerProduct(B.Id(),F.sign()* F.c(UP,0)),
		                                 kroneckerProduct(B.Id(),F.sign())));
		Terms.nextn.push_back(make_tuple(-tPrime.x,
		                                 kroneckerProduct(B.Id(),F.cdag(DN,0)),
		                                 kroneckerProduct(B.Id(),F.sign()* F.c(DN,0)),
		                                 kroneckerProduct(B.Id(),F.sign())));
		Terms.nextn.push_back(make_tuple(+tPrime.x,
		                                 kroneckerProduct(B.Id(),F.c(UP,0)),
		                                 kroneckerProduct(B.Id(),F.sign()* F.cdag(UP,0)),
		                                 kroneckerProduct(B.Id(),F.sign())));
		Terms.nextn.push_back(make_tuple(+tPrime.x,
		                                 kroneckerProduct(B.Id(),F.c(DN,0)),
		                                 kroneckerProduct(B.Id(),F.sign()* F.cdag(DN,0)),
		                                 kroneckerProduct(B.Id(),F.sign())));
	}
	
	
	// local terms
	
	// t⟂
	param0d tPerp = P.fill_array0d<double>("t","tPerp",loc);
	save_label(tPerp.label);
	
	// V⟂
	param0d Vperp = P.fill_array0d<double>("V","Vperp",loc);
	save_label(Vperp.label);
	
	// Hubbard U
	auto [U,Uorb,Ulabel] = P.fill_array1d<double>("U","Uorb",F.orbitals(),loc);
	save_label(Ulabel);
	
	// mu
	auto [mu,muorb,mulabel] = P.fill_array1d<double>("mu","muorb",F.orbitals(),loc);
	save_label(mulabel);
	
	// t0
	auto [t0,t0orb,t0label] = P.fill_array1d<double>("t0","t0orb",F.orbitals(),loc);
	save_label(t0label);
	
	// Kz anisotropy
	auto [Kz,Kzorb,Kzlabel] = P.fill_array1d<double>("Kz","Kzorb",F.orbitals(),loc);
	save_label(Kzlabel);
	
	// Bz substrate
	auto [Bzsub,Bzsuborb,Bzsublabel] = P.fill_array1d<double>("Bzsub","Bzsuborb",F.orbitals(),loc);
	save_label(Bzsublabel);
	
	// Bz impurities
	auto [Bz,Bzorb,Bzlabel] = P.fill_array1d<double>("Bz","Bzorb",F.orbitals(),loc);
	save_label(Bzlabel);
	
	auto Himp = kroneckerProduct(B.HeisenbergHamiltonian(0.,0.,Bzorb,B.ZeroField(),Kzorb,B.ZeroField(),0.,P.get<bool>("CYLINDER")),F.Id());
	auto Hsub = kroneckerProduct(B.Id(),F.HubbardHamiltonian(Uorb,t0orb-muorb,Bzsuborb,B.ZeroField(),tPerp.x,Vperp.x,0., P.get<bool>("CYLINDER")));
	auto Hloc = Himp + Hsub;
	
	// Kondo-J
	auto [J,Jorb,Jlabel] = P.fill_array1d<double>("J","Jorb",F.orbitals(),loc);
	save_label(Jlabel);
	
	for (int i=0; i<F.orbitals(); ++i)
	{
		if (Jorb(i) != 0.)
		{
			Hloc += -Jorb(i)    * kroneckerProduct(B.Scomp(SZ,i),F.Sz(i));
			Hloc += -0.5*Jorb(i)* kroneckerProduct(B.Scomp(SP,i),F.Sm(i));
			Hloc += -0.5*Jorb(i)* kroneckerProduct(B.Scomp(SM,i),F.Sp(i));
		}
	}
	
	Terms.name = "Kondo";
	
	Terms.local.push_back(make_tuple(1.,Hloc));
	
	return Terms;
}

} //end namespace VMPS

#endif

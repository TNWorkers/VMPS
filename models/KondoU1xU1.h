#ifndef STRAWBERRY_KONDOMODEL
#define STRAWBERRY_KONDOMODEL

#include "models/HubbardU1xU1.h"
#include "bases/FermionBase.h"
#include "bases/SpinBase.h"
#include "symmetry/qarray.h"

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
  \todo Most of the observalbes need to be adjusted properly.
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
	KondoU1xU1 (const variant<size_t,std::array<size_t,2> > &L, const vector<Param> &params);
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
	MpoQ<Symmetry> Simp (size_t locx, SPINOP_LABEL SOP, size_t locy=0);
	MpoQ<Symmetry> Ssub (size_t locx, SPINOP_LABEL SOP, size_t locy=0);
	MpoQ<Symmetry> SimpSimp (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y=0, size_t loc2y=0);
	MpoQ<Symmetry> SsubSsub (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y=0, size_t loc2y=0);
	MpoQ<Symmetry> SimpSsub (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y=0, size_t loc2y=0);
	MpoQ<Symmetry> SimpSsubSimpSimp (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, 
	                          size_t loc3x, SPINOP_LABEL SOP3, size_t loc4x, SPINOP_LABEL SOP4,
	                          size_t loc1y=0, size_t loc2y=0, size_t loc3y=0, size_t loc4y=0);
	MpoQ<Symmetry> SimpSsubSimpSsub (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, 
	                          size_t loc3x, SPINOP_LABEL SOP3, size_t loc4x, SPINOP_LABEL SOP4,
	                          size_t loc1y=0, size_t loc2y=0, size_t loc3y=0, size_t loc4y=0);
	MpoQ<Symmetry> d (size_t locx, size_t locy=0);
	MpoQ<Symmetry> c (SPIN_INDEX sigma, size_t locx, size_t locy=0);
	MpoQ<Symmetry> cdag (SPIN_INDEX sigma, size_t locx, size_t locy=0);
	MpoQ<Symmetry> cdagc (SPIN_INDEX sigma, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
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
KondoU1xU1 (const variant<size_t,std::array<size_t,2> > &L, const vector<Param> &params)
:MpoQ<Symmetry> (holds_alternative<size_t>(L)? get<0>(L):get<1>(L)[0],
                 holds_alternative<size_t>(L)? 1        :get<1>(L)[1],
                 qarray<Symmetry::Nq>({0,0}), KondoU1xU1::NMlabel, "")//, KondoU1xU1::N_halveM())
{
	ParamHandler P(params,defaults);
	
	size_t Lcell = P.size();
	vector<SuperMatrix<Symmetry,double> > G;
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	B.resize(N_sites); F.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		F[l] = FermionBase<Symmetry>(N_legs, !isfinite(P.get<double>("U",l%Lcell)), true); //true means basis n,m
		B[l] = SpinBase<Symmetry>(N_legs, P.get<size_t>("D",l%Lcell));
		
		setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l);
		
		Terms[l] = set_operators(B[l],F[l],P,l%Lcell);
		this->Daux = Terms[l].auxdim();
		
		G.push_back(Generator(Terms[l]));
		setOpBasis(G[l].calc_qOp(),l);
	}
	
	this->generate_label(Terms[0].name,Terms,Lcell);
	this->construct(G, this->W, this->Gvec, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
}

// MpoQ<Sym::U1xU1<double> > KondoU1xU1::
// Simp (size_t locx, SPINOP_LABEL SOP, size_t locy)
// {
// 	assert(locx<this->N_sites);
// 	stringstream ss;
// 	ss << SOP << "(" << locx << "," << locy << ")";
// 	MpoQ<Symmetry> Mout(this->N_sites, this->N_legs, locBasis(), {0,0}, KondoU1xU1::NMlabel, ss.str());
// 	Mout.setLocal(locx, kroneckerProduct(B.Scomp[locx](SOP,locy),F.Id()));
// 	return Mout;
// }

// MpoQ<Sym::U1xU1<double> > KondoU1xU1::
// Ssub (size_t locx, SPINOP_LABEL SOP, size_t locy)
// {
// 	assert(locx<this->N_sites);
// 	stringstream ss;
// 	ss << SOP << "(" << locx << "," << locy << ")";
// 	MpoQ<Symmetry> Mout(this->N_sites, this->N_legs, locBasis(), {0,0}, KondoU1xU1::NMlabel, ss.str());
// 	Mout.setLocal(locx, kroneckerProduct(B.Id(), F.Scomp(SOP,locy)));
// 	return Mout;
// }

// MpoQ<Sym::U1xU1<double> > KondoU1xU1::
// SimpSimp (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y, size_t loc2y)
// {
// 	assert(loc1x<this->N_sites and loc2x<this->N_sites);
// 	stringstream ss;
// 	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")";
// 	MpoQ<Symmetry> Mout(this->N_sites, this->N_legs, locBasis(), {0,0}, KondoU1xU1::NMlabel, ss.str());
// 	Mout.setLocal({loc1x, loc2x}, {kroneckerProduct(B.Scomp(SOP1,loc1y),F.Id()), kroneckerProduct(B.Scomp(SOP2,loc2y),F.Id())});
// 	return Mout;
// }

// MpoQ<Sym::U1xU1<double> > KondoU1xU1::
// SsubSsub (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y, size_t loc2y)
// {
// 	assert(loc1x<this->N_sites and loc2x<this->N_sites);
// 	stringstream ss;
// 	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")";
// 	MpoQ<Symmetry> Mout(this->N_sites, this->N_legs, locBasis(), {0,0}, KondoU1xU1::NMlabel, ss.str());
// 	Mout.setLocal({loc1x, loc2x}, {kroneckerProduct(B.Id(),F.Scomp(SOP1,loc1y)), kroneckerProduct(B.Id(),F.Scomp(SOP2,loc2y))});
// 	return Mout;
// }

// MpoQ<Sym::U1xU1<double> > KondoU1xU1::
// SimpSsub (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y, size_t loc2y)
// {
// 	assert(loc1x<this->N_sites and loc2x<this->N_sites);
// 	stringstream ss;
// 	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")";
// 	MpoQ<Symmetry> Mout(this->N_sites, this->N_legs, locBasis(), {0,0}, KondoU1xU1::NMlabel, ss.str());
// 	Mout.setLocal({loc1x, loc2x}, {kroneckerProduct(B.Scomp(SOP1,loc1y),F.Id()), kroneckerProduct(B.Id(),F.Scomp(SOP2,loc2y))});
// 	return Mout;
// }

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

// MpoQ<Sym::U1xU1<double> > KondoU1xU1::
// c (SPIN_INDEX sigma, size_t locx, size_t locy)
// {
// 	assert(locx<this->N_sites and locy<N_legs);
// 	stringstream ss;
// 	ss << "c(" << locx << "," << locy << ",σ=" << sigma << ")";
// 	qType qdiff;
// 	(sigma==UP) ? qdiff = {-1,-1} : qdiff = {-1,+1};
	
// 	vector<SuperMatrix<double> > M(N_sites);
// 	for (size_t l=0; l<locx; ++l)
// 	{
// 		SparseMatrixXd IdImp(MpoQ<Symmetry>::qloc[l].size()/F.dim(), MpoQ<Symmetry>::qloc[l].size()/F.dim()); IdImp.setIdentity();
// 		M[l].setMatrix(1,S.dim()*F.dim());
// 		SparseMatrixXd tmp = kroneckerProduct(IdImp,F.sign());
// 		M[l](0,0) = tmp;
// 	}
// 	SparseMatrixXd IdImp(MpoQ<Symmetry>::qloc[locx].size()/F.dim(), MpoQ<Symmetry>::qloc[locx].size()/F.dim()); IdImp.setIdentity();
// 	M[locx].setMatrix(1,S.dim()*F.dim());
// 	SparseMatrixXd tmp = (sigma==UP)? kroneckerProduct(IdImp,F.sign_local(locy)*F.c(UP,locy)) : kroneckerProduct(IdImp,F.sign_local(locy)*F.c(DN,locy));
// 	M[locx](0,0) = tmp;
// 	for (size_t l=locx+1; l<N_sites; ++l)
// 	{
// 		M[l].setMatrix(1,S.dim()*F.dim());
// 		M[l](0,0).setIdentity();
// 	}
	
// 	return MpoQ<Symmetry>(N_sites, N_legs, M, locBasis(), qdiff, KondoU1xU1::NMlabel, ss.str());
// }

// MpoQ<Sym::U1xU1<double> > KondoU1xU1::
// cdag (SPIN_INDEX sigma, size_t locx, size_t locy)
// {
// 	assert(locx<N_sites and locy<N_legs);
// 	stringstream ss;
// 	ss << "c†(" << locx << "," << locy << ",σ=" << sigma << ")";
// 	qType qdiff;
// 	(sigma==UP) ? qdiff = {+1,+1} : qdiff = {+1,-1};
	
// 	vector<SuperMatrix<double> > M(N_sites);
// 	for (size_t l=0; l<locx; ++l)
// 	{
// 		SparseMatrixXd IdImp(MpoQ<Symmetry>::qloc[l].size()/F.dim(), MpoQ<Symmetry>::qloc[l].size()/F.dim()); IdImp.setIdentity();
// 		M[l].setMatrix(1,S.dim()*F.dim());
// 		SparseMatrixXd tmp = kroneckerProduct(IdImp,F.sign());
// 		M[l](0,0) = tmp;
// 	}
// 	SparseMatrixXd IdImp(MpoQ<Symmetry>::qloc[locx].size()/F.dim(), MpoQ<Symmetry>::qloc[locx].size()/F.dim()); IdImp.setIdentity();
// 	M[locx].setMatrix(1,S.dim()*F.dim());
// 	SparseMatrixXd tmp = (sigma==UP)? kroneckerProduct(IdImp,F.sign_local(locy)*F.cdag(UP,locy)) : kroneckerProduct(IdImp,F.sign_local(locy)*F.cdag(DN,locy));
// 	M[locx](0,0) = tmp;
// 	for (size_t l=locx+1; l<N_sites; ++l)
// 	{
// 		M[l].setMatrix(1,S.dim()*F.dim());
// 		M[l](0,0).setIdentity();
// 	}
	
// 	return MpoQ<Symmetry>(N_sites, N_legs, M, locBasis(), qdiff, KondoU1xU1::NMlabel, ss.str());
// }

// MpoQ<Sym::U1xU1<double> > KondoU1xU1::
// cdagc (SPIN_INDEX sigma, size_t locx1, size_t locx2, size_t locy1, size_t locy2)
// {
// 	assert(locx1<N_sites and locx2<N_sites and locy1<N_legs and locy2<N_legs);
// 	stringstream ss;
// 	ss << "c†(" << locx1 << "," << locy1 << ",σ=" << sigma << ") " << "c(" << locx2 << "," << locy2 << ",σ=" << sigma << ")";
// 	qType qdiff = {0,0};

// 	vector<SuperMatrix<double> > M(N_sites);
// 	SparseMatrixXd IdImp;
	
// 	if (locx1 < locx2)
// 	{
// 		for (size_t l=0; l<locx1; ++l)
// 		{
// 			M[l].setMatrix(1,S.dim()*F.dim());
// 			M[l](0,0).setIdentity();
// 			// M[l](0,0) = kroneckerProduct(IdImp,F.sign());
// 		}
// 		IdImp.resize(MpoQ<Symmetry>::qloc[locx1].size()/F.dim(), MpoQ<Symmetry>::qloc[locx1].size()/F.dim()); IdImp.setIdentity();
// 		M[locx1].setMatrix(1,S.dim()*F.dim());
// 		SparseMatrixXd tmp = (sigma==UP) ? kroneckerProduct(IdImp,F.cdag(UP,locy1)) : kroneckerProduct(IdImp,F.cdag(DN,locy1));
// 		M[locx1](0,0) = tmp;
// 		for (size_t l=locx1+1; l<locx2; ++l)
// 		{
// 			IdImp.resize(MpoQ<Symmetry>::qloc[l].size()/F.dim(), MpoQ<Symmetry>::qloc[l].size()/F.dim()); IdImp.setIdentity();
// 			M[l].setMatrix(1,S.dim()*F.dim());
// 			SparseMatrixXd tmp = kroneckerProduct(IdImp,F.sign());
// 			M[l](0,0) = tmp;
// 		}
// 		IdImp.resize(MpoQ<Symmetry>::qloc[locx2].size()/F.dim(), MpoQ<Symmetry>::qloc[locx2].size()/F.dim()); IdImp.setIdentity();
// 		M[locx2].setMatrix(1,S.dim()*F.dim());
// 		tmp = (sigma==UP) ? kroneckerProduct(IdImp,F.sign_local(locy2)*F.c(UP,locy2)) : kroneckerProduct(IdImp,F.sign_local(locy2)*F.c(DN,locy2));
// 		M[locx2](0,0) = tmp;
// 		for (size_t l=locx2+1; l<N_sites; ++l)
// 		{
// 			M[l].setMatrix(1,S.dim()*F.dim());
// 			M[l](0,0).setIdentity();
// 		}
// 	}
// 	else if(locx1 > locx2)
// 	{
// 		for (size_t l=0; l<locx2; ++l)
// 		{
// 			M[l].setMatrix(1,S.dim()*F.dim());
// 			M[l](0,0).setIdentity();
// 			// M[l](0,0) = kroneckerProduct(IdImp,F.sign());
// 		}
// 		IdImp.resize(MpoQ<Symmetry>::qloc[locx2].size()/F.dim(), MpoQ<Symmetry>::qloc[locx2].size()/F.dim()); IdImp.setIdentity();
// 		M[locx2].setMatrix(1,S.dim()*F.dim());
// 		SparseMatrixXd tmp = (sigma==UP) ? kroneckerProduct(IdImp,F.c(UP,locy2)) : kroneckerProduct(IdImp,F.c(DN,locy2));
// 		M[locx2](0,0) = tmp;
// 		for (size_t l=locx2+1; l<locx1; ++l)
// 		{
// 			IdImp.resize(MpoQ<Symmetry>::qloc[l].size()/F.dim(), MpoQ<Symmetry>::qloc[l].size()/F.dim()); IdImp.setIdentity();
// 			M[l].setMatrix(1,S.dim()*F.dim());
// 			SparseMatrixXd tmp = kroneckerProduct(IdImp,F.sign());
// 			M[l](0,0) = tmp;
// 		}
// 		IdImp.resize(MpoQ<Symmetry>::qloc[locx1].size()/F.dim(), MpoQ<Symmetry>::qloc[locx1].size()/F.dim()); IdImp.setIdentity();
// 		M[locx1].setMatrix(1,S.dim()*F.dim());
// 		tmp = (sigma==UP) ? kroneckerProduct(IdImp,F.sign_local(locy1)*F.cdag(UP,locy1)) : kroneckerProduct(IdImp,F.sign_local(locy1)*F.cdag(DN,locy1));
// 		M[locx1](0,0) = tmp;
// 		for (size_t l=locx1+1; l<N_sites; ++l)
// 		{
// 			M[l].setMatrix(1,S.dim()*F.dim());
// 			M[l](0,0).setIdentity();
// 		}		
// 	}
// 	else if(locx1 == locx2)
// 	{
// 		for (size_t l=0; l<locx1; ++l)
// 		{
// 			M[l].setMatrix(1,S.dim()*F.dim());
// 			M[l](0,0).setIdentity();
// 			// M[l](0,0) = kroneckerProduct(IdImp,F.sign());
// 		}
// 		IdImp.resize(MpoQ<Symmetry>::qloc[locx1].size()/F.dim(), MpoQ<Symmetry>::qloc[locx1].size()/F.dim()); IdImp.setIdentity();
// 		M[locx1].setMatrix(1,S.dim()*F.dim());
// 		SparseMatrixXd tmp = (sigma==UP) ? kroneckerProduct(IdImp,F.cdag(UP,locy1)*F.sign_local(locy1)*F.c(UP,locy2))
// 			: kroneckerProduct(IdImp,F.cdag(DN,locy1)*F.sign_local(locy1)*F.c(DN,locy2));
// 		M[locx1](0,0) = tmp;
// 		for (size_t l=locx1+1; l<N_sites; ++l)
// 		{
// 			M[l].setMatrix(1,S.dim()*F.dim());
// 			M[l](0,0).setIdentity();
// 		}		

	
// 	}
// 	return MpoQ<Symmetry>(N_sites, N_legs, M, locBasis(), qdiff, KondoU1xU1::NMlabel, ss.str());
// }

// MpoQ<Sym::U1xU1<double> > KondoU1xU1::
// d (size_t locx, size_t locy)
// {
// 	assert(locx<N_sites and locy<N_legs);
// 	stringstream ss;
// 	ss << "double_occ(" << locx << "," << locy << ")";
// 	SparseMatrixXd IdImp(MpoQ<Symmetry>::qloc[locx].size()/F.dim(), MpoQ<Symmetry>::qloc[locx].size()/F.dim()); IdImp.setIdentity();
// 	MpoQ<Symmetry> Mout(N_sites, N_legs, locBasis(), {0,0}, KondoU1xU1::NMlabel, ss.str());
// 	Mout.setLocal(locx, kroneckerProduct(IdImp,F.d(locy)));
// 	return Mout;
// }

bool KondoU1xU1::
validate (qType qnum) const
{
	frac S_elec(qnum[0],2); //electrons have spin 1/2
	frac Smax = S_elec;
	for (size_t l=0; l<N_sites; ++l) { Smax+=B[l].orbitals()*frac(B[l].get_D()-1,2); } //add local spins to Smax
	
	frac S_tot(qnum[1],2);
	cout << S_tot << "\t" << Smax << endl;
	if (Smax.denominator()==S_tot.denominator() and S_tot<=Smax and qnum[0]<=2*static_cast<int>(this->N_sites*this->N_legs) and qnum[0]>0) {return true;}
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
			                                 kroneckerProduct(B.Id(), F.cdag(UP,i)),
			                                 kroneckerProduct(B.Id(),F.sign()* F.c(UP,j))));
			Terms.tight.push_back(make_tuple(-tPara(i,j),
			                                 kroneckerProduct(B.Id(), F.cdag(DN,i)),
			                                 kroneckerProduct(B.Id(),F.sign()* F.c(DN,j))));
			Terms.tight.push_back(make_tuple(+tPara(i,j),
			                                 kroneckerProduct(B.Id(), F.c(UP,i)),
			                                 kroneckerProduct(B.Id(),F.sign()* F.cdag(UP,j))));
			Terms.tight.push_back(make_tuple(+tPara(i,j),
			                                 kroneckerProduct(B.Id(), F.c(DN,i)),
			                                 kroneckerProduct(B.Id(),F.sign()* F.cdag(DN,j))));
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

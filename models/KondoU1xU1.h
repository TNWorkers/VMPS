#ifndef STRAWBERRY_KONDOMODEL
#define STRAWBERRY_KONDOMODEL

#include "models/HubbardU1xU1.h"
#include "FermionBase.h"
#include "SpinBase.h"
#include "qarray.h"

namespace VMPS
{

/** \class KondoU1xU1
  * \ingroup Models
  *
  * \brief Kondo Model
  *
  * MPO representation of 
  \f$
  H = - \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} -t^{\prime} \sum_{<<ij>>\sigma} c^\dagger_{i\sigma}c_{j\sigma} - J \sum_{i \in I} \mathbf{S}_i \cdot \mathbf{s}_i - \sum_{i \in I} B_i^z S_i^z
  \f$.
  *
  \note Take use of the \f$S_z\f$ U(1) symmetry and the U(1) particle conservation symmetry.
  \note The default variable settings can be seen in \p KondoU1xU1::defaults.
  \note \f$J<0\f$ is antiferromagnetic
  \note If nnn-hopping is positive, the GS-energy is lowered.
  \todo Most of the observalbes need to be adjusted properly.
*/
class KondoU1xU1 : public MpoQ<Sym::U1xU1<double>,double>
{
typedef Sym::U1xU1<double> Symmetry;
typedef typename Symmetry::qType qType;
public:
	/**Does nothing.*/
	KondoU1xU1 () : MpoQ(){};
	KondoU1xU1 (variant<size_t,std::array<size_t,2> > L, vector<Param> params);
	
	/**
	   \param B : Base class from which the local spin-operators are received
	   \param F : Base class from which the local fermion-operators are received
	   \param P : The parameters
	*/
	template<typename Symmetry_> 
	static HamiltonianTermsXd<Symmetry_> set_operators (const SpinBase<Symmetry_> &B, const FermionBase<Symmetry_> &F,
														const ParamHandler &P, size_t loc=0);

	/**Operator Quantum numbers: \f$\{ Id,S_z:k=\left|0\right>; S_+:k=\left|+2\right>; S_-:k=\left|-2\right>\}\f$ */
	static const vector<qType > qOp();

	/**Makes half-integers in the output for the magnetization quantum number.*/
	static string N_halveM (qType qnum);
	
	/**Labels the conserved quantum numbers as "N", "M".*/
	static const std::array<string,2> NMlabel;
	
	///@{
	/**Typedef for convenient reference (no need to specify \p Symmetry, \p Scalar all the time).*/
	typedef MpsQ<Symmetry,double>                           StateXd;
	typedef MpsQ<Symmetry,complex<double> >                 StateXcd;
	typedef DmrgSolverQ<Symmetry,KondoU1xU1,double>         Solver;
	typedef MpsQCompressor<Symmetry,double,double>          CompressorXd;
	typedef MpsQCompressor<Symmetry,complex<double>,double> CompressorXcd;
	typedef MpoQ<Symmetry>                                  Operator;
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

protected:	
	const std::map<string,std::any> defaults = 
	{
		{"J",-1.}, {"U",0.}, {"V",0.}, {"mu",0.},
		{"t",1.}, {"tPara",0.}, {"tPerp",0.},{"tPrime",0.},
		{"D",2ul}, {"Bz",0.}, {"Bz_elec",0.}, {"Bx",0.}, {"K",0.},
		{"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}
	};

	vector<FermionBase<Symmetry> > F;
	vector<SpinBase<Symmetry> > B;
};

const std::array<string,2> KondoU1xU1::NMlabel{"N","M"};

const vector<qarray<2> > KondoU1xU1::
qOp ()
{
	vector<qarray<2> > vout;
	vout.push_back({0,0}); //Id
	vout.push_back({+1,+1});//cUpDAg
	vout.push_back({-1,-1});//cUp
	vout.push_back({-1,+1});//cDNDag
	vout.push_back({+1,-1});//cDn
	// vout.push_back({+1,-1});//
	// vout.push_back({-1,+1});
	return vout;
}

KondoU1xU1::
KondoU1xU1 (variant<size_t,std::array<size_t,2> > L, vector<Param> params)
	:MpoQ<Symmetry> (holds_alternative<size_t>(L)? get<0>(L):get<1>(L)[0],
					 holds_alternative<size_t>(L)? 1        :get<1>(L)[1],
					 qarray<Symmetry::Nq>({0,0}), KondoU1xU1::qOp(), KondoU1xU1::NMlabel, "")//, KondoU1xU1::N_halveM())
{
	ParamHandler P(params,defaults);
	
	size_t Lcell = P.size();
	vector<SuperMatrix<Symmetry,double> > G;
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	B.resize(N_sites);	F.resize(N_sites);

	for (size_t l=0; l<N_sites; ++l)
	{
		F[l] = FermionBase<Symmetry>(N_legs,!isfinite(P.get<double>("U",l%Lcell)),true); //true means basis n,m
		B[l] = SpinBase<Symmetry>(N_legs,P.get<size_t>("D",l%Lcell));

		setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l);
		
		Terms[l] = set_operators(B[l],F[l],P,l%Lcell);
		this->Daux = Terms[l].auxdim();
		
		G.push_back(Generator(Terms[l])); // boost::multi_array has stupid assignment
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

// bool KondoU1xU1::
// validate (qType qnum) const
// {
// 	int Sx2 = static_cast<int>(D-1); // necessary because of size_t
// 	return (qnum[0]+N_legs*Sx2*imploc.size())%2 == qnum[1]%2;
// }

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
	Terms.name = "Kondo";

	frac S = frac(B.get_D()-1,2);
	stringstream Slabel;
	Slabel << "S=" << print_frac_nice(S);
	Terms.info.push_back(Slabel.str());

	// stringstream ss;
	// IOFormat CommaInitFmt(StreamPrecision, DontAlignCols, ",", ",", "", "", "{", "}");
	
	// hopping terms

	auto [t,tPara,tlabel] = P.fill_array2d<double>("t","tPara",F.orbitals(),loc);
	if(!tlabel.empty()) {Terms.info.push_back(tlabel);}
	
	for (int i=0; i<F.orbitals(); ++i)
	for (int j=0; j<F.orbitals(); ++j)
	{
		if (tPara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(-tPara(i,j),kroneckerProduct(B.Id(), F.cdag(UP,i)),
											 kroneckerProduct(B.Id(),F.sign()* F.c(UP,j))));
			Terms.tight.push_back(make_tuple(-tPara(i,j),kroneckerProduct(B.Id(), F.cdag(DN,i)),
											 kroneckerProduct(B.Id(),F.sign()* F.c(DN,j))));
			Terms.tight.push_back(make_tuple(+tPara(i,j),kroneckerProduct(B.Id(), F.c(UP,i)),
											 kroneckerProduct(B.Id(),F.sign()* F.cdag(UP,j))));
			Terms.tight.push_back(make_tuple(+tPara(i,j),kroneckerProduct(B.Id(), F.c(DN,i)),
											 kroneckerProduct(B.Id(),F.sign()* F.cdag(DN,j))));
		}
	}
	
	// V terms
	
	double V = P.get_default<double>("V");
	
	if (P.HAS("V",loc))
	{
		for (int i=0; i<F.orbitals(); ++i)
		{
			Terms.tight.push_back(make_tuple(V, kroneckerProduct(B.Id(),F.n(i)), kroneckerProduct(B.Id(),F.n(i))));
		}
		stringstream Vlabel;
		Vlabel << "V=" << V;
		Terms.info.push_back(Vlabel.str());
	}
	
	/// NNN-terms
	
	double tPrime = P.get_default<double>("tPrime");
	
	if (P.HAS("tPrime",loc))
	{
		tPrime = P.get<double>("tPrime", loc);
		assert((B.orbitals() == 1 or tPrime == 0.) and "Cannot interpret Ly>1 and t'!=0");
		stringstream tPrimelabel;
		tPrimelabel << "t'=" << tPrime;
		Terms.info.push_back(tPrimelabel.str());
	}
	if(tPrime!=0)
	{
		Terms.nextn.push_back(make_tuple(-tPrime,
										 kroneckerProduct(B.Id(),F.cdag(UP,0)),
										 kroneckerProduct(B.Id(),F.sign()* F.c(UP,0)),
										 kroneckerProduct(B.Id(),F.sign())));
		Terms.nextn.push_back(make_tuple(-tPrime,
										 kroneckerProduct(B.Id(),F.cdag(DN,0)),
										 kroneckerProduct(B.Id(),F.sign()* F.c(DN,0)),
										 kroneckerProduct(B.Id(),F.sign())));
		Terms.nextn.push_back(make_tuple(+tPrime,
										 kroneckerProduct(B.Id(),F.c(UP,0)),
										 kroneckerProduct(B.Id(),F.sign()* F.cdag(UP,0)),
										 kroneckerProduct(B.Id(),F.sign())));
		Terms.nextn.push_back(make_tuple(+tPrime,
										 kroneckerProduct(B.Id(),F.c(DN,0)),
										 kroneckerProduct(B.Id(),F.sign()* F.cdag(DN,0)),
										 kroneckerProduct(B.Id(),F.sign())));
	}
	
		
	// local terms
	
	// Kondo-J
	auto [J,Jorb,Jlabel] = P.fill_array1d<double>("J","Jorb",F.orbitals(),loc);
	Terms.info.push_back(Jlabel);

	// t⟂
	double tPerp = P.get_default<double>("tPerp");
	if (P.HAS("tPerp",loc))
	{
		tPerp = P.get<double>("tPerp",loc);
		stringstream ss;
		ss << "t⟂=" << tPerp;
		Terms.info.push_back(ss.str());
	}
	else if (P.HAS("t",loc))
	{
		tPerp = P.get<double>("t",loc);
	}

	// Hubbard U
	auto [U,Uorb,Ulabel] = P.fill_array1d<double>("U","Uorb",F.orbitals(),loc);
	if(!Ulabel.empty()) {Terms.info.push_back(Ulabel);}

	// mu
	auto [mu,muorb,mulabel] = P.fill_array1d<double>("mu","muorb",F.orbitals(),loc);
	if(!mulabel.empty()) {Terms.info.push_back(mulabel);}

	// K (S_z anisotropy)
	auto [K,Korb,Klabel] = P.fill_array1d<double>("K","Korb",F.orbitals(),loc);
	if(!Klabel.empty()) {Terms.info.push_back(Klabel);}

	// Bz electronic sites
	auto [Bz_elec,Bz_elecorb,Bz_eleclabel] = P.fill_array1d<double>("Bz_elec","Bz_elecorb",F.orbitals(),loc);
	if(!Bz_eleclabel.empty()) {Terms.info.push_back(Bz_eleclabel);}

	// Bz spins
	auto [Bz,Bzorb,Bzlabel] = P.fill_array1d<double>("Bz","Bzorb",F.orbitals(),loc);
	if(!Bzlabel.empty()) {Terms.info.push_back(Bzlabel);}

	// Bx spins
	auto [Bx,Bxorb,Bxlabel] = P.fill_array1d<double>("Bx","Bxorb",F.orbitals(),loc);
	if(!Bxlabel.empty()) {Terms.info.push_back(Bxlabel);}

	auto Hheis = kroneckerProduct(B.HeisenbergHamiltonian(0.,0.,Bzorb,Bxorb,Korb),F.Id());
	auto Hhubb = kroneckerProduct(B.Id(),F.HubbardHamiltonian(Uorb,muorb,Bz_elecorb,tPerp,V,J, P.get<bool>("CYLINDER")));
	auto Hkondo = Hheis + Hhubb;
	for (int i=0; i<F.orbitals(); ++i)
	{
		if(Jorb[i] != 0.)
		{
			Hkondo += -Jorb[i]* kroneckerProduct(B.Scomp(SZ,i),F.Sz(i));
			Hkondo += -Jorb[i]*0.5* kroneckerProduct(B.Scomp(SP,i),F.Sm(i));
			Hkondo += -Jorb[i]*0.5* kroneckerProduct(B.Scomp(SM,i),F.Sp(i));
		}
	}

	Terms.local.push_back(make_tuple(1., Hkondo));
	
	return Terms;
}

} //end namespace VMPS

#endif
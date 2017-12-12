#ifndef STRAWBERRY_KONDOMODEL
#define STRAWBERRY_KONDOMODEL

#include "models/HubbardU1xU1.h"
#include "FermionBase.h"
#include "SpinBase.h"
#include "qarray.h"

namespace VMPS
{

/**MPO representation of 
\f$
H = - \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} -t^{\prime} \sum_{<<ij>>\sigma} c^\dagger_{i\sigma}c_{j\sigma} - J \sum_{i \in I} \mathbf{S}_i \cdot \mathbf{s}_i - \sum_{i \in I} B_i^z S_i^z
\f$.
The set of impurities \f$I\f$ is completely free to choose.
\note \f$J<0\f$ : antiferromagnetic
\note The local magnetic fields act on the impurities only.
\note If nnn-hopping is positive, the GS-energy is lowered.*/
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

	// /**Constructs a Kondo Lattice Model on a N-ladder.
	// \param Lx_input : chain length
	// \param J_input : \f$J\f$
	// \param Ly_input : chain width
	// \param tPrime_input : \f$t^{\prime}\f$ next nearest neighbour (nnn) hopping. \f$t^{\prime}>0\f$ is common sign.
	// \param U_input : \f$U\f$ (local Hubbard interaction)
	// \param Bz_input : \f$B_z\f$
	// \param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	// \param D_input : \f$2S+1\f$ (impurity spin)*/
	// KondoU1xU1 (size_t Lx_input, double J_input=-1., size_t Ly_input=1, double tPrime_input=0.,
	//             double U_input=0., double Bz_input=0., bool CALC_SQUARE=false, size_t D_input=2);

	// /**Constructs a Kondo Impurity Model on a N-ladder (aka a diluted Kondo Model) using initializer lists for the set of impurities.
	// \param Lx_input : chain length
	// \param J_input : \f$J\f$
	// \param imploc_input : list with locations of the impurities
	// \param Bzval_input : list with locations of the local magnetic fields
	// \param Ly_input : chain width
	// \param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	// \param D_input : \f$2S+1\f$ (impurity spin)*/
	// KondoU1xU1 (size_t Lx_input, double J_input, initializer_list<size_t> imploc_input, initializer_list<double> Bzval_input={},
	//             size_t Ly_input=1, bool CALC_SQUARE=true, size_t D_input=2);

	// /**Constructs a Kondo Impurity Model on a N-ladder (aka a diluted Kondo Model) using vectors for the set of impurities.
	// \param Lx_input : chain length
	// \param J_input : \f$J\f$
	// \param imploc_input : list with locations of the impurities
	// \param Bzval_input : list with locations of the local magnetic fields
	// \param Ly_input : chain width
	// \param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	// \param D_input : \f$2S+1\f$ (impurity spin)*/
	// KondoU1xU1 (size_t Lx_input, double J_input, vector<size_t> imploc_input, vector<double> Bzval_input={},
	//             size_t Ly_input=1, bool CALC_SQUARE=true, size_t D_input=2);

	/**Determines the operators of the Hamiltonian. Made static to be called from other classes, e.g. TransverseKondoModel.
	\param F : the FermionBase class where the local Fermion operators are pulled from
	\param S : the SpinBase class where the local Spin operators are pulled from
	\param J : \f$J\f$
	\param Bz : \f$B_{z}\f$
	\param tInter: hopping matrix for hopping from site \f$i\f$ to \f$i+1\f$ from orbital \f$m\f$ to \f$m^{\prime}\f$
	\param tIntra: hopping inside the super site.
	\param Bx: \f$B_{x}\f$
	\param tPrime : \f$t'\f$
	\param U : \f$U\f$
	*/
	// static HamiltonianTermsXd set_operators (const FermionBase &F, const SpinBase &S, 
	//                                          double J, double Bz, MatrixXd tInter, double tIntra, double Bx=0., double tPrime=0., 
	//                                          double U=0., double mu=0., double K=0.);
	// class qarrayIterator;

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
	// double J=-1., Bz=0., t=1., tPrime=0., U=0.;
	// size_t D=2;
	
	// vector<double> Bzval;
	// vector<size_t> imploc;
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
	Terms.info.push_back(tlabel);
	
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
	Terms.info.push_back(Ulabel);

	// mu
	auto [mu,muorb,mulabel] = P.fill_array1d<double>("mu","muorb",F.orbitals(),loc);
	Terms.info.push_back(mulabel);

	// K (S_z anisotropy)
	auto [K,Korb,Klabel] = P.fill_array1d<double>("K","Korb",F.orbitals(),loc);
	Terms.info.push_back(Klabel);

	// Bz electronic sites
	auto [Bz_elec,Bz_elecorb,Bz_eleclabel] = P.fill_array1d<double>("Bz_elec","Bz_elecorb",F.orbitals(),loc);
	Terms.info.push_back(Bz_eleclabel);

	// Bz spins
	auto [Bz,Bzorb,Bzlabel] = P.fill_array1d<double>("Bz","Bzorb",F.orbitals(),loc);
	Terms.info.push_back(Bzlabel);

	// Bx spins
	auto [Bx,Bxorb,Bxlabel] = P.fill_array1d<double>("Bx","Bxorb",F.orbitals(),loc);
	Terms.info.push_back(Bxlabel);

	auto Hheis = kroneckerProduct(B.HeisenbergHamiltonian(0.,0.,Bzorb,Bxorb,Korb),F.Id());
	auto Hhubb = kroneckerProduct(B.Id(),F.HubbardHamiltonian(Uorb,muorb,Bz_elecorb,tPerp,V,J, P.get<bool>("CYLINDER")));
	auto Hkondo = Hheis + Hhubb;
	for (int i=0; i<F.orbitals(); ++i)
	{
		if(Jzloc[i] != 0.) {Hkondo += -Jzloc[i]* kroneckerProduct(B.Scomp(SZ,i),F.Sz(i));}
		if(Jxyloc[i] != 0.)
		{
			Hkondo += -Jxyloc[i]*0.5* kroneckerProduct(B.Scomp(SP,i),F.Sm(i));
			Hkondo += -Jxyloc[i]*0.5* kroneckerProduct(B.Scomp(SM,i),F.Sp(i));
		}
	}

	Terms.local.push_back(make_tuple(1., Hkondo));
	
	return Terms;

	// ArrayXd Jloc(F.orbitals());   Jloc.setZero();
	// ArrayXd Jzloc(F.orbitals());  Jzloc.setZero();
	// ArrayXd Jxyloc(F.orbitals()); Jxyloc.setZero();

	// double J   = P.get_default<double>("J");
	// double Jxy = P.get_default<double>("Jxy");
	// double Jz  = P.get_default<double>("Jz");

	// if (P.HAS("Jloc",loc))
	// {
	// 	Jloc = P.get<double>("Jloc",loc);
	// 	Jxyloc = Jloc;
	// 	Jzloc  = Jloc;
	// 	Terms.name = "Kondo";
	// 	ss << "S=" << print_frac_nice(S) << ",J=" << Jloc.format(CommaInitFmt);
	// }
	// else if (P.HAS("J",loc))
	// {
	// 	J = P.get<double>("J",loc);
	// 	Jloc = J;
	// 	Jxyloc = Jloc;
	// 	Jzloc  = Jloc;
	// 	ss << "S=" << print_frac_nice(S) << ",J=" << J;
	// 	Terms.name = "Kondo";
	// }
	// else if (P.HAS("Jxyloc",loc) or P.HAS("Jzloc",loc))
	// {
	// 	if (P.HAS("Jxyloc",loc))
	// 	{
	// 		Jxyloc = P.get<double>("Jxyloc",loc);
	// 	}
	// 	if (P.HAS("Jzloc",loc))
	// 	{
	// 		Jzloc = P.get<double>("Jzloc",loc);
	// 	}
		
	// 	if      (Jxyloc.matrix().norm() == 0.) {Terms.name = "Kondo-Ising"; ss << "S="    << print_frac_nice(S) << ",J=" << Jzloc.format(CommaInitFmt);}
	// 	else if (Jzloc.matrix().norm()  == 0.) {Terms.name = "Kondo-XX";    ss << "S="    << print_frac_nice(S) << ",J=" << Jxyloc.format(CommaInitFmt);}
	// 	else {Terms.name = "Kondo-XXZ";   ss << "S="    << print_frac_nice(S)
	// 										 << ",Jxy=" << Jxyloc.format(CommaInitFmt) << ",Jz=" << Jzloc.format(CommaInitFmt);}
	// }
	// else if (P.HAS("Jxy",loc) or P.HAS("Jz",loc))
	// {
	// 	if (P.HAS("Jxy",loc))
	// 	{
	// 		Jxyloc = P.get<double>("Jxy",loc);
	// 	}
	// 	if (P.HAS("Jz",loc))
	// 	{
	// 		Jzloc = P.get<double>("Jz",loc);
	// 	}
		
	// 	if      (Jxy == 0.) {Terms.name = "Kondo-Ising"; ss << "S="    << print_frac_nice(S) << ",J=" << Jz;}
	// 	else if (Jz  == 0.) {Terms.name = "Kondo-XX";    ss << "S="    << print_frac_nice(S) << ",J=" << Jxy;}
	// 	else                {Terms.name = "Kondo-XXZ";   ss << "S="    << print_frac_nice(S) << ",Jxy=" << Jxy << ",Jz=" << Jz;}
	// }

	// ArrayXd Uloc(F.orbitals()); Uloc.setZero();
	// double U = P.get_default<double>("U");
	
	// if (P.HAS("Uloc",loc))
	// {
	// 	Uloc = P.get<double>("Uloc",loc);
	// 	ss << ",U=" << Uloc.format(CommaInitFmt);
	// }
	// else if (P.HAS("U",loc))
	// {
	// 	U = P.get<double>("U",loc);
	// 	Uloc = U;
	// 	ss << ",U=" << U;
	// }
	
	
	// // mu
	
	// ArrayXd muloc(F.orbitals()); muloc.setZero();
	// double mu = P.get_default<double>("mu");
	
	// if (P.HAS("muloc",loc))
	// {
	// 	muloc = P.get<double>("muloc",loc);
	// 	ss << ",mu=" << muloc.format(CommaInitFmt);
	// }
	// if (P.HAS("mu",loc))
	// {
	// 	mu = P.get<double>("mu",loc);
	// 	muloc = mu;
	// 	ss << ",mu=" << mu;
	// }
	
	// // K
	
	// ArrayXd Kloc(F.orbitals()); Kloc.setZero();
	// double K = P.get_default<double>("K");
	
	// if (P.HAS("Kloc",loc))
	// {
	// 	Kloc = P.get<double>("Kloc",loc);
	// 	ss << ",K=" << Kloc.format(CommaInitFmt);
	// }
	// if (P.HAS("K",loc))
	// {
	// 	K = P.get<double>("K",loc);
	// 	Kloc = K;
	// 	ss << ",K=" << K;
	// }

	// // Bz elec
	
	// ArrayXd Bzloc_elec(F.orbitals()); Bzloc_elec.setZero();
	// double Bz_elec = P.get_default<double>("Bz_elec");
	
	// if (P.HAS("Bzloc_elec",loc))
	// {
	// 	Bzloc_elec = P.get<double>("Bzloc_elec",loc);
	// 	ss << ",Bz_elec=" << Bzloc_elec.format(CommaInitFmt);
	// }
	// else if (P.HAS("Bz_elec",loc))
	// {
	// 	Bz_elec = P.get<double>("Bz_elec",loc);
	// 	Bzloc_elec = Bz_elec;
	// 	ss << ",Bz_elec=" << Bz_elec;
	// }

	// // Bz spins
	
	// ArrayXd Bzloc(F.orbitals()); Bzloc.setZero();
	// double Bz = P.get_default<double>("Bz");
	
	// if (P.HAS("Bzloc",loc))
	// {
	// 	Bzloc = P.get<double>("Bzloc",loc);
	// 	ss << ",Bz=" << Bzloc.format(CommaInitFmt);
	// }
	// else if (P.HAS("Bz",loc))
	// {
	// 	Bz = P.get<double>("Bz",loc);
	// 	Bzloc = Bz;
	// 	ss << ",Bz=" << Bz;
	// }

	// // Bx spins

	// ArrayXd Bxloc(F.orbitals()); Bxloc.setZero();
	// double Bx = P.get_default<double>("Bx");
	
	// if (P.HAS("Bxloc",loc))
	// {
	// 	Bxloc = P.get<double>("Bxloc",loc);
	// 	ss << ",Bx=" << Bxloc.format(CommaInitFmt);
	// }
	// else if (P.HAS("Bx",loc))
	// {
	// 	Bx = P.get<double>("Bx",loc);
	// 	Bxloc = Bx;
	// 	ss << ",Bx=" << Bx;
	// }

	// Terms.info = ss.str();
}

} //end namespace VMPS

#endif
// HamiltonianTermsXd KondoU1xU1::
// set_operators (const FermionBase &F, const SpinBase &S, double J, double Bz, MatrixXd tInter, double tIntra, double Bx, double tPrime, double U, double mu, double K)
// {
// 	HamiltonianTermsXd Terms;
	
	
// 	SparseMatrixXd KondoHamiltonian(F.dim()*S.dim(), F.dim()*S.dim());
// 	SparseMatrixXd H1(F.dim()*S.dim(), F.dim()*S.dim());
// 	SparseMatrixXd H2(F.dim()*S.dim(), F.dim()*S.dim());
// 	SparseMatrixXd H3(F.dim()*S.dim(), F.dim()*S.dim());
// 	SparseMatrixXd H4(F.dim()*S.dim(), F.dim()*S.dim());
// 	SparseMatrixXd H5(F.dim()*S.dim(), F.dim()*S.dim());
	
// 	SparseMatrixXd IdSpins(S.dim(),S.dim()); IdSpins.setIdentity();
// 	SparseMatrixXd IdElectrons(F.dim(),F.dim()); IdElectrons.setIdentity();
	
// 	//set Hubbard part of Kondo Hamiltonian
// 	std::vector<double> Uvec(F.orbitals()); fill(Uvec.begin(),Uvec.end(),U);
// 	std::vector<double> muvec(F.orbitals()); fill(muvec.begin(),muvec.end(),-mu);
// 	H1 = kroneckerProduct(IdSpins,F.HubbardHamiltonian(Uvec,muvec,tIntra));
	
// 	//set Heisenberg part of Hamiltonian
// 	H2 = kroneckerProduct(S.HeisenbergHamiltonian(0.,0.,Bz,Bx,K),IdElectrons);
	
// 	//set interaction part of Hamiltonian.
// 	for (int i=0; i<F.orbitals(); ++i)
// 	{
// 		H3 += -J* kroneckerProduct(S.Scomp(SZ,i),F.Sz(i));
// 		H4 += -0.5*J* kroneckerProduct(S.Scomp(SP,i),F.Sm(i));
// 		H5 += -0.5*J* kroneckerProduct(S.Scomp(SM,i),F.Sp(i));
// 	}
	
// 	KondoHamiltonian = H1 + H2 + H3 + H4 + H5;
	
// 	//set local interaction
// 	Terms.local.push_back(make_tuple(1.,KondoHamiltonian));
	
// 	//set nearest neighbour term
// 	for (int legI=0; legI<F.orbitals(); ++legI)
// 	for (int legJ=0; legJ<F.orbitals(); ++legJ)
// 	{
// 		if (tInter(legI,legJ) != 0 )
// 		{
// 			Terms.tight.push_back(make_tuple(-tInter(legI,legJ),kroneckerProduct(IdSpins, F.cdag(UP,legI)),
// 											 kroneckerProduct(IdSpins,F.sign()* F.c(UP,legJ))));
// 			Terms.tight.push_back(make_tuple(-tInter(legI,legJ),kroneckerProduct(IdSpins, F.cdag(DN,legI)),
// 											 kroneckerProduct(IdSpins,F.sign()* F.c(DN,legJ))));
// 			Terms.tight.push_back(make_tuple(tInter(legI,legJ),kroneckerProduct(IdSpins, F.c(UP,legI)),
// 											 kroneckerProduct(IdSpins,F.sign()* F.cdag(UP,legJ))));
// 			Terms.tight.push_back(make_tuple(tInter(legI,legJ),kroneckerProduct(IdSpins, F.c(DN,legI)),
// 											 kroneckerProduct(IdSpins,F.sign()* F.cdag(DN,legJ))));
// 		}
// 	}
	
// 	if (tPrime != 0.)
// 	{
// 		//set next nearest neighbour term
// 		Terms.nextn.push_back(make_tuple(-tPrime,
// 		                                      kroneckerProduct(IdSpins,F.cdag(UP,0)),
// 		                                      kroneckerProduct(IdSpins,F.sign()* F.c(UP,0)),
// 		                                      kroneckerProduct(IdSpins,F.sign())));
// 		Terms.nextn.push_back(make_tuple(-tPrime,
// 		                                      kroneckerProduct(IdSpins,F.cdag(DN,0)),
// 		                                      kroneckerProduct(IdSpins,F.sign()* F.c(DN,0)),
// 		                                      kroneckerProduct(IdSpins,F.sign())));
// 		Terms.nextn.push_back(make_tuple(tPrime,
// 		                                      kroneckerProduct(IdSpins,F.c(UP,0)),
// 		                                      kroneckerProduct(IdSpins,F.sign()* F.cdag(UP,0)),
// 		                                      kroneckerProduct(IdSpins,F.sign())));
// 		Terms.nextn.push_back(make_tuple(tPrime,
// 		                                      kroneckerProduct(IdSpins,F.c(DN,0)),
// 		                                      kroneckerProduct(IdSpins,F.sign()* F.cdag(DN,0)),
// 		                                      kroneckerProduct(IdSpins,F.sign())));
// 	}
	
// 	return Terms;
// }

// KondoU1xU1::
// KondoU1xU1 (size_t Lx_input, double J_input, size_t Ly_input, double tPrime_input, double U_input, double Bz_input, bool CALC_SQUARE, size_t D_input)
// :MpoQ<Symmetry> (),
// J(J_input), Bz(Bz_input), tPrime(tPrime_input), U(U_input), D(D_input)
// {
// 	// assign stuff
// 	this->N_sites = Lx_input;
// 	this->N_legs = Ly_input;
// 	this->Qtot = {0,0};
// 	this->qlabel = NMlabel;
// 	this->label = "KondoU1xU1";
// 	this->format = N_halveM;
	
// 	assert(N_legs>1 and tPrime==0. or N_legs==1 and "Cannot build a ladder with t'-hopping!");
	
// 	// initialize member variable imploc
// 	this->imploc.resize(Lx_input);
// 	std::iota(this->imploc.begin(), this->imploc.end(), 0);
	
// 	stringstream ss;
// 	ss << "(J=" << J << ",Bz=" << Bz << ",t'=" << tPrime << ",U=" << U << ")";
// 	this->label += ss.str();
	
// 	F = FermionBase(N_legs);
// 	S = SpinBase(N_legs,D);
	
// 	MatrixXd tInter(N_legs,N_legs); tInter.setIdentity(); // tInter*=-1.;
	
// 	MpoQ<Symmetry>::qloc.resize(N_sites);
// 	for (size_t l=0; l<this->N_sites; ++l)
// 	{
// 		MpoQ<Symmetry>::qloc[l].resize(F.dim()*S.dim());
// 		for (size_t j=0; j<S.dim(); j++)
// 			for (size_t i=0; i<F.dim(); i++)
// 			{
// 				MpoQ<Symmetry>::qloc[l][i+F.dim()*j] = F.qNums(i);
// 				MpoQ<Symmetry>::qloc[l][i+F.dim()*j][1] += S.qNums(j)[0];
// 			}
// 	}
	
// 	HamiltonianTermsXd Terms = set_operators(F,S, J,Bz,tInter,1.,0.,tPrime,U);
// 	SuperMatrix<double> G = ::Generator(Terms);
// 	this->Daux = Terms.auxdim();
	
// 	this->construct(G, this->W, this->Gvec);
	
// 	if (CALC_SQUARE == true)
// 	{
// 		this->construct(tensor_product(G,G), this->Wsq, this->GvecSq);
// 		this->GOT_SQUARE = true;
// 	}
// 	else
// 	{
// 		this->GOT_SQUARE = false;
// 	}
// }

// KondoU1xU1::
// KondoU1xU1 (size_t Lx_input, double J_input, vector<size_t> imploc_input, vector<double> Bzval_input, size_t Ly_input, bool CALC_SQUARE, size_t D_input)
// :MpoQ<Symmetry,double>(), J(J_input), imploc(imploc_input), D(D_input)
// {
// 	// if Bzval_input empty, set it to zero
// 	if (Bzval_input.size() == 0)
// 	{
// 		Bzval.assign(imploc.size(),0.);
// 	}
// 	else
// 	{
// 		assert(imploc_input.size() == Bzval_input.size() and "Impurities and B-fields do not match!");
// 		Bzval = Bzval_input;
// 	}
	
// 	// assign stuff
// 	this->N_sites = Lx_input;
// 	this->N_legs = Ly_input;
// 	this->Qtot = {0,0};
// 	this->qlabel = NMlabel;
// 	this->label = "KondoU1xU1 (impurity)";
// 	this->format = N_halveM;
	
// 	F = FermionBase(N_legs);
// 	S = SpinBase(N_legs,D);
	
// 	MatrixXd tInter(N_legs,N_legs); tInter.setIdentity();// tInter*=-1.;
	
// 	MpoQ<Symmetry,double>::qloc.resize(this->N_sites);
	
// 	// make a pretty label
// 	stringstream ss;
// 	ss << "(S=" << frac(D-1,2) << ",J=" << J << ",imps={";
// 	for (auto i=0; i<imploc.size(); ++i)
// 	{
// 		assert(imploc[i] < this->N_sites and "Invalid impurity location!");
// 		ss << imploc[i];
// 		if (i!=imploc.size()-1) {ss << ",";}
// 	}
// 	ss << "}";
// 	ss << ",Bz={";
// 	for (auto i=0; i<Bzval.size(); ++i)
// 	{
// 		ss << Bzval[i];
// 		if (i!=Bzval.size()-1) {ss << ",";}
// 	}
// 	ss << "})";
// 	this->label += ss.str();
	
// 	// create the SuperMatrices
// 	vector<SuperMatrix<double> > G(this->N_sites);
// 	vector<SuperMatrix<double> > Gsq;
// 	if (CALC_SQUARE == true)
// 	{
// 		Gsq.resize(this->N_sites);
// 	}
	
// 	for (size_t l=0; l<this->N_sites; ++l)
// 	{
// 		auto it = find(imploc.begin(),imploc.end(),l);
// 		// got an impurity
// 		if (it!=imploc.end())
// 		{
// 			MpoQ<Symmetry>::qloc[l].resize(F.dim()*S.dim());
// 			for (size_t s1=0; s1<S.dim(); s1++)
// 			for (size_t s2=0; s2<F.dim(); s2++)
// 			{
// 				MpoQ<Symmetry>::qloc[l][s2+F.dim()*s1] = F.qNums(s2);
// 				MpoQ<Symmetry>::qloc[l][s2+F.dim()*s1][1] += S.qNums(s1)[0];
// 			}
			
// 			size_t i = it-imploc.begin();
// 			if (l==0)
// 			{
// 				HamiltonianTermsXd Terms = set_operators(F,S, J,Bzval[i],tInter,1.,0.,0.,0.);
// 				this->Daux = Terms.auxdim();
// 				G[l].setRowVector(Daux,F.dim()*S.dim());
// 				G[l] = ::Generator(Terms).row(Daux-1);
// 				if (CALC_SQUARE == true)
// 				{
// 					Gsq[l].setRowVector(Daux*Daux,F.dim()*S.dim());
// 					Gsq[l] = tensor_product(G[l],G[l]);
// 				}
// 			}
// 			else if (l==this->N_sites-1)
// 			{
// 				HamiltonianTermsXd Terms = set_operators(F,S, J,Bzval[i],tInter,1.,0.,0.,0.);
// 				this->Daux = Terms.auxdim();
// 				G[l].setColVector(Daux,F.dim()*S.dim());
// 				G[l] = ::Generator(Terms).col(0);
// 				if (CALC_SQUARE == true)
// 				{
// 					Gsq[l].setColVector(Daux*Daux,F.dim()*S.dim());
// 					Gsq[l] = tensor_product(G[l],G[l]);
// 				}
// 			}
// 			else
// 			{
// 				HamiltonianTermsXd Terms = set_operators(F,S, J,Bzval[i],tInter,1.,0.,0.,0.);
// 				this->Daux = Terms.auxdim();
// 				G[l].setMatrix(Daux,F.dim()*S.dim());
// 				G[l] = ::Generator(Terms);
// 				if (CALC_SQUARE == true)
// 				{
// 					Gsq[l].setMatrix(Daux*Daux,F.dim()*S.dim());
// 					Gsq[l] = tensor_product(G[l],G[l]);
// 				}
// 			}
// 		}
// 		// no impurity
// 		else
// 		{
// 			MpoQ<Symmetry>::qloc[l].resize(F.dim());
// 			for (size_t s=0; s<F.dim(); s++)
// 			{
// 				MpoQ<Symmetry>::qloc[l][s] = F.qNums(s);
// 			}
			
// 			if (l==0)
// 			{
// 				HamiltonianTermsXd Terms = HubbardModel::set_operators(F,0.);
// 				this->Daux = Terms.auxdim();
// 				G[l].setRowVector(Daux,F.dim());
// 				G[l] = ::Generator(Terms).row(Daux-1);
// 				if (CALC_SQUARE == true)
// 				{
// 					Gsq[l].setRowVector(Daux*Daux,F.dim());
// 					Gsq[l] = tensor_product(G[l],G[l]);
// 				}
// 			}
// 			else if (l==this->N_sites-1)
// 			{
// 				HamiltonianTermsXd Terms = HubbardModel::set_operators(F,0.);
// 				this->Daux = Terms.auxdim();
// 				G[l].setColVector(Daux,F.dim());
// 				G[l] = ::Generator(Terms).col(0);
// 				if (CALC_SQUARE == true)
// 				{
// 					Gsq[l].setColVector(Daux*Daux,F.dim());
// 					Gsq[l] = tensor_product(G[l],G[l]);
// 				}
// 			}
// 			else
// 			{
// 				HamiltonianTermsXd Terms = HubbardModel::set_operators(F,0.);
// 				this->Daux = Terms.auxdim();
// 				G[l].setMatrix(Daux,F.dim());
// 				G[l] = ::Generator(Terms);
// 				if (CALC_SQUARE == true)
// 				{
// 					Gsq[l].setMatrix(Daux*Daux,F.dim());
// 					Gsq[l] = tensor_product(G[l],G[l]);
// 				}
// 			}
// 		}
// 	}
	
// 	this->construct(G, this->W, this->Gvec);
	
// 	if (CALC_SQUARE == true)
// 	{
// 		this->construct(Gsq, this->Wsq, this->GvecSq);
// 		this->GOT_SQUARE = true;
// 	}
// 	else
// 	{
// 		this->GOT_SQUARE = false;
// 	}
// }

// KondoU1xU1::
// KondoU1xU1 (size_t Lx_input, double J_input, initializer_list<size_t> imploc_input, initializer_list<double> Bzval_input, size_t Ly_input, bool CALC_SQUARE, size_t D_input)
// :KondoU1xU1(Lx_input, J_input, vector<size_t>(begin(imploc_input),end(imploc_input)), vector<double>(begin(Bzval_input),end(Bzval_input)), Ly_input, CALC_SQUARE, D_input)
// {}

// class KondoU1xU1::qarrayIterator
// {
// public:
	
// 	/**
// 	\param qloc_input : vector of local bases
// 	\param l_frst : first site
// 	\param l_last : last site
// 	\param N_legs : Dimension in y-direction
// 	*/
// 	qarrayIterator (const vector<vector<qarray<2> > > &qloc_input, int l_frst, int l_last, size_t N_legs=1)
// 	{
// 		int Nimps = 0;
// 		size_t D = 1;
// 		if (l_last < 0 or l_frst >= qloc_input.size())
// 		{
// 			N_sites = 0;
// 		}
// 		else
// 		{
// 			N_sites = l_last-l_frst+1;
			
// 			// count the impurities between l_frst and l_last
// 			for (size_t l=l_frst; l<=l_last; ++l)
// 			{
// 				if (qloc_input[l].size()/pow(4,N_legs) > 1)
// 				{
// 					Nimps += static_cast<int>(N_legs);
// 					while (qloc_input[l].size() != pow(4,N_legs)*pow(D,N_legs)) {++D;}
// 				}
// 			}
// 		}
		
// 		int Sx2 = static_cast<int>(D-1); // necessary because of size_t		
// 		int N_legsInt = static_cast<int>(N_legs); // necessary because of size_t
		
// 		for (int Sz=-Sx2*Nimps; Sz<=Sx2*Nimps; Sz+=2)
// 		for (int Nup=0; Nup<=N_sites*N_legsInt; ++Nup)
// 		for (int Ndn=0; Ndn<=N_sites*N_legsInt; ++Ndn)
// 		{
// 			qarray<2> q = {Nup+Ndn, Sz+Nup-Ndn};
// 			qarraySet.insert(q);
// 		}
		
// 		it = qarraySet.begin();
// 	};
	
// 	qarray<2> operator*() {return value;}
	
// 	qarrayIterator& operator= (const qarray<2> a) {value=a;}
// 	bool operator!=           (const qarray<2> a) {return value!=a;}
// 	bool operator<=           (const qarray<2> a) {return value<=a;}
// 	bool operator<            (const qarray<2> a) {return value< a;}
	
// 	qarray<2> begin()
// 	{
// 		return *(qarraySet.begin());
// 	}
	
// 	qarray<2> end()
// 	{
// 		return *(qarraySet.end());
// 	}
	
// 	void operator++()
// 	{
// 		++it;
// 		value = *it;
// 	}
	
// //	bool contains (qarray<2> qnum)
// //	{
// //		return (qarraySet.find(qnum)!=qarraySet.end())? true : false;
// //	}
	
// private:
	
// 	qarray<2> value;
	
// 	set<qarray<2> > qarraySet;
// 	set<qarray<2> >::iterator it;
	
// 	int N_sites;
// };

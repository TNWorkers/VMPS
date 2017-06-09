#ifndef STRAWBERRY_KONDOMODEL
#define STRAWBERRY_KONDOMODEL

#include "MpHubbardModel.h"
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
class KondoModel : public MpoQ<Sym::U1xU1<double>,double>
{
typedef Sym::U1xU1<double> Symmetry;

public:
	/**Does nothing.*/
	KondoModel() : MpoQ(){};
	
	/**Constructs a Kondo Lattice Model on a N-ladder.
	\param Lx_input : chain length
	\param J_input : \f$J\f$
	\param Ly_input : chain width
	\param tPrime_input : \f$t^{\prime}\f$ next nearest neighbour (nnn) hopping. \f$t^{\prime}>0\f$ is common sign.
	\param U_input : \f$U\f$ (local Hubbard interaction)
	\param Bz_input : \f$B_z\f$
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	\param D_input : \f$2S+1\f$ (impurity spin)*/
	KondoModel (size_t Lx_input, double J_input=-1., size_t Ly_input=1, double tPrime_input=0.,
	            double U_input=0., double Bz_input=0., bool CALC_SQUARE=false, size_t D_input=2);

	/**Constructs a Kondo Impurity Model on a N-ladder (aka a diluted Kondo Model) using initializer lists for the set of impurities.
	\param Lx_input : chain length
	\param J_input : \f$J\f$
	\param imploc_input : list with locations of the impurities
	\param Bzval_input : list with locations of the local magnetic fields
	\param Ly_input : chain width
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	\param D_input : \f$2S+1\f$ (impurity spin)*/
	KondoModel (size_t Lx_input, double J_input, initializer_list<size_t> imploc_input, initializer_list<double> Bzval_input={},
	            size_t Ly_input=1, bool CALC_SQUARE=true, size_t D_input=2);

	/**Constructs a Kondo Impurity Model on a N-ladder (aka a diluted Kondo Model) using vectors for the set of impurities.
	\param Lx_input : chain length
	\param J_input : \f$J\f$
	\param imploc_input : list with locations of the impurities
	\param Bzval_input : list with locations of the local magnetic fields
	\param Ly_input : chain width
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	\param D_input : \f$2S+1\f$ (impurity spin)*/
	KondoModel (size_t Lx_input, double J_input, vector<size_t> imploc_input, vector<double> Bzval_input={},
	            size_t Ly_input=1, bool CALC_SQUARE=true, size_t D_input=2);

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
	static HamiltonianTermsXd set_operators (const FermionBase &F, const SpinBase &S, 
	                                         double J, double Bz, MatrixXd tInter, double tIntra, double Bx=0., double tPrime=0., 
	                                         double U=0., double mu=0., double K=0.);
	
	/**Makes half-integers in the output for the magnetization quantum number.*/
	static string N_halveM (qarray<2> qnum);
	
	/**Labels the conserved quantum numbers as "N", "M".*/
	static const std::array<string,2> NMlabel;
	
	///@{
	/**Typedef for convenient reference (no need to specify \p Symmetry, \p Scalar all the time).*/
	typedef MpsQ<Symmetry,double>                           StateXd;
	typedef MpsQ<Symmetry,complex<double> >                 StateXcd;
	typedef DmrgSolverQ<Symmetry,KondoModel,double>         Solver;
	typedef MpsQCompressor<Symmetry,double,double>          CompressorXd;
	typedef MpsQCompressor<Symmetry,complex<double>,double> CompressorXcd;
	typedef MpoQ<Symmetry>                                  Operator;
	///@}
	
	class qarrayIterator;
	
	/**Validates whether a given \p qnum is a valid combination of \p N and \p M for the given model.
	\returns \p true if valid, \p false if not*/
	bool validate (qarray<2> qnum) const;

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
	
protected:
	
	double J=-1., Bz=0., t=1., tPrime=0., U=0.;
	size_t D=2;
	
	vector<double> Bzval;
	vector<size_t> imploc;
	FermionBase F;
	SpinBase S;
};

const std::array<string,2> KondoModel::NMlabel{"N","M"};

HamiltonianTermsXd KondoModel::
set_operators (const FermionBase &F, const SpinBase &S, double J, double Bz, MatrixXd tInter, double tIntra, double Bx, double tPrime, double U, double mu, double K)
{
	HamiltonianTermsXd Terms;
	
	
	SparseMatrixXd KondoHamiltonian(F.dim()*S.dim(), F.dim()*S.dim());
	SparseMatrixXd H1(F.dim()*S.dim(), F.dim()*S.dim());
	SparseMatrixXd H2(F.dim()*S.dim(), F.dim()*S.dim());
	SparseMatrixXd H3(F.dim()*S.dim(), F.dim()*S.dim());
	SparseMatrixXd H4(F.dim()*S.dim(), F.dim()*S.dim());
	SparseMatrixXd H5(F.dim()*S.dim(), F.dim()*S.dim());
	
	SparseMatrixXd IdSpins(S.dim(),S.dim()); IdSpins.setIdentity();
	SparseMatrixXd IdElectrons(F.dim(),F.dim()); IdElectrons.setIdentity();
	
	//set Hubbard part of Kondo Hamiltonian
	std::vector<double> Uvec(F.orbitals()); fill(Uvec.begin(),Uvec.end(),U);
	std::vector<double> muvec(F.orbitals()); fill(muvec.begin(),muvec.end(),-mu);
	H1 = kroneckerProduct(IdSpins,F.HubbardHamiltonian(Uvec,muvec,tIntra));
	
	//set Heisenberg part of Hamiltonian
	H2 = kroneckerProduct(S.HeisenbergHamiltonian(0.,0.,Bz,Bx,K),IdElectrons);
	
	//set interaction part of Hamiltonian.
	for (int i=0; i<F.orbitals(); ++i)
	{
		H3 += -J* kroneckerProduct(S.Scomp(SZ,i),F.Sz(i));
		H4 += -0.5*J* kroneckerProduct(S.Scomp(SP,i),F.Sm(i));
		H5 += -0.5*J* kroneckerProduct(S.Scomp(SM,i),F.Sp(i));
	}
	
	KondoHamiltonian = H1 + H2 + H3 + H4 + H5;
	
	//set local interaction
	Terms.local.push_back(make_tuple(1.,KondoHamiltonian));
	
	//set nearest neighbour term
	for (int legI=0; legI<F.orbitals(); ++legI)
	for (int legJ=0; legJ<F.orbitals(); ++legJ)
	{
		if (tInter(legI,legJ) != 0 )
		{
			Terms.tight.push_back(make_tuple(-tInter(legI,legJ),kroneckerProduct(IdSpins, F.cdag(UP,legI)), kroneckerProduct(IdSpins,F.sign()* F.c(UP,legJ))));
			Terms.tight.push_back(make_tuple(-tInter(legI,legJ),kroneckerProduct(IdSpins, F.cdag(DN,legI)), kroneckerProduct(IdSpins,F.sign()* F.c(DN,legJ))));
			Terms.tight.push_back(make_tuple(tInter(legI,legJ),kroneckerProduct(IdSpins, F.c(UP,legI)), kroneckerProduct(IdSpins,F.sign()* F.cdag(UP,legJ))));
			Terms.tight.push_back(make_tuple(tInter(legI,legJ),kroneckerProduct(IdSpins, F.c(DN,legI)), kroneckerProduct(IdSpins,F.sign()* F.cdag(DN,legJ))));
		}
	}
	
	if (tPrime != 0.)
	{
		//set next nearest neighbour term
		Terms.nextn.push_back(make_tuple(-tPrime,
		                                      kroneckerProduct(IdSpins,F.cdag(UP,0)),
		                                      kroneckerProduct(IdSpins,F.sign()* F.c(UP,0)),
		                                      kroneckerProduct(IdSpins,F.sign())));
		Terms.nextn.push_back(make_tuple(-tPrime,
		                                      kroneckerProduct(IdSpins,F.cdag(DN,0)),
		                                      kroneckerProduct(IdSpins,F.sign()* F.c(DN,0)),
		                                      kroneckerProduct(IdSpins,F.sign())));
		Terms.nextn.push_back(make_tuple(tPrime,
		                                      kroneckerProduct(IdSpins,F.c(UP,0)),
		                                      kroneckerProduct(IdSpins,F.sign()* F.cdag(UP,0)),
		                                      kroneckerProduct(IdSpins,F.sign())));
		Terms.nextn.push_back(make_tuple(tPrime,
		                                      kroneckerProduct(IdSpins,F.c(DN,0)),
		                                      kroneckerProduct(IdSpins,F.sign()* F.cdag(DN,0)),
		                                      kroneckerProduct(IdSpins,F.sign())));
	}
	
	return Terms;
}

KondoModel::
KondoModel (size_t Lx_input, double J_input, size_t Ly_input, double tPrime_input, double U_input, double Bz_input, bool CALC_SQUARE, size_t D_input)
:MpoQ<Symmetry> (),
J(J_input), Bz(Bz_input), tPrime(tPrime_input), U(U_input), D(D_input)
{
	// assign stuff
	this->N_sites = Lx_input;
	this->N_legs = Ly_input;
	this->Qtot = {0,0};
	this->qlabel = NMlabel;
	this->label = "KondoModel";
	this->format = N_halveM;
	
	assert(N_legs>1 and tPrime==0. or N_legs==1 and "Cannot build a ladder with t'-hopping!");
	
	// initialize member variable imploc
	this->imploc.resize(Lx_input);
	std::iota(this->imploc.begin(), this->imploc.end(), 0);
	
	stringstream ss;
	ss << "(J=" << J << ",Bz=" << Bz << ",t'=" << tPrime << ",U=" << U << ")";
	this->label += ss.str();
	
	F = FermionBase(N_legs);
	S = SpinBase(N_legs,D);
	
	MatrixXd tInter(N_legs,N_legs); tInter.setIdentity(); // tInter*=-1.;
	
	MpoQ<Symmetry>::qloc.resize(N_sites);
	for (size_t l=0; l<this->N_sites; ++l)
	{
		MpoQ<Symmetry>::qloc[l].resize(F.dim()*S.dim());
		for (size_t j=0; j<S.dim(); j++)
			for (size_t i=0; i<F.dim(); i++)
			{
				MpoQ<Symmetry>::qloc[l][i+F.dim()*j] = F.qNums(i);
				MpoQ<Symmetry>::qloc[l][i+F.dim()*j][1] += S.qNums(j)[0];
			}
	}
	
	HamiltonianTermsXd Terms = set_operators(F,S, J,Bz,tInter,1.,0.,tPrime,U);
	SuperMatrix<double> G = ::Generator(Terms);
	this->Daux = Terms.auxdim();
	
	this->construct(G, this->W, this->Gvec);
	
	if (CALC_SQUARE == true)
	{
		this->construct(tensor_product(G,G), this->Wsq, this->GvecSq);
		this->GOT_SQUARE = true;
	}
	else
	{
		this->GOT_SQUARE = false;
	}
}

KondoModel::
KondoModel (size_t Lx_input, double J_input, vector<size_t> imploc_input, vector<double> Bzval_input, size_t Ly_input, bool CALC_SQUARE, size_t D_input)
:MpoQ<Symmetry,double>(), J(J_input), imploc(imploc_input), D(D_input)
{
	// if Bzval_input empty, set it to zero
	if (Bzval_input.size() == 0)
	{
		Bzval.assign(imploc.size(),0.);
	}
	else
	{
		assert(imploc_input.size() == Bzval_input.size() and "Impurities and B-fields do not match!");
		Bzval = Bzval_input;
	}
	
	// assign stuff
	this->N_sites = Lx_input;
	this->N_legs = Ly_input;
	this->Qtot = {0,0};
	this->qlabel = NMlabel;
	this->label = "KondoModel (impurity)";
	this->format = N_halveM;
	
	F = FermionBase(N_legs);
	S = SpinBase(N_legs,D);
	
	MatrixXd tInter(N_legs,N_legs); tInter.setIdentity();// tInter*=-1.;
	
	MpoQ<Symmetry,double>::qloc.resize(this->N_sites);
	
	// make a pretty label
	stringstream ss;
	ss << "(S=" << frac(D-1,2) << ",J=" << J << ",imps={";
	for (auto i=0; i<imploc.size(); ++i)
	{
		assert(imploc[i] < this->N_sites and "Invalid impurity location!");
		ss << imploc[i];
		if (i!=imploc.size()-1) {ss << ",";}
	}
	ss << "}";
	ss << ",Bz={";
	for (auto i=0; i<Bzval.size(); ++i)
	{
		ss << Bzval[i];
		if (i!=Bzval.size()-1) {ss << ",";}
	}
	ss << "})";
	this->label += ss.str();
	
	// create the SuperMatrices
	vector<SuperMatrix<double> > G(this->N_sites);
	vector<SuperMatrix<double> > Gsq;
	if (CALC_SQUARE == true)
	{
		Gsq.resize(this->N_sites);
	}
	
	for (size_t l=0; l<this->N_sites; ++l)
	{
		auto it = find(imploc.begin(),imploc.end(),l);
		// got an impurity
		if (it!=imploc.end())
		{
			MpoQ<Symmetry>::qloc[l].resize(F.dim()*S.dim());
			for (size_t s1=0; s1<S.dim(); s1++)
			for (size_t s2=0; s2<F.dim(); s2++)
			{
				MpoQ<Symmetry>::qloc[l][s2+F.dim()*s1] = F.qNums(s2);
				MpoQ<Symmetry>::qloc[l][s2+F.dim()*s1][1] += S.qNums(s1)[0];
			}
			
			size_t i = it-imploc.begin();
			if (l==0)
			{
				HamiltonianTermsXd Terms = set_operators(F,S, J,Bzval[i],tInter,1.,0.,0.,0.);
				this->Daux = Terms.auxdim();
				G[l].setRowVector(Daux,F.dim()*S.dim());
				G[l] = ::Generator(Terms).row(Daux-1);
				if (CALC_SQUARE == true)
				{
					Gsq[l].setRowVector(Daux*Daux,F.dim()*S.dim());
					Gsq[l] = tensor_product(G[l],G[l]);
				}
			}
			else if (l==this->N_sites-1)
			{
				HamiltonianTermsXd Terms = set_operators(F,S, J,Bzval[i],tInter,1.,0.,0.,0.);
				this->Daux = Terms.auxdim();
				G[l].setColVector(Daux,F.dim()*S.dim());
				G[l] = ::Generator(Terms).col(0);
				if (CALC_SQUARE == true)
				{
					Gsq[l].setColVector(Daux*Daux,F.dim()*S.dim());
					Gsq[l] = tensor_product(G[l],G[l]);
				}
			}
			else
			{
				HamiltonianTermsXd Terms = set_operators(F,S, J,Bzval[i],tInter,1.,0.,0.,0.);
				this->Daux = Terms.auxdim();
				G[l].setMatrix(Daux,F.dim()*S.dim());
				G[l] = ::Generator(Terms);
				if (CALC_SQUARE == true)
				{
					Gsq[l].setMatrix(Daux*Daux,F.dim()*S.dim());
					Gsq[l] = tensor_product(G[l],G[l]);
				}
			}
		}
		// no impurity
		else
		{
			MpoQ<Symmetry>::qloc[l].resize(F.dim());
			for (size_t s=0; s<F.dim(); s++)
			{
				MpoQ<Symmetry>::qloc[l][s] = F.qNums(s);
			}
			
			if (l==0)
			{
				HamiltonianTermsXd Terms = HubbardModel::set_operators(F,0.);
				this->Daux = Terms.auxdim();
				G[l].setRowVector(Daux,F.dim());
				G[l] = ::Generator(Terms).row(Daux-1);
				if (CALC_SQUARE == true)
				{
					Gsq[l].setRowVector(Daux*Daux,F.dim());
					Gsq[l] = tensor_product(G[l],G[l]);
				}
			}
			else if (l==this->N_sites-1)
			{
				HamiltonianTermsXd Terms = HubbardModel::set_operators(F,0.);
				this->Daux = Terms.auxdim();
				G[l].setColVector(Daux,F.dim());
				G[l] = ::Generator(Terms).col(0);
				if (CALC_SQUARE == true)
				{
					Gsq[l].setColVector(Daux*Daux,F.dim());
					Gsq[l] = tensor_product(G[l],G[l]);
				}
			}
			else
			{
				HamiltonianTermsXd Terms = HubbardModel::set_operators(F,0.);
				this->Daux = Terms.auxdim();
				G[l].setMatrix(Daux,F.dim());
				G[l] = ::Generator(Terms);
				if (CALC_SQUARE == true)
				{
					Gsq[l].setMatrix(Daux*Daux,F.dim());
					Gsq[l] = tensor_product(G[l],G[l]);
				}
			}
		}
	}
	
	this->construct(G, this->W, this->Gvec);
	
	if (CALC_SQUARE == true)
	{
		this->construct(Gsq, this->Wsq, this->GvecSq);
		this->GOT_SQUARE = true;
	}
	else
	{
		this->GOT_SQUARE = false;
	}
}

KondoModel::
KondoModel (size_t Lx_input, double J_input, initializer_list<size_t> imploc_input, initializer_list<double> Bzval_input, size_t Ly_input, bool CALC_SQUARE, size_t D_input)
:KondoModel(Lx_input, J_input, vector<size_t>(begin(imploc_input),end(imploc_input)), vector<double>(begin(Bzval_input),end(Bzval_input)), Ly_input, CALC_SQUARE, D_input)
{}

string KondoModel::
N_halveM (qarray<2> qnum)
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

MpoQ<Sym::U1xU1<double> > KondoModel::
Simp (size_t locx, SPINOP_LABEL SOP, size_t locy)
{
	assert(locx<this->N_sites);
	stringstream ss;
	ss << SOP << "(" << locx << "," << locy << ")";
	MpoQ<Symmetry> Mout(this->N_sites, this->N_legs, locBasis(), {0,0}, KondoModel::NMlabel, ss.str());
	SparseMatrixXd IdSub(F.dim(),F.dim()); IdSub.setIdentity();
	Mout.setLocal(locx, kroneckerProduct(S.Scomp(SOP,locy),IdSub));
	return Mout;
}

MpoQ<Sym::U1xU1<double> > KondoModel::
Ssub (size_t locx, SPINOP_LABEL SOP, size_t locy)
{
	assert(locx<this->N_sites);
	stringstream ss;
	ss << SOP << "(" << locx << "," << locy << ")";
	MpoQ<Symmetry> Mout(this->N_sites, this->N_legs, locBasis(), {0,0}, KondoModel::NMlabel, ss.str());
	SparseMatrixXd IdImp(MpoQ<Symmetry,double>::qloc[locx].size()/F.dim(), MpoQ<Symmetry,double>::qloc[locx].size()/F.dim()); IdImp.setIdentity();
	Mout.setLocal(locx, kroneckerProduct(IdImp, F.Scomp(SOP,locy)));
	return Mout;
}

MpoQ<Sym::U1xU1<double> > KondoModel::
SimpSimp (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y, size_t loc2y)
{
	assert(loc1x<this->N_sites and loc2x<this->N_sites);
	stringstream ss;
	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")";
	MpoQ<Symmetry> Mout(this->N_sites, this->N_legs, locBasis(), {0,0}, KondoModel::NMlabel, ss.str());
	SparseMatrixXd IdSub(F.dim(),F.dim()); IdSub.setIdentity();
	Mout.setLocal({loc1x, loc2x}, {kroneckerProduct(S.Scomp(SOP1,loc1y),IdSub), 
	                               kroneckerProduct(S.Scomp(SOP2,loc2y),IdSub)}
	             );
	return Mout;
}

MpoQ<Sym::U1xU1<double> > KondoModel::
SsubSsub (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y, size_t loc2y)
{
	assert(loc1x<this->N_sites and loc2x<this->N_sites);
	stringstream ss;
	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")";
	MpoQ<Symmetry> Mout(this->N_sites, this->N_legs, locBasis(), {0,0}, KondoModel::NMlabel, ss.str());
	SparseMatrixXd IdImp1(MpoQ<Symmetry>::qloc[loc1x].size()/F.dim(), MpoQ<Symmetry>::qloc[loc1x].size()/F.dim()); IdImp1.setIdentity();
	SparseMatrixXd IdImp2(MpoQ<Symmetry>::qloc[loc2x].size()/F.dim(), MpoQ<Symmetry>::qloc[loc2x].size()/F.dim()); IdImp2.setIdentity();
	Mout.setLocal({loc1x, loc2x}, {kroneckerProduct(IdImp1,F.Scomp(SOP1,loc1y)), 
	                               kroneckerProduct(IdImp2,F.Scomp(SOP2,loc2y))}
	             );
	return Mout;
}

MpoQ<Sym::U1xU1<double> > KondoModel::
SimpSsub (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y, size_t loc2y)
{
	assert(loc1x<this->N_sites and loc2x<this->N_sites);
	stringstream ss;
	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")";
	MpoQ<Symmetry> Mout(this->N_sites, this->N_legs, locBasis(), {0,0}, KondoModel::NMlabel, ss.str());
	SparseMatrixXd IdSub(F.dim(),F.dim()); IdSub.setIdentity();
	SparseMatrixXd IdImp(MpoQ<Symmetry>::qloc[loc2x].size()/F.dim(), MpoQ<Symmetry>::qloc[loc2x].size()/F.dim()); IdImp.setIdentity();
	Mout.setLocal({loc1x, loc2x}, {kroneckerProduct(S.Scomp(SOP1,loc1y),IdSub), 
	                               kroneckerProduct(IdImp,F.Scomp(SOP2,loc2y))}
	             );
	return Mout;
}

MpoQ<Sym::U1xU1<double> > KondoModel::
SimpSsubSimpSimp (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc3x, SPINOP_LABEL SOP3, size_t loc4x, SPINOP_LABEL SOP4,
                  size_t loc1y, size_t loc2y, size_t loc3y, size_t loc4y)
{
	assert(loc1x<this->N_sites and loc2x<this->N_sites and loc3x<this->N_sites and loc4x<this->N_sites);
	stringstream ss;
	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")" <<
	      SOP3 << "(" << loc3x << "," << loc3y << ")" << SOP4 << "(" << loc4x << "," << loc4y << ")";
	MpoQ<Symmetry> Mout(this->N_sites, this->N_legs, locBasis(), {0,0}, KondoModel::NMlabel, ss.str());
	SparseMatrixXd IdSub(F.dim(),F.dim()); IdSub.setIdentity();
	SparseMatrixXd IdImp(MpoQ<Symmetry>::qloc[loc2x].size()/F.dim(), MpoQ<Symmetry>::qloc[loc2x].size()/F.dim()); IdImp.setIdentity();
	Mout.setLocal({loc1x, loc2x, loc3x, loc4x}, {kroneckerProduct(S.Scomp(SOP1,loc1y),IdSub), 
	                                             kroneckerProduct(IdImp,F.Scomp(SOP2,loc2y)),
	                                             kroneckerProduct(S.Scomp(SOP3,loc3y),IdSub),
	                                             kroneckerProduct(S.Scomp(SOP4,loc4y),IdSub)});
	return Mout;
}

MpoQ<Sym::U1xU1<double> > KondoModel::
SimpSsubSimpSsub (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc3x, SPINOP_LABEL SOP3, size_t loc4x, SPINOP_LABEL SOP4,
                  size_t loc1y, size_t loc2y, size_t loc3y, size_t loc4y)
{
	assert(loc1x<this->N_sites and loc2x<this->N_sites and loc3x<this->N_sites and loc4x<this->N_sites);
	stringstream ss;
	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")" <<
	      SOP3 << "(" << loc3x << "," << loc3y << ")" << SOP4 << "(" << loc4x << "," << loc4y << ")";
	MpoQ<Symmetry> Mout(this->N_sites, this->N_legs, locBasis(), {0,0}, KondoModel::NMlabel, ss.str());
	SparseMatrixXd IdSub(F.dim(),F.dim()); IdSub.setIdentity();
	SparseMatrixXd IdImp(MpoQ<Symmetry>::qloc[loc2x].size()/F.dim(), MpoQ<Symmetry>::qloc[loc2x].size()/F.dim()); IdImp.setIdentity();
	Mout.setLocal({loc1x, loc2x, loc3x, loc4x}, {kroneckerProduct(S.Scomp(SOP1,loc1y),IdSub), 
	                                             kroneckerProduct(IdImp,F.Scomp(SOP2,loc2y)),
	                                             kroneckerProduct(S.Scomp(SOP3,loc3y),IdSub),
	                                             kroneckerProduct(IdImp,F.Scomp(SOP4,loc4y))}
	             );
	return Mout;
}

MpoQ<Sym::U1xU1<double> > KondoModel::
c (SPIN_INDEX sigma, size_t locx, size_t locy)
{
	assert(locx<this->N_sites and locy<N_legs);
	stringstream ss;
	ss << "c(" << locx << "," << locy << ",σ=" << sigma << ")";
	qarray<2> qdiff;
	(sigma==UP) ? qdiff = {-1,-1} : qdiff = {-1,+1};
	
	vector<SuperMatrix<double> > M(N_sites);
	for (size_t l=0; l<locx; ++l)
	{
		SparseMatrixXd IdImp(MpoQ<Symmetry>::qloc[l].size()/F.dim(), MpoQ<Symmetry>::qloc[l].size()/F.dim()); IdImp.setIdentity();
		M[l].setMatrix(1,S.dim()*F.dim());
		SparseMatrixXd tmp = kroneckerProduct(IdImp,F.sign());
		M[l](0,0) = tmp;
	}
	SparseMatrixXd IdImp(MpoQ<Symmetry>::qloc[locx].size()/F.dim(), MpoQ<Symmetry>::qloc[locx].size()/F.dim()); IdImp.setIdentity();
	M[locx].setMatrix(1,S.dim()*F.dim());
	SparseMatrixXd tmp = (sigma==UP)? kroneckerProduct(IdImp,F.sign_local(locy)*F.c(UP,locy)) : kroneckerProduct(IdImp,F.sign_local(locy)*F.c(DN,locy));
	M[locx](0,0) = tmp;
	for (size_t l=locx+1; l<N_sites; ++l)
	{
		M[l].setMatrix(1,S.dim()*F.dim());
		M[l](0,0).setIdentity();
	}
	
	return MpoQ<Symmetry>(N_sites, N_legs, M, locBasis(), qdiff, KondoModel::NMlabel, ss.str());
}

MpoQ<Sym::U1xU1<double> > KondoModel::
cdag (SPIN_INDEX sigma, size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << "c†(" << locx << "," << locy << ",σ=" << sigma << ")";
	qarray<2> qdiff;
	(sigma==UP) ? qdiff = {+1,+1} : qdiff = {+1,-1};
	
	vector<SuperMatrix<double> > M(N_sites);
	for (size_t l=0; l<locx; ++l)
	{
		SparseMatrixXd IdImp(MpoQ<Symmetry>::qloc[l].size()/F.dim(), MpoQ<Symmetry>::qloc[l].size()/F.dim()); IdImp.setIdentity();
		M[l].setMatrix(1,S.dim()*F.dim());
		SparseMatrixXd tmp = kroneckerProduct(IdImp,F.sign());
		M[l](0,0) = tmp;
	}
	SparseMatrixXd IdImp(MpoQ<Symmetry>::qloc[locx].size()/F.dim(), MpoQ<Symmetry>::qloc[locx].size()/F.dim()); IdImp.setIdentity();
	M[locx].setMatrix(1,S.dim()*F.dim());
	SparseMatrixXd tmp = (sigma==UP)? kroneckerProduct(IdImp,F.sign_local(locy)*F.cdag(UP,locy)) : kroneckerProduct(IdImp,F.sign_local(locy)*F.cdag(DN,locy));
	M[locx](0,0) = tmp;
	for (size_t l=locx+1; l<N_sites; ++l)
	{
		M[l].setMatrix(1,S.dim()*F.dim());
		M[l](0,0).setIdentity();
	}
	
	return MpoQ<Symmetry>(N_sites, N_legs, M, locBasis(), qdiff, KondoModel::NMlabel, ss.str());
}

MpoQ<Sym::U1xU1<double> > KondoModel::
cdagc (SPIN_INDEX sigma, size_t locx1, size_t locx2, size_t locy1, size_t locy2)
{
	assert(locx1<N_sites and locx2<N_sites and locy1<N_legs and locy2<N_legs);
	stringstream ss;
	ss << "c†(" << locx1 << "," << locy1 << ",σ=" << sigma << ") " << "c(" << locx2 << "," << locy2 << ",σ=" << sigma << ")";
	qarray<2> qdiff = {0,0};

	vector<SuperMatrix<double> > M(N_sites);
	SparseMatrixXd IdImp;
	
	if (locx1 < locx2)
	{
		for (size_t l=0; l<locx1; ++l)
		{
			M[l].setMatrix(1,S.dim()*F.dim());
			M[l](0,0).setIdentity();
			// M[l](0,0) = kroneckerProduct(IdImp,F.sign());
		}
		IdImp.resize(MpoQ<Symmetry>::qloc[locx1].size()/F.dim(), MpoQ<Symmetry>::qloc[locx1].size()/F.dim()); IdImp.setIdentity();
		M[locx1].setMatrix(1,S.dim()*F.dim());
		SparseMatrixXd tmp = (sigma==UP) ? kroneckerProduct(IdImp,F.cdag(UP,locy1)) : kroneckerProduct(IdImp,F.cdag(DN,locy1));
		M[locx1](0,0) = tmp;
		for (size_t l=locx1+1; l<locx2; ++l)
		{
			IdImp.resize(MpoQ<Symmetry>::qloc[l].size()/F.dim(), MpoQ<Symmetry>::qloc[l].size()/F.dim()); IdImp.setIdentity();
			M[l].setMatrix(1,S.dim()*F.dim());
			SparseMatrixXd tmp = kroneckerProduct(IdImp,F.sign());
			M[l](0,0) = tmp;
		}
		IdImp.resize(MpoQ<Symmetry>::qloc[locx2].size()/F.dim(), MpoQ<Symmetry>::qloc[locx2].size()/F.dim()); IdImp.setIdentity();
		M[locx2].setMatrix(1,S.dim()*F.dim());
		tmp = (sigma==UP) ? kroneckerProduct(IdImp,F.sign_local(locy2)*F.c(UP,locy2)) : kroneckerProduct(IdImp,F.sign_local(locy2)*F.c(DN,locy2));
		M[locx2](0,0) = tmp;
		for (size_t l=locx2+1; l<N_sites; ++l)
		{
			M[l].setMatrix(1,S.dim()*F.dim());
			M[l](0,0).setIdentity();
		}
	}
	else if(locx1 > locx2)
	{
		for (size_t l=0; l<locx2; ++l)
		{
			M[l].setMatrix(1,S.dim()*F.dim());
			M[l](0,0).setIdentity();
			// M[l](0,0) = kroneckerProduct(IdImp,F.sign());
		}
		IdImp.resize(MpoQ<Symmetry>::qloc[locx2].size()/F.dim(), MpoQ<Symmetry>::qloc[locx2].size()/F.dim()); IdImp.setIdentity();
		M[locx2].setMatrix(1,S.dim()*F.dim());
		SparseMatrixXd tmp = (sigma==UP) ? kroneckerProduct(IdImp,F.c(UP,locy2)) : kroneckerProduct(IdImp,F.c(DN,locy2));
		M[locx2](0,0) = tmp;
		for (size_t l=locx2+1; l<locx1; ++l)
		{
			IdImp.resize(MpoQ<Symmetry>::qloc[l].size()/F.dim(), MpoQ<Symmetry>::qloc[l].size()/F.dim()); IdImp.setIdentity();
			M[l].setMatrix(1,S.dim()*F.dim());
			SparseMatrixXd tmp = kroneckerProduct(IdImp,F.sign());
			M[l](0,0) = tmp;
		}
		IdImp.resize(MpoQ<Symmetry>::qloc[locx1].size()/F.dim(), MpoQ<Symmetry>::qloc[locx1].size()/F.dim()); IdImp.setIdentity();
		M[locx1].setMatrix(1,S.dim()*F.dim());
		tmp = (sigma==UP) ? kroneckerProduct(IdImp,F.sign_local(locy1)*F.cdag(UP,locy1)) : kroneckerProduct(IdImp,F.sign_local(locy1)*F.cdag(DN,locy1));
		M[locx1](0,0) = tmp;
		for (size_t l=locx1+1; l<N_sites; ++l)
		{
			M[l].setMatrix(1,S.dim()*F.dim());
			M[l](0,0).setIdentity();
		}		
	}
	else if(locx1 == locx2)
	{
		for (size_t l=0; l<locx1; ++l)
		{
			M[l].setMatrix(1,S.dim()*F.dim());
			M[l](0,0).setIdentity();
			// M[l](0,0) = kroneckerProduct(IdImp,F.sign());
		}
		IdImp.resize(MpoQ<Symmetry>::qloc[locx1].size()/F.dim(), MpoQ<Symmetry>::qloc[locx1].size()/F.dim()); IdImp.setIdentity();
		M[locx1].setMatrix(1,S.dim()*F.dim());
		SparseMatrixXd tmp = (sigma==UP) ? kroneckerProduct(IdImp,F.cdag(UP,locy1)*F.sign_local(locy1)*F.c(UP,locy2))
			: kroneckerProduct(IdImp,F.cdag(DN,locy1)*F.sign_local(locy1)*F.c(DN,locy2));
		M[locx1](0,0) = tmp;
		for (size_t l=locx1+1; l<N_sites; ++l)
		{
			M[l].setMatrix(1,S.dim()*F.dim());
			M[l](0,0).setIdentity();
		}		

	
	}
	return MpoQ<Symmetry>(N_sites, N_legs, M, locBasis(), qdiff, KondoModel::NMlabel, ss.str());
}

MpoQ<Sym::U1xU1<double> > KondoModel::
d (size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << "double_occ(" << locx << "," << locy << ")";
	SparseMatrixXd IdImp(MpoQ<Symmetry>::qloc[locx].size()/F.dim(), MpoQ<Symmetry>::qloc[locx].size()/F.dim()); IdImp.setIdentity();
	MpoQ<Symmetry> Mout(N_sites, N_legs, locBasis(), {0,0}, KondoModel::NMlabel, ss.str());
	Mout.setLocal(locx, kroneckerProduct(IdImp,F.d(locy)));
	return Mout;
}

class KondoModel::qarrayIterator
{
public:
	
	/**
	\param qloc_input : vector of local bases
	\param l_frst : first site
	\param l_last : last site
	\param N_legs : Dimension in y-direction
	*/
	qarrayIterator (const vector<vector<qarray<2> > > &qloc_input, int l_frst, int l_last, size_t N_legs=1)
	{
		int Nimps = 0;
		size_t D = 1;
		if (l_last < 0 or l_frst >= qloc_input.size())
		{
			N_sites = 0;
		}
		else
		{
			N_sites = l_last-l_frst+1;
			
			// count the impurities between l_frst and l_last
			for (size_t l=l_frst; l<=l_last; ++l)
			{
				if (qloc_input[l].size()/pow(4,N_legs) > 1)
				{
					Nimps += static_cast<int>(N_legs);
					while (qloc_input[l].size() != pow(4,N_legs)*pow(D,N_legs)) {++D;}
				}
			}
		}
		
		int Sx2 = static_cast<int>(D-1); // necessary because of size_t		
		int N_legsInt = static_cast<int>(N_legs); // necessary because of size_t
		
		for (int Sz=-Sx2*Nimps; Sz<=Sx2*Nimps; Sz+=2)
		for (int Nup=0; Nup<=N_sites*N_legsInt; ++Nup)
		for (int Ndn=0; Ndn<=N_sites*N_legsInt; ++Ndn)
		{
			qarray<2> q = {Nup+Ndn, Sz+Nup-Ndn};
			qarraySet.insert(q);
		}
		
		it = qarraySet.begin();
	};
	
	qarray<2> operator*() {return value;}
	
	qarrayIterator& operator= (const qarray<2> a) {value=a;}
	bool operator!=           (const qarray<2> a) {return value!=a;}
	bool operator<=           (const qarray<2> a) {return value<=a;}
	bool operator<            (const qarray<2> a) {return value< a;}
	
	qarray<2> begin()
	{
		return *(qarraySet.begin());
	}
	
	qarray<2> end()
	{
		return *(qarraySet.end());
	}
	
	void operator++()
	{
		++it;
		value = *it;
	}
	
//	bool contains (qarray<2> qnum)
//	{
//		return (qarraySet.find(qnum)!=qarraySet.end())? true : false;
//	}
	
private:
	
	qarray<2> value;
	
	set<qarray<2> > qarraySet;
	set<qarray<2> >::iterator it;
	
	int N_sites;
};


bool KondoModel::
validate (qarray<2> qnum) const
{
	int Sx2 = static_cast<int>(D-1); // necessary because of size_t
	return (qnum[0]+N_legs*Sx2*imploc.size())%2 == qnum[1]%2;
}

};

#endif

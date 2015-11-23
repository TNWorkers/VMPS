#ifndef STRAWBERRY_KONDOMODEL
#define STRAWBERRY_KONDOMODEL

#include "MpHubbardModel.h"
//#include "MpHeisenbergModel.h"
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
class KondoModel : public MpoQ<2,double>
{
public:
	/**Does nothing.*/
	KondoModel () {};
	
	/**Constructs a Kondo Lattice Model.
	\param Lx_input : chain length
	\param J_input : \f$J\f$
	\param tPrime_input : \f$t^{\prime}\f$ next nearest neighbour (nnn) hopping. \f$t^{\prime}>0\f$ is common sign.
	\param Lx_input : chain width
	\param U_input : \f$U\f$ (local Hubbard interaction)
	\param Bz_input : \f$B_z\f$
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	\param D_input : \f$2S+1\f$ (impurity spin)*/
	KondoModel (size_t Lx_input, double J_input=-1., double tPrime_input=0., size_t Ly_input=1, double U_input=0., double Bz_input=0., bool CALC_SQUARE=true, size_t D_input=2);

	/**Constructs a Kondo Impurity Model (aka a diluted Kondo Model) using initializer lists for the set of impurities.
	\param Lx_input : chain length
	\param J_input : \f$J\f$
	\param imploc_input : list with locations of the impurities
	\param Bzval_input : list with locations of the local magnetic fields
	\param Lx_input : chain width
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	\param D_input : \f$2S+1\f$ (impurity spin)*/
	KondoModel (size_t Lx_input, double J_input, initializer_list<size_t> imploc_input, initializer_list<double> Bzval_input={}, size_t Ly_input=1, bool CALC_SQUARE=true, size_t D_input=2);

	/**Constructs a Kondo Impurity Model (aka a diluted Kondo Model) using vectors for the set of impurities.
	\param Lx_input : chain length
	\param J_input : \f$J\f$
	\param imploc_input : list with locations of the impurities
	\param Bzval_input : list with locations of the local magnetic fields
	\param Lx_input : chain width
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	\param D_input : \f$2S+1\f$ (impurity spin)*/
	KondoModel (size_t Lx_input, double J_input, vector<size_t> imploc_input, vector<double> Bzval_input={}, size_t Ly_input=1, bool CALC_SQUARE=true, size_t D_input=2);

	/**Construct basic MPO stuff for derived models from the Kondo model.*/
	KondoModel (size_t Lx_input, string modelDescription, size_t Ly_input=1, size_t D_input=2);

	static void set_operators (LocalTermsXd &Olocal, TightTermsXd &Otight, NextnTermsXd &Onextn, const FermionBase &F, const SpinBase &S,
							   double J, double Bz, double Bx=0., double t=-1., double tPrime=0., double U=0., size_t D=2);

//	static SuperMatrix<double> Generator (double J, double Bz, double Bx=0., double t=-1., double tPrime=0., double U=0., size_t D=2);
	
	/**Makes half-integers in the output for the magnetization quantum number.*/
	static string N_halveM (qarray<2> qnum);
	
	/**Labels the conserved quantum numbers as "N", "M".*/
	static const std::array<string,2> NMlabel;
	
	/**local basis: \f$\{ \left|0\right>, 
	                      \left|\uparrow\right>, 
	                      \left|\downarrow\right>, 
	                      \left|\uparrow\downarrow\right> 
	                   \}_{electrons}
	                      \otimes
	                   \{
	                      \left|\uparrow\right>,
	                      \left|\downarrow\right>
	                   \}_{spins}
	                \f$.
	The quantum numbers are \f$N=N_{\uparrow}+N_{\downarrow}\f$ and \f$2M=2M_{spins}+N_{\uparrow}-N_{\downarrow}\f$.*/
	static const vector<qarray<2> > qloc (size_t D=2, size_t Ly=1)
	{
		vector<qarray<2> > vout;
		int Sx2 = static_cast<int>(D-1);
		for (int M=Sx2; M>=-Sx2; M-=2)
		{
			vout.push_back(qarray<2>{0,M});
			vout.push_back(qarray<2>{1,M+1});
			vout.push_back(qarray<2>{1,M-1});
			vout.push_back(qarray<2>{2,M});
		}
		return vout;
	}
	
	///@{
	/**Typedef for convenient reference (no need to specify \p Nq, \p Scalar all the time).*/
	typedef MpsQ<2,double>                           StateXd;
	typedef MpsQ<2,complex<double> >                 StateXcd;
	typedef DmrgSolverQ<2,KondoModel>                Solver;
	typedef MpsQCompressor<2,double,double>          CompressorXd;
	typedef MpsQCompressor<2,complex<double>,double> CompressorXcd;
	typedef MpoQ<2>                                  Operator;
	///@}
	
	class qarrayIterator;
	
	/**Validates whether a given \p qnum is a valid combination of \p N and \p M for the given model.
	\returns \p true if valid, \p false if not*/
	bool validate (qarray<2> qnum) const;

	///@{
	MpoQ<2> Simp (size_t L, size_t locx, SPINOP_LABEL SOP, size_t locy=0);
	MpoQ<2> Ssub (size_t L, size_t locx, SPINOP_LABEL SOP, size_t locy=0);
	MpoQ<2> SimpSimp (size_t L, size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y=0, size_t loc2y=0);
	MpoQ<2> SsubSsub (size_t L, size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y=0, size_t loc2y=0);
	MpoQ<2> SimpSsub (size_t L, size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y=0, size_t loc2y=0);
	MpoQ<2> SimpSsubSimpSimp (size_t L, size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, 
							  size_t loc3x, SPINOP_LABEL SOP3, size_t loc4x, SPINOP_LABEL SOP4,
							  size_t loc1y=0, size_t loc2y=0, size_t loc3y=0, size_t loc4y=0);
	MpoQ<2> SimpSsubSimpSsub (size_t L, size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, 
							  size_t loc3x, SPINOP_LABEL SOP3, size_t loc4x, SPINOP_LABEL SOP4,
							  size_t loc1y=0, size_t loc2y=0, size_t loc3y=0, size_t loc4y=0);

	///@}
	
private:
	
	double J=-1., Bz=0., t=-1., tPrime=0., U=0.;
	size_t D=2; size_t Ly=1;
	
	vector<double> Bzval;
	vector<size_t> imploc;
	FermionBase F; SpinBase S;	
};

const std::array<string,2> KondoModel::NMlabel{"N","M"};

void KondoModel::
set_operators (LocalTermsXd &Olocal, TightTermsXd &Otight, NextnTermsXd &Onextn, const FermionBase &F, const SpinBase &S,
			   double J, double Bz, double Bx, double t, double tPrime, double U, size_t D)

{
	//clear old values of operators
	Olocal.resize(0);
	Otight.resize(0);
	Onextn.resize(0);

	SparseMatrixXd KondoHamiltonian(F.dim()*S.dim(),F.dim()*S.dim());
	SparseMatrixXd H1(F.dim()*S.dim(),F.dim()*S.dim());
	SparseMatrixXd H2(F.dim()*S.dim(),F.dim()*S.dim());
	SparseMatrixXd H3(F.dim()*S.dim(),F.dim()*S.dim());
	SparseMatrixXd H4(F.dim()*S.dim(),F.dim()*S.dim());
	SparseMatrixXd H5(F.dim()*S.dim(),F.dim()*S.dim());
	
	SparseMatrixXd IdSpins(S.dim(),S.dim()); IdSpins.setIdentity();
	SparseMatrixXd IdElectrons(F.dim(),F.dim()); IdSpins.setIdentity();

	//set Hubbard part of Kondo Hamiltonian
	H1 = kroneckerProduct(IdSpins,F.HubbardHamiltonian(U,t));

	//set Heisenberg part of Hamiltonian
	H2 = kroneckerProduct(S.HeisenbergHamiltonian(0.,Bz),IdElectrons);

	//set interaction part of Hamiltonian.
	for (int i=0; i<F.orbitals(); ++i)
	{
		H3 += -J* kroneckerProduct(S.Scomp(SZ,i),F.Sz(i));
		H4 += -0.5*J* kroneckerProduct(S.Scomp(SP,i),F.Sm(i));
		H5 += -0.5*J* kroneckerProduct(S.Scomp(SM,i),F.Sp(i));
	}

	KondoHamiltonian = H1 + H2 + H3 + H4 + H5;

	//set local interaction Olocal
	Olocal.push_back(std::make_tuple(1.,KondoHamiltonian));

	//set nearest neighbour term Otight
	for (int leg=0; leg<F.orbitals(); ++leg)
	{
		Otight.push_back(std::make_tuple(-t,kroneckerProduct(IdSpins, F.cdag(UP,leg)), kroneckerProduct(IdSpins, F.sign() * F.c(UP,leg))));
		Otight.push_back(std::make_tuple(-t,kroneckerProduct(IdSpins, F.cdag(DN,leg)), kroneckerProduct(IdSpins, F.sign() * F.c(DN,leg))));
		Otight.push_back(std::make_tuple(t,kroneckerProduct(IdSpins, F.c(UP,leg)), kroneckerProduct(IdSpins, F.sign() * F.cdag(UP,leg))));
		Otight.push_back(std::make_tuple(t,kroneckerProduct(IdSpins, F.c(DN,leg)), kroneckerProduct(IdSpins, F.sign() * F.cdag(DN,leg))));		
	
	}	

	if (tPrime != 0.)
	{
		//set next nearest neighbour term Onextn
		Onextn.push_back(std::make_tuple(-tPrime,
										 kroneckerProduct(IdSpins,F.cdag(UP,0)),
										 kroneckerProduct(IdSpins,F.sign()*F.c(UP,0)),
										 kroneckerProduct(IdSpins,F.sign())));
		Onextn.push_back(std::make_tuple(-tPrime,
										 kroneckerProduct(IdSpins,F.cdag(DN,0)),
										 kroneckerProduct(IdSpins,F.sign()*F.c(DN,0)),
										 kroneckerProduct(IdSpins,F.sign())));
		Onextn.push_back(std::make_tuple(tPrime,
										 kroneckerProduct(IdSpins,F.c(UP,0)),
										 kroneckerProduct(IdSpins,F.sign()*F.cdag(UP,0)),
										 kroneckerProduct(IdSpins,F.sign())));
		Onextn.push_back(std::make_tuple(tPrime,
										 kroneckerProduct(IdSpins,F.c(DN,0)),
										 kroneckerProduct(IdSpins,F.sign()*F.cdag(DN,0)),
										 kroneckerProduct(IdSpins,F.sign())));
		
	}
	
}				

KondoModel::
KondoModel (size_t Lx_input, string modelDescription, size_t Ly_input, size_t D_input)
	:MpoQ<2> (Lx_input, Ly_input,KondoModel::qloc(D_input), {0,0}, KondoModel::NMlabel, modelDescription, N_halveM)
{}
	
KondoModel::
KondoModel (size_t Lx_input, double J_input, double tPrime_input, size_t Ly_input, double U_input, double Bz_input, bool CALC_SQUARE, size_t D_input)
	:MpoQ<2> (Lx_input, Ly_input, KondoModel::qloc(D_input, Ly_input), {0,0}, KondoModel::NMlabel, "KondoModel", N_halveM),
	J(J_input), Bz(Bz_input), tPrime(tPrime_input), U(U_input), D(D_input), Ly(Ly_input)
{
	assert(N_legs>1 and tPrime==0. or N_legs==1 and "Cannot build a ladder with t'-hopping!");

	// initialize member variable imploc
	this->imploc.resize(Lx_input);
	std::iota(this->imploc.begin(), this->imploc.end(), 0);

	stringstream ss;
	ss << "(J=" << J << ",Bz=" << Bz << ",t'=" << tPrime << ",U=" << U << ")";
	this->label += ss.str();

	this->N_legs = Ly_input;
	
	F = FermionBase(Ly);
	S = SpinBase(Ly,D);

	for (size_t l=0; l<this->N_sites; ++l)
	{
		MpoQ<2>::qloc[l].resize(F.dim()*S.dim());
		for (size_t j=0; j<S.dim(); j++)
			for (size_t i=0; i<F.dim(); i++)
			{
				MpoQ<2>::qloc[l][i+F.dim()*j] = ::qvacuum<2>();
				MpoQ<2>::qloc[l][i+F.dim()*j] = F.qNums(i);
				MpoQ<2>::qloc[l][i+F.dim()*j][1] += S.qNums(j)[0];
			}
	}

	set_operators(Olocal, Otight, Onextn, F, S, J, Bz, 0., -1., tPrime, U, D);
	this->Daux = 2 + Otight.size() + 2*Onextn.size();
	
	SuperMatrix<double> G = ::Generator(Olocal, Otight, Onextn);
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
:MpoQ<2,double>(), J(J_input), imploc(imploc_input), D(D_input)
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
	this->Qtot = {0,0};
	this->qlabel = NMlabel;
	this->label = "KondoModel";
	this->format = N_halveM;

	F = FermionBase(Ly);
	S = SpinBase(Ly,D);

	MpoQ<2,double>::qloc.resize(this->N_sites);
	
	// make a pretty label
	stringstream ss;
	ss << "(S=" << ",J=" << J << ",imps={"; //frac(D-1,2) <<
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
			MpoQ<2,double>::qloc[l] = qloc(D);
			
			size_t i = it-imploc.begin();
			if (l==0)
			{
				G[l].setRowVector(6,8);
				set_operators(Olocal, Otight, Onextn, F,S,J,Bzval[i],0.,-1.,0.,0.,D);
				this->Daux = 2 + Otight.size() + 2*Onextn.size();
				G[l] = ::Generator(this->Olocal,this->Otight,this->Onextn).row(5);
				if (CALC_SQUARE == true)
				{
					Gsq[l].setRowVector(6*6,8);
					Gsq[l] = tensor_product(G[l],G[l]);
				}
			}
			else if (l==this->N_sites-1)
			{
				G[l].setColVector(6,8);
				set_operators(Olocal, Otight, Onextn, F,S,J,Bzval[i],0.,-1.,0.,0.,D);
				G[l] = ::Generator(this->Olocal,this->Otight,this->Onextn).col(0);
				if (CALC_SQUARE == true)
				{
					Gsq[l].setColVector(6*6,8);
					Gsq[l] = tensor_product(G[l],G[l]);
				}
			}
			else
			{
				G[l].setMatrix(6,8);
				set_operators(Olocal, Otight, Onextn, F,S,J,Bzval[i],0.,-1.,0.,0.,D);
				G[l] = ::Generator(this->Olocal,this->Otight,this->Onextn);
				if (CALC_SQUARE == true)
				{
					Gsq[l].setMatrix(6*6,8);
					Gsq[l] = tensor_product(G[l],G[l]);
				}
			}
		}
		// no impurity
		else
		{
			MpoQ<2,double>::qloc[l] = vector<qarray<2> >(begin(HubbardModel::qlocNM),end(HubbardModel::qlocNM));
			
			if (l==0)
			{
				G[l].setRowVector(6,4);
				HubbardModel::set_operators(Olocal, Otight, Onextn, F,0.);
				G[l] = ::Generator(this->Olocal,this->Otight,this->Onextn).row(5);
				if (CALC_SQUARE == true)
				{
					Gsq[l].setRowVector(6*6,4);
					Gsq[l] = tensor_product(G[l],G[l]);
				}
			}
			else if (l==this->N_sites-1)
			{
				G[l].setColVector(6,4);
				HubbardModel::set_operators(Olocal, Otight, Onextn, F,0.);
				G[l] = ::Generator(this->Olocal,this->Otight,this->Onextn).col(0);
				if (CALC_SQUARE == true)
				{
					Gsq[l].setColVector(6*6,4);
					Gsq[l] = tensor_product(G[l],G[l]);
				}
			}
			else
			{
				G[l].setMatrix(6,4);
				HubbardModel::set_operators(Olocal, Otight, Onextn, F,0.);
				G[l] = ::Generator(this->Olocal,this->Otight,this->Onextn);
				if (CALC_SQUARE == true)
				{
					Gsq[l].setMatrix(6*6,4);
					Gsq[l] = tensor_product(G[l],G[l]);
				}
			}
		}

		//clear old values of operators
		this->Olocal.resize(0);
		this->Otight.resize(0);
		this->Onextn.resize(0);
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
	:KondoModel(Lx_input, J_input, vector<size_t>(begin(imploc_input),end(imploc_input)), vector<double>(begin(Bzval_input),end(Bzval_input)), CALC_SQUARE, D_input, Ly_input)
{}

string KondoModel::
N_halveM (qarray<2> qnum)
{
	stringstream ss;
	ss << "(" << qnum[0] << ",";
	
	qarray<1> mag;
	mag[0] = qnum[1];
	string halfmag = "test"; //HeisenbergModel::halve(mag);
	halfmag.erase(0,1);
	ss << halfmag;
	
	return ss.str();
}

MpoQ<2> KondoModel::
Simp (size_t L, size_t locx, SPINOP_LABEL SOP, size_t locy)
{
	assert(locx<L);
	stringstream ss;
	ss << SOP << "(" << locx << "," << locy << ")";
	MpoQ<2> Mout(L, this->N_legs, locBasis(), {0,0}, KondoModel::NMlabel, ss.str());
	MatrixXd Id4(F.dim(),F.dim()); Id4.setIdentity();
	Mout.setLocal(locx, kroneckerProduct(S.Scomp(SOP,locy),Id4));
	return Mout;
}

MpoQ<2> KondoModel::
Ssub (size_t L, size_t locx, SPINOP_LABEL SOP, size_t locy)
{
	assert(locx<L);
	stringstream ss;
	ss << SOP << "(" << locx << "," << locy << ")";
	MpoQ<2> Mout(L, this->N_legs, locBasis(), {0,0}, KondoModel::NMlabel, ss.str());
	MatrixXd IdImp(MpoQ<2,double>::qloc[locx].size()/4, MpoQ<2,double>::qloc[locx].size()/4); IdImp.setIdentity();
	Mout.setLocal(locx, kroneckerProduct(IdImp, F.Scomp(SOP,locy)));
	return Mout;
}

MpoQ<2> KondoModel::
SimpSimp (size_t L, size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y, size_t loc2y)
{
	assert(loc1x<L and loc2x<L);
	stringstream ss;
	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")";
	MpoQ<2> Mout(L, this->N_legs, locBasis(), {0,0}, KondoModel::NMlabel, ss.str());
	MatrixXd Id4(F.dim(),F.dim()); Id4.setIdentity();
	Mout.setLocal({loc1x, loc2x}, {kroneckerProduct(S.Scomp(SOP1,loc1y),Id4), 
	                             kroneckerProduct(S.Scomp(SOP2,loc2y),Id4)}
	             );
	return Mout;
}


MpoQ<2> KondoModel::
SsubSsub (size_t L, size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y, size_t loc2y)
{
	assert(loc1x<L and loc2x<L);
	stringstream ss;
	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")";
	MpoQ<2> Mout(L, this->N_legs, locBasis(), {0,0}, KondoModel::NMlabel, ss.str());
	MatrixXd IdImp1(MpoQ<2>::qloc[loc1x].size()/4, MpoQ<2>::qloc[loc1x].size()/4); IdImp1.setIdentity();
	MatrixXd IdImp2(MpoQ<2>::qloc[loc2x].size()/4, MpoQ<2>::qloc[loc2x].size()/4); IdImp2.setIdentity();
	Mout.setLocal({loc1x, loc2x}, {kroneckerProduct(IdImp1,F.Scomp(SOP1,loc1y)), 
				kroneckerProduct(IdImp2,F.Scomp(SOP2,loc2y))}
	             );
	return Mout;
}

MpoQ<2> KondoModel::
SimpSsub (size_t L, size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y, size_t loc2y)
{
	assert(loc1x<L and loc2x<L);
	stringstream ss;
	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")";
	MpoQ<2> Mout(L, this->N_legs, locBasis(), {0,0}, KondoModel::NMlabel, ss.str());
	MatrixXd Id4(F.dim(),F.dim()); Id4.setIdentity();
	MatrixXd IdImp(MpoQ<2>::qloc[loc2x].size()/4, MpoQ<2>::qloc[loc2x].size()/4); IdImp.setIdentity();
	Mout.setLocal({loc1x, loc2x}, {kroneckerProduct(S.Scomp(SOP1,loc1y),Id4), 
				kroneckerProduct(IdImp,F.Scomp(SOP2,loc2y))}
	             );
	return Mout;
}

MpoQ<2> KondoModel::
SimpSsubSimpSimp (size_t L, size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc3x, SPINOP_LABEL SOP3, size_t loc4x, SPINOP_LABEL SOP4,
				  size_t loc1y, size_t loc2y, size_t loc3y, size_t loc4y)
{
	assert(loc1x<L and loc2x<L and loc3x<L and loc4x<L);
	stringstream ss;
	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")" <<
		SOP3 << "(" << loc3x << "," << loc3y << ")" << SOP4 << "(" << loc4x << "," << loc4y << ")";	
	MpoQ<2> Mout(L, this->N_legs, locBasis(), {0,0}, KondoModel::NMlabel, ss.str());
	MatrixXd Id4(F.dim(),F.dim()); Id4.setIdentity();
	MatrixXd IdImp(MpoQ<2>::qloc[loc2x].size()/4, MpoQ<2>::qloc[loc2x].size()/4); IdImp.setIdentity();
	Mout.setLocal({loc1x, loc2x, loc3x, loc4x}, {kroneckerProduct(S.Scomp(SOP1,loc1y),Id4), 
				kroneckerProduct(IdImp,F.Scomp(SOP2,loc2y)),
				kroneckerProduct(S.Scomp(SOP3,loc3y),Id4),
				kroneckerProduct(S.Scomp(SOP4,loc4y),Id4)});
	return Mout;
}

MpoQ<2> KondoModel::
SimpSsubSimpSsub (size_t L, size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc3x, SPINOP_LABEL SOP3, size_t loc4x, SPINOP_LABEL SOP4,
				  size_t loc1y, size_t loc2y, size_t loc3y, size_t loc4y)
{
	assert(loc1x<L and loc2x<L and loc3x<L and loc4x<L);
	stringstream ss;
	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")" <<
		SOP3 << "(" << loc3x << "," << loc3y << ")" << SOP4 << "(" << loc4x << "," << loc4y << ")";
	MpoQ<2> Mout(L, this->N_legs, locBasis(), {0,0}, KondoModel::NMlabel, ss.str());
	MatrixXd Id4(F.dim(),F.dim()); Id4.setIdentity();
	MatrixXd IdImp(MpoQ<2>::qloc[loc2x].size()/4, MpoQ<2>::qloc[loc2x].size()/4); IdImp.setIdentity();
	Mout.setLocal({loc1x, loc2x, loc3x, loc4x}, {kroneckerProduct(S.Scomp(SOP1,loc1y),Id4), 
				kroneckerProduct(IdImp,F.Scomp(SOP2,loc2y)),
				kroneckerProduct(S.Scomp(SOP3,loc3y),Id4),
				kroneckerProduct(IdImp,F.Scomp(SOP4,loc4y))}
		);
	return Mout;
}

class KondoModel::qarrayIterator
{
public:
	
	/**
	\param qloc_input : vector of local bases
	\param l_frst : first site
	\param l_last : last site
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
					++Nimps;
					D = static_cast<size_t>(pow(qloc_input[l].size(),1./N_legs))/4;
				}
			}
		}

		int Sx2 = static_cast<int>(D-1); // necessary because of size_t
		int N_legsInt = static_cast<int>(N_legs); // necessary because of size_t
		for (int Sz=-Sx2*Nimps*N_legsInt; Sz<=Sx2*Nimps*N_legsInt; Sz+=2)
		for (int Nup=0; Nup<=N_sites*N_legsInt; ++Nup)
		for (int Ndn=0; Ndn<=N_sites*N_legsInt; ++Ndn)
		{
			qarray<2> q = {Nup+Ndn, Sz+Nup-Ndn};
			qarraySet.insert(q);
//			cout << q << endl;
		}
//		cout << endl;
		
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
	return (qnum[0]+Sx2*imploc.size())%2 == qnum[1]%2;
}

};

#endif

#ifndef STRAWBERRY_KONDOMODEL
#define STRAWBERRY_KONDOMODEL

#include "MpHubbardModel.h"
#include "MpHeisenbergModel.h"

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
template<size_t D=2>
class KondoModel : public MpoQ<2,double>
{
public:
	
	/**Constructs a Kondo Lattice Model.
	\param L_input : chain length
	\param J_input : \f$J\f$
	\param tPrime_input : \f$t^{\prime}\f$ next nearest neighbour (nnn) hopping. \f$t^{\prime}>0\f$ is common sign.
	\param U_input : \f$U\f$ (local Hubbard interaction)
	\param Bz_input : \f$B_z\f$
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$*/
	KondoModel (size_t L_input, double J_input=-1., double tPrime_input=0., double U_input=0., double Bz_input=0., bool CALC_SQUARE=true);

	/**Constructs a Kondo Impurity Model (aka a diluted Kondo Model) using initializer lists for the set of impurities.
	\param L_input : chain length
	\param J_input : \f$J\f$
	\param imploc_input : list with locations of the impurities
	\param Bzval_input : list with locations of the local magnetic fields
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$*/
	KondoModel (size_t L_input, double J_input, initializer_list<size_t> imploc_input, initializer_list<double> Bzval_input={}, bool CALC_SQUARE=true);

	/**Constructs a Kondo Impurity Model (aka a diluted Kondo Model) using vectors for the set of impurities.
	\param L_input : chain length
	\param J_input : \f$J\f$
	\param imploc_input : list with locations of the impurities
	\param Bzval_input : list with locations of the local magnetic fields
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$*/
	KondoModel (size_t L_input, double J_input, vector<size_t> imploc_input, vector<double> Bzval_input={}, bool CALC_SQUARE=true);

	/**Constructs a Kondo Lattice Model with partial Kondo screening (PKS) geometry.
	   \param L_input : chain length
	   \param Bzval_input : \f$B^z_i\f$ (site dependent magnetic field acting on the impurities)
	   \param J_input : \f$J\f$
	   \param t_input : \f$t\f$. \f$t>0\f$ is the common sign.  
	   \param tPrime_input : \f$t^{\prime}\f$ nnn hopping (\f$t^{\prime}>0\f$ is common sign).
	   \param U_input : \f$U\f$ (local Hubbard interaction)
	   \param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$*/
	KondoModel (size_t L_input, vector<double> Bzval_input={}, double J_input=-1., double t_input=-1., double tPrime_input=-1., double U_input=0., bool CALC_SQUARE=true);

	static SuperMatrix<double> Generator (double J, double Bz, double Bx=0., double t=-1., double tPrime=0., double U=0.);
	
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
	static const std::array<qarray<2>,4*D> q;
	
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
	MpoQ<2> Simp (size_t L, size_t loc, SPINOP_LABEL SOP);
	MpoQ<2> Ssub (size_t L, size_t loc, SPINOP_LABEL SOP);
	MpoQ<2> SimpSimp (size_t L, size_t loc1, SPINOP_LABEL SOP1, size_t loc2, SPINOP_LABEL SOP2);
	MpoQ<2> SsubSsub (size_t L, size_t loc1, SPINOP_LABEL SOP1, size_t loc2, SPINOP_LABEL SOP2);
	MpoQ<2> SimpSsub (size_t L, size_t loc1, SPINOP_LABEL SOP1, size_t loc2, SPINOP_LABEL SOP2);
	///@}
	
private:
	
	double J, Bz, t, tPrime, U;
	
	vector<double> Bzval;
	vector<size_t> imploc;
};

template<>
const std::array<qarray<2>,8> KondoModel<2>::q
{
	// Mimp = +1
	qarray<2>{0,+1},
	qarray<2>{1,+2},
	qarray<2>{1, 0},
	qarray<2>{2,+1},
	// Mimp = -1
	qarray<2>{0,-1},
	qarray<2>{1, 0},
	qarray<2>{1,-2},
	qarray<2>{2,-1}
};

template<>
const std::array<qarray<2>,12> KondoModel<3>::q
{
	// Mimp = +2
	qarray<2>{0,+2},
	qarray<2>{1,+3},
	qarray<2>{1,+1},
	qarray<2>{2,+2},
	// Mimp = 0
	qarray<2>{0, 0},
	qarray<2>{1,+1},
	qarray<2>{1,-1},
	qarray<2>{2, 0},
	// Mimp = -2
	qarray<2>{0,-2},
	qarray<2>{1,-1},
	qarray<2>{1,-3},
	qarray<2>{2,-2}
};

template<>
const std::array<qarray<2>,16> KondoModel<4>::q
{
	// Mimp = +3
	qarray<2>{0,+3},
	qarray<2>{1,+4},
	qarray<2>{1,+2},
	qarray<2>{2,+3},
	// Mimp = +1
	qarray<2>{0,+1},
	qarray<2>{1,+2},
	qarray<2>{1, 0},
	qarray<2>{2,+1},
	// Mimp = -1
	qarray<2>{0,-1},
	qarray<2>{1, 0},
	qarray<2>{1,-2},
	qarray<2>{2,-1},
	// Mimp = -3
	qarray<2>{0,-3},
	qarray<2>{1,-2},
	qarray<2>{1,-4},
	qarray<2>{2,-3}
};

template<size_t D> const std::array<string,2> KondoModel<D>::NMlabel{"N","M"};

template<size_t D>
SuperMatrix<double> KondoModel<D>::
Generator (double J, double Bz, double Bx, double t, double tPrime, double U)
{
	SuperMatrix<double> G;
	if (tPrime==0.) {
		size_t Daux = 6;
		G.setMatrix(Daux,D*4);
		G.setZero();
	
		MatrixXd Id4(4,4); Id4.setIdentity();
		MatrixXd IdSpins(D,D); IdSpins.setIdentity();
	
		G(0,0).setIdentity();
		G(1,0) = kroneckerProduct(IdSpins, t*HubbardModel::cUP.transpose());
		G(2,0) = kroneckerProduct(IdSpins, t*HubbardModel::cDN.transpose());
		G(3,0) = kroneckerProduct(IdSpins, t*HubbardModel::cUP);
		G(4,0) = kroneckerProduct(IdSpins, t*HubbardModel::cDN);
	
		G(5,0) = -0.5*J * kroneckerProduct(SpinBase<D>::Sp,        HubbardModel::Sp.transpose())
			-0.5*J * kroneckerProduct(SpinBase<D>::Sp.transpose(), HubbardModel::Sp)
			-J *     kroneckerProduct(SpinBase<D>::Sz,             HubbardModel::Sz)
			-Bz *    kroneckerProduct(SpinBase<D>::Sz,             Id4)
			-Bx *    kroneckerProduct(SpinBase<D>::Sx,             Id4)
			+U*kroneckerProduct(IdSpins, HubbardModel::nUP_nDN);
	
		// note: fsign takes care of the fermionic sign
		G(5,1) = kroneckerProduct(IdSpins,-HubbardModel::fsign * HubbardModel::cUP);
		G(5,2) = kroneckerProduct(IdSpins,-HubbardModel::fsign * HubbardModel::cDN);
		G(5,3) = kroneckerProduct(IdSpins, HubbardModel::fsign * HubbardModel::cUP.transpose());
		G(5,4) = kroneckerProduct(IdSpins, HubbardModel::fsign * HubbardModel::cDN.transpose());
		G(5,5).setIdentity();
	}
	else
	{
		size_t Daux = 14;
		G.setMatrix(Daux,D*4);
		G.setZero();

		MatrixXd Id4(4,4) ; Id4.setIdentity();
		MatrixXd IdSpins(D,D); IdSpins.setIdentity();

		G(0,0).setIdentity();
		G(1,0) = kroneckerProduct(IdSpins, tPrime*HubbardModel::cDN);
		G(2,0) = kroneckerProduct(IdSpins, tPrime*HubbardModel::cUP);
		G(3,0) = kroneckerProduct(IdSpins, tPrime*HubbardModel::cDN.transpose());
		G(4,0) = kroneckerProduct(IdSpins, tPrime*HubbardModel::cUP.transpose());
		G(5,0) = kroneckerProduct(IdSpins, t*HubbardModel::cUP.transpose());
		G(6,0) = kroneckerProduct(IdSpins, t*HubbardModel::cDN.transpose());
		G(7,0) = kroneckerProduct(IdSpins, t*HubbardModel::cUP);
		G(8,0) = kroneckerProduct(IdSpins, t*HubbardModel::cDN);

		G(13,0) = -0.5*J * kroneckerProduct(SpinBase<D>::Sp,        HubbardModel::Sp.transpose())
			-0.5*J * kroneckerProduct(SpinBase<D>::Sp.transpose(), HubbardModel::Sp)
			-J *     kroneckerProduct(SpinBase<D>::Sz,             HubbardModel::Sz)
			-Bz *    kroneckerProduct(SpinBase<D>::Sz,             Id4)
			-Bx *    kroneckerProduct(SpinBase<D>::Sx,             Id4)
			+U*kroneckerProduct(IdSpins, HubbardModel::nUP_nDN);

		G(13,5) = kroneckerProduct(IdSpins,-HubbardModel::fsign * HubbardModel::cUP);
		G(13,6) = kroneckerProduct(IdSpins,-HubbardModel::fsign * HubbardModel::cDN);
		G(13,7) = kroneckerProduct(IdSpins, HubbardModel::fsign * HubbardModel::cUP.transpose());
		G(13,8) = kroneckerProduct(IdSpins, HubbardModel::fsign * HubbardModel::cDN.transpose());
		G(13,9) = tPrime*kroneckerProduct(IdSpins, -HubbardModel::fsign * HubbardModel::cUP);
		G(13,10) = tPrime*kroneckerProduct(IdSpins,-HubbardModel::fsign * HubbardModel::cDN);
		G(13,11) = tPrime*kroneckerProduct(IdSpins, HubbardModel::fsign * HubbardModel::cUP.transpose());
		G(13,12) = tPrime*kroneckerProduct(IdSpins, HubbardModel::fsign * HubbardModel::cDN.transpose());

		G(9,4) = kroneckerProduct(IdSpins, HubbardModel::fsign);
		G(10,3) = kroneckerProduct(IdSpins, HubbardModel::fsign);
		G(11,2) = kroneckerProduct(IdSpins, HubbardModel::fsign);
		G(12,1) = kroneckerProduct(IdSpins, HubbardModel::fsign);
		G(13,13).setIdentity();
	}

	return G;
}

template<size_t D>
KondoModel<D>::
KondoModel (size_t L_input, double J_input, double tPrime_input, double U_input, double Bz_input, bool CALC_SQUARE)
:MpoQ<2> (L_input, vector<qarray<2> >(begin(KondoModel<D>::q),end(KondoModel<D>::q)), {0,0}, KondoModel<D>::NMlabel, "KondoModel", N_halveM),
	J(J_input), Bz(Bz_input), tPrime(tPrime_input), U(U_input)
{
	// initialize member variable imploc
	this->imploc.resize(L_input);
	std::iota(this->imploc.begin(), this->imploc.end(), 0);

	stringstream ss;
	ss << "(J=" << J << ",Bz=" << Bz << ",tPrime=" << tPrime << ",U=" << U << ")";
	this->label += ss.str();

	this->Daux = (tPrime==0.)? 6 : 14;
	
	SuperMatrix<double> G = Generator(J, Bz, 0., -1., tPrime, U);
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

template<size_t D>
KondoModel<D>::
KondoModel (size_t L_input, double J_input, vector<size_t> imploc_input, vector<double> Bzval_input, bool CALC_SQUARE)
:MpoQ<2,double>(), J(J_input), imploc(imploc_input)
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
	this->N_sites = L_input;
	this->Qtot = {0,0};
	this->qlabel = NMlabel;
	this->label = "KondoModel";
	this->format = N_halveM;
	this->Daux = 6;
	this->qloc.resize(this->N_sites);
	
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
			this->qloc[l] = vector<qarray<2> >(begin(q),end(q));
			
			size_t i = it-imploc.begin();
			if (l==0)
			{
				G[l].setRowVector(6,8);
				G[l] = Generator(J,Bzval[i],0.).row(5);
				if (CALC_SQUARE == true)
				{
					Gsq[l].setRowVector(6*6,8);
					Gsq[l] = tensor_product(G[l],G[l]);
				}
			}
			else if (l==this->N_sites-1)
			{
				G[l].setColVector(6,8);
				G[l] = Generator(J,Bzval[i],0.).col(0);
				if (CALC_SQUARE == true)
				{
					Gsq[l].setColVector(6*6,8);
					Gsq[l] = tensor_product(G[l],G[l]);
				}
			}
			else
			{
				G[l].setMatrix(6,8);
				G[l] = Generator(J,Bzval[i],0.);
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
			this->qloc[l] = vector<qarray<2> >(begin(HubbardModel::qlocNM),end(HubbardModel::qlocNM));
			
			if (l==0)
			{
				G[l].setRowVector(6,4);
				G[l] = HubbardModel::Generator(0,0).row(5);
				if (CALC_SQUARE == true)
				{
					Gsq[l].setRowVector(6*6,4);
					Gsq[l] = tensor_product(G[l],G[l]);
				}
			}
			else if (l==this->N_sites-1)
			{
				G[l].setColVector(6,4);
				G[l] = HubbardModel::Generator(0,0).col(0);
				if (CALC_SQUARE == true)
				{
					Gsq[l].setColVector(6*6,4);
					Gsq[l] = tensor_product(G[l],G[l]);
				}
			}
			else
			{
				G[l].setMatrix(6,4);
				G[l] = HubbardModel::Generator(0,0);
				if (CALC_SQUARE == true)
				{
					Gsq[l].setMatrix(6*6,4);
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

template<size_t D>
KondoModel<D>::
KondoModel (size_t L_input, double J_input, initializer_list<size_t> imploc_input, initializer_list<double> Bzval_input, bool CALC_SQUARE)
:KondoModel(L_input, J_input, vector<size_t>(begin(imploc_input),end(imploc_input)), vector<double>(begin(Bzval_input),end(Bzval_input)), CALC_SQUARE)
{}

template<size_t D>
KondoModel<D>::
KondoModel (size_t L_input, vector<double> Bzval_input, double J_input, double t_input, double tPrime_input, double U_input, bool CALC_SQUARE)
	:MpoQ<2,double>(L_input, vector<qarray<2> >(begin(KondoModel<D>::q),end(KondoModel<D>::q)), {0,0}, KondoModel<D>::NMlabel, "KondoModel (PKS geometry) ", N_halveM), J(J_input), t(t_input), tPrime(tPrime_input), U(U_input)
{
	assert(L_input % 3 == 2 and "L mod 3 = 2 is required for PKS topology.");
	// assign stuff
	this->Daux = 14;

	// initialize member variable imploc
	this->imploc.resize(L_input);
	std::iota(this->imploc.begin(), this->imploc.end(), 0);
	
	// if Bzval_input empty, set it to zero
	if (Bzval_input.size() == 0)
	{
		Bzval.assign(this->N_sites,0.);
	}
	else
	{
		assert(this->N_sites == Bzval_input.size() and "Impurities and B-fields do not match!");
		Bzval = Bzval_input;
	}

	// label
	stringstream ss;
	ss << "(S=" << frac(D-1,2) << ",J=" << J << ",t=" << t << ",tPrime=" << tPrime << ",U=" << U << ")";
	this->label += ss.str();

	//construct PKS topology with t-tPrime structure
	std::vector<double> tVec(this->N_sites);
	std::vector<double> tPrimeVec(this->N_sites);

	for (size_t l=0; l<this->N_sites; ++l)
	{
		if(l == 0) {tVec[l]=this->t; tPrimeVec[l]=this->tPrime;}
		if(l == 1) {tVec[l]=this->t; tPrimeVec[l]=this->tPrime;}
		if(((l+1) % 3 == 0) and (l > 1)) {tVec[l]=this->tPrime; tPrimeVec[l]=this->tPrime;}
		if(((l+1) % 3 == 1) and (l > 1)) {tVec[l]=this->tPrime; tPrimeVec[l]=this->t;}
		if(((l+1) % 3 == 2) and (l > 1)) {tVec[l]=this->t; tPrimeVec[l]=this->tPrime;}
	}

	// create the SuperMatrices
	vector<SuperMatrix<double> > G(this->N_sites);
	vector<SuperMatrix<double> > Gsq;
	if (CALC_SQUARE == true)
	{
		Gsq.resize(this->N_sites);
	}
	
	for (size_t l=0; l<this->N_sites; ++l)
	{
		if (l==0)
		{
			G[l].setRowVector(14,8);
			G[l] = Generator(this->J,this->Bzval[l],0.,tVec[l],tPrimeVec[l],this->U).row(13);
			if (CALC_SQUARE == true)
			{
				Gsq[l].setRowVector(14*14,8);
				Gsq[l] = tensor_product(G[l],G[l]);
			}
		}
		else if (l==this->N_sites-1)
		{
			G[l].setColVector(14,8);
			G[l] = Generator(this->J,this->Bzval[l],0.,tVec[l],tPrimeVec[l],this->U).col(0);
			if (CALC_SQUARE == true)
			{
				Gsq[l].setColVector(14*14,8);
				Gsq[l] = tensor_product(G[l],G[l]);
			}
		}
		else
		{
			G[l].setMatrix(14,8);
			G[l] = Generator(this->J,this->Bzval[l],0.,tVec[l],tPrimeVec[l],this->U);
			if (CALC_SQUARE == true)
			{
				Gsq[l].setMatrix(14*14,8);
				Gsq[l] = tensor_product(G[l],G[l]);
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
		
template<size_t D>
string KondoModel<D>::
N_halveM (qarray<2> qnum)
{
	stringstream ss;
	ss << "(" << qnum[0] << ",";
	
	qarray<1> mag;
	mag[0] = qnum[1];
	string halfmag = HeisenbergModel<2>::halve(mag);
	halfmag.erase(0,1);
	ss << halfmag;
	
	return ss.str();
}

template<size_t D>
MpoQ<2> KondoModel<D>::
Simp (size_t L, size_t loc, SPINOP_LABEL SOP)
{
	assert(loc<L);
	stringstream ss;
	ss << SOP << "(" << loc << ")";
	MpoQ<2> Mout(L, locBasis(), {0,0}, KondoModel<D>::NMlabel, ss.str());
	MatrixXd Id4(4,4); Id4.setIdentity();
	Mout.setLocal(loc, kroneckerProduct(SpinBase<D>::Scomp(SOP),Id4));
	return Mout;
}

template<size_t D>
MpoQ<2> KondoModel<D>::
Ssub (size_t L, size_t loc, SPINOP_LABEL SOP)
{
	assert(loc<L);
	stringstream ss;
	ss << SOP << "(" << loc << ")";
	MpoQ<2> Mout(L, locBasis(), {0,0}, KondoModel<D>::NMlabel, ss.str());
	MatrixXd IdImp(qloc[loc].size()/4, qloc[loc].size()/4); IdImp.setIdentity();
	Mout.setLocal(loc, kroneckerProduct(IdImp, HubbardModel::Scomp(SOP)));
	return Mout;
}

template<size_t D>
MpoQ<2> KondoModel<D>::
SimpSimp (size_t L,size_t loc1, SPINOP_LABEL SOP1, size_t loc2, SPINOP_LABEL SOP2)
{
	assert(loc1<L and loc2<L);
	stringstream ss;
	ss << SOP1 << "(" << loc1 << ")" << SOP2 << "(" << loc2 << ")";
	MpoQ<2> Mout(L, locBasis(), {0,0}, KondoModel<D>::NMlabel, ss.str());
	MatrixXd Id4(4,4); Id4.setIdentity();
	Mout.setLocal({loc1, loc2}, {kroneckerProduct(SpinBase<D>::Scomp(SOP1),Id4), 
	                             kroneckerProduct(SpinBase<D>::Scomp(SOP2),Id4)}
	             );
	return Mout;
}

template<size_t D>
MpoQ<2> KondoModel<D>::
SsubSsub (size_t L,size_t loc1, SPINOP_LABEL SOP1, size_t loc2, SPINOP_LABEL SOP2)
{
	assert(loc1<L and loc2<L);
	stringstream ss;
	ss << SOP1 << "(" << loc1 << ")" << SOP2 << "(" << loc2 << ")";
	MpoQ<2> Mout(L, locBasis(), {0,0}, KondoModel<D>::NMlabel, ss.str());
	MatrixXd IdImp1(qloc[loc1].size()/4, qloc[loc1].size()/4); IdImp1.setIdentity();
	MatrixXd IdImp2(qloc[loc2].size()/4, qloc[loc2].size()/4); IdImp2.setIdentity();
	Mout.setLocal({loc1, loc2}, {kroneckerProduct(IdImp1,HubbardModel::Scomp(SOP1)), 
	                             kroneckerProduct(IdImp2,HubbardModel::Scomp(SOP2))}
	             );
	return Mout;
}

template<size_t D>
MpoQ<2> KondoModel<D>::
SimpSsub (size_t L,size_t loc1, SPINOP_LABEL SOP1, size_t loc2, SPINOP_LABEL SOP2)
{
	assert(loc1<L and loc2<L);
	stringstream ss;
	ss << SOP1 << "(" << loc1 << ")" << SOP2 << "(" << loc2 << ")";
	MpoQ<2> Mout(L, locBasis(), {0,0}, KondoModel<D>::NMlabel, ss.str());
	MatrixXd Id4(4,4); Id4.setIdentity();
	MatrixXd IdImp(qloc[loc2].size()/4, qloc[loc2].size()/4); IdImp.setIdentity();
	Mout.setLocal({loc1, loc2}, {kroneckerProduct(SpinBase<D>::Scomp(SOP1),Id4), 
	                             kroneckerProduct(IdImp,HubbardModel::Scomp(SOP2))}
	             );
	return Mout;
}

template<size_t D>
class KondoModel<D>::qarrayIterator
{
public:
	
	/**
	\param qloc_input : vector of local bases
	\param l_frst : first site
	\param l_last : last site
	*/
	qarrayIterator (const vector<vector<qarray<2> > > &qloc_input, int l_frst, int l_last)
	{
		int Nimps = 0;
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
				if (qloc_input[l].size() == 4*D)
				{
					++Nimps;
				}
			}
		}
		
		int Sx2 = static_cast<int>(D-1); // necessary because of size_t
		for (int Sz=-Sx2*Nimps; Sz<=Sx2*Nimps; Sz+=2)
		for (int Nup=0; Nup<=N_sites; ++Nup)
		for (int Ndn=0; Ndn<=N_sites; ++Ndn)
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

template<size_t D>
bool KondoModel<D>::
validate (qarray<2> qnum) const
{
	int Sx2 = static_cast<int>(D-1); // necessary because of size_t
	return (qnum[0]+Sx2*imploc.size())%2 == qnum[1]%2;
}

};

#endif

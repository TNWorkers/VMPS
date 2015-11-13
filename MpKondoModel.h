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
class KondoModel : public MpoQ<2,double>
{
public:
	/**Does nothing.*/
	KondoModel () {};
	
	/**Constructs a Kondo Lattice Model.
	\param L_input : chain length
	\param J_input : \f$J\f$
	\param tPrime_input : \f$t^{\prime}\f$ next nearest neighbour (nnn) hopping. \f$t^{\prime}>0\f$ is common sign.
	\param U_input : \f$U\f$ (local Hubbard interaction)
	\param Bz_input : \f$B_z\f$
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	\param D_input : \f$2S+1\f$ (impurity spin)*/
	KondoModel (size_t L_input, double J_input=-1., double tPrime_input=0., double U_input=0., double Bz_input=0., bool CALC_SQUARE=true, size_t D_input=2);

	/**Constructs a Kondo Impurity Model (aka a diluted Kondo Model) using initializer lists for the set of impurities.
	\param L_input : chain length
	\param J_input : \f$J\f$
	\param imploc_input : list with locations of the impurities
	\param Bzval_input : list with locations of the local magnetic fields
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	\param D_input : \f$2S+1\f$ (impurity spin)*/
	KondoModel (size_t L_input, double J_input, initializer_list<size_t> imploc_input, initializer_list<double> Bzval_input={}, bool CALC_SQUARE=true, size_t D_input=2);

	/**Constructs a Kondo Impurity Model (aka a diluted Kondo Model) using vectors for the set of impurities.
	\param L_input : chain length
	\param J_input : \f$J\f$
	\param imploc_input : list with locations of the impurities
	\param Bzval_input : list with locations of the local magnetic fields
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	\param D_input : \f$2S+1\f$ (impurity spin)*/
	KondoModel (size_t L_input, double J_input, vector<size_t> imploc_input, vector<double> Bzval_input={}, bool CALC_SQUARE=true, size_t D_input=2);

	/**Construct basic MPO stuff for derived models from the Kondo model.*/
	KondoModel (size_t L_input, string modelDescription, size_t D_input=2);

	static void set_operators (vector<tuple<double,Eigen::MatrixXd> > &Olocal, vector<tuple<double,Eigen::MatrixXd,Eigen::MatrixXd> > &Otight, vector<tuple<double,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> > &Onextn,
							   double J, double Bz, double Bx=0., double t=-1., double tPrime=0., double U=0., size_t D=2);

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
	static const vector<qarray<2> > qloc (size_t D=2)
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
	MpoQ<2> Simp (size_t L, size_t loc, SPINOP_LABEL SOP);
	MpoQ<2> Ssub (size_t L, size_t loc, SPINOP_LABEL SOP);
	MpoQ<2> SimpSimp (size_t L, size_t loc1, SPINOP_LABEL SOP1, size_t loc2, SPINOP_LABEL SOP2);
	MpoQ<2> SsubSsub (size_t L, size_t loc1, SPINOP_LABEL SOP1, size_t loc2, SPINOP_LABEL SOP2);
	MpoQ<2> SimpSsub (size_t L, size_t loc1, SPINOP_LABEL SOP1, size_t loc2, SPINOP_LABEL SOP2);
	MpoQ<2> SimpSsubSimpSimp (size_t L, size_t loc1, SPINOP_LABEL SOP1, size_t loc2, SPINOP_LABEL SOP2, 
	                                    size_t loc3, SPINOP_LABEL SOP3, size_t loc4, SPINOP_LABEL SOP4);
	MpoQ<2> SimpSsubSimpSsub (size_t L, size_t loc1, SPINOP_LABEL SOP1, size_t loc2, SPINOP_LABEL SOP2, 
	                                    size_t loc3, SPINOP_LABEL SOP3, size_t loc4, SPINOP_LABEL SOP4);

	///@}
	
private:
	
	double J=-1., Bz=0., t=-1., tPrime=0., U=0.;
	size_t D=2;
	
	vector<double> Bzval;
	vector<size_t> imploc;
};

const std::array<string,2> KondoModel::NMlabel{"N","M"};

void KondoModel::
set_operators (vector<tuple<double,Eigen::MatrixXd> > &Olocal, vector<tuple<double,Eigen::MatrixXd,Eigen::MatrixXd> > &Otight, vector<tuple<double,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> > &Onextn, double J, double Bz, double Bx, double t, double tPrime, double U, size_t D)

{
	Eigen::MatrixXd Id4(4,4); Id4.setIdentity();
	Eigen::MatrixXd IdSpins(D,D); IdSpins.setIdentity();

	//clear old values of operators
	Olocal.resize(0);
	Otight.resize(0);
	Onextn.resize(0);

	//set local interaction Olocal
	Olocal.push_back(std::make_tuple(-0.5*J, kroneckerProduct(SpinBase::Scomp(SP,D), FermionBase::Sp.transpose())));
	Olocal.push_back(std::make_tuple(-0.5*J, kroneckerProduct(SpinBase::Scomp(SM,D), FermionBase::Sp)));
	Olocal.push_back(std::make_tuple(-J, kroneckerProduct(SpinBase::Scomp(SZ,D), FermionBase::Sz)));
	if (Bz != 0.)
	{
		Olocal.push_back(std::make_tuple(-Bz, kroneckerProduct(SpinBase::Scomp(SZ,D), Id4)));
	}
	if (Bx != 0.)
	{
		Olocal.push_back(std::make_tuple(-Bx, kroneckerProduct(SpinBase::Scomp(SX,D), Id4)));
	}
	if (U != 0.)
	{
		Olocal.push_back(std::make_tuple(U, kroneckerProduct(IdSpins, FermionBase::d)));
	}

	//set nearest neighbour term Otight
	Otight.push_back(std::make_tuple(-t,kroneckerProduct(IdSpins, FermionBase::cUP.transpose()), kroneckerProduct(IdSpins, FermionBase::fsign * FermionBase::cUP)));
	Otight.push_back(std::make_tuple(-t,kroneckerProduct(IdSpins, FermionBase::cDN.transpose()), kroneckerProduct(IdSpins, FermionBase::fsign * FermionBase::cDN)));
	Otight.push_back(std::make_tuple(t,kroneckerProduct(IdSpins, FermionBase::cUP), kroneckerProduct(IdSpins, FermionBase::fsign * FermionBase::cUP.transpose())));
	Otight.push_back(std::make_tuple(t,kroneckerProduct(IdSpins, FermionBase::cDN), kroneckerProduct(IdSpins, FermionBase::fsign * FermionBase::cDN.transpose())));

	if (tPrime != 0.)
	{
		//set next nearest neighbour term Onextn
		Onextn.push_back(std::make_tuple(-tPrime,
											   kroneckerProduct(IdSpins, FermionBase::cUP.transpose()),
											   kroneckerProduct(IdSpins, FermionBase::fsign * FermionBase::cUP),
											   kroneckerProduct(IdSpins, FermionBase::fsign)));
		Onextn.push_back(std::make_tuple(-tPrime,
											   kroneckerProduct(IdSpins, FermionBase::cDN.transpose()),
											   kroneckerProduct(IdSpins, FermionBase::fsign * FermionBase::cDN),
											   kroneckerProduct(IdSpins, FermionBase::fsign)));
		Onextn.push_back(std::make_tuple(tPrime,
											   kroneckerProduct(IdSpins, FermionBase::cUP),
											   kroneckerProduct(IdSpins, FermionBase::fsign * FermionBase::cUP.transpose()),
											   kroneckerProduct(IdSpins, FermionBase::fsign)));
		Onextn.push_back(std::make_tuple(tPrime,
											   kroneckerProduct(IdSpins, FermionBase::cDN),
											   kroneckerProduct(IdSpins, FermionBase::fsign * FermionBase::cDN.transpose()),
											   kroneckerProduct(IdSpins, FermionBase::fsign)));		
	}
}

KondoModel::
KondoModel (size_t L_input, string modelDescription, size_t D_input)
	:MpoQ<2> (L_input, KondoModel::qloc(D_input), {0,0}, KondoModel::NMlabel, modelDescription, N_halveM)
{}
	
KondoModel::
KondoModel (size_t L_input, double J_input, double tPrime_input, double U_input, double Bz_input, bool CALC_SQUARE, size_t D_input)
:MpoQ<2> (L_input, KondoModel::qloc(D_input), {0,0}, KondoModel::NMlabel, "KondoModel", N_halveM),
J(J_input), Bz(Bz_input), tPrime(tPrime_input), U(U_input), D(D_input)
{
	// initialize member variable imploc
	this->imploc.resize(L_input);
	std::iota(this->imploc.begin(), this->imploc.end(), 0);

	stringstream ss;
	ss << "(J=" << J << ",Bz=" << Bz << ",t'=" << tPrime << ",U=" << U << ")";
	this->label += ss.str();

	set_operators(this->Olocal, this->Otight, this->Onextn, J, Bz, 0., -1., tPrime, U, D);
	this->Daux = 2 + this->Otight.size() + 2*this->Onextn.size();

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
KondoModel (size_t L_input, double J_input, vector<size_t> imploc_input, vector<double> Bzval_input, bool CALC_SQUARE, size_t D_input)
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
	this->N_sites = L_input;
	this->Qtot = {0,0};
	this->qlabel = NMlabel;
	this->label = "KondoModel";
	this->format = N_halveM;

	MpoQ<2,double>::qloc.resize(this->N_sites);
	
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
			MpoQ<2,double>::qloc[l] = qloc(D);
			
			size_t i = it-imploc.begin();
			if (l==0)
			{
				G[l].setRowVector(6,8);
				set_operators(Olocal, Otight, Onextn, J,Bzval[i],0.,-1.,0.,0.,D);
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
				set_operators(Olocal, Otight, Onextn, J,Bzval[i],0.,-1.,0.,0.,D);
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
				set_operators(Olocal, Otight, Onextn, J,Bzval[i],0.,-1.,0.,0.,D);
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
				HubbardModel::set_operators(Olocal, Otight, Onextn, 0.);
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
				HubbardModel::set_operators(Olocal, Otight, Onextn, 0.);
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
				HubbardModel::set_operators(Olocal, Otight, Onextn, 0.);
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
KondoModel (size_t L_input, double J_input, initializer_list<size_t> imploc_input, initializer_list<double> Bzval_input, bool CALC_SQUARE, size_t D_input)
:KondoModel(L_input, J_input, vector<size_t>(begin(imploc_input),end(imploc_input)), vector<double>(begin(Bzval_input),end(Bzval_input)), CALC_SQUARE, D_input)
{}

string KondoModel::
N_halveM (qarray<2> qnum)
{
	stringstream ss;
	ss << "(" << qnum[0] << ",";
	
	qarray<1> mag;
	mag[0] = qnum[1];
	string halfmag = HeisenbergModel::halve(mag);
	halfmag.erase(0,1);
	ss << halfmag;
	
	return ss.str();
}

MpoQ<2> KondoModel::
Simp (size_t L, size_t loc, SPINOP_LABEL SOP)
{
	assert(loc<L);
	stringstream ss;
	ss << SOP << "(" << loc << ")";
	MpoQ<2> Mout(L, locBasis(), {0,0}, KondoModel::NMlabel, ss.str());
	MatrixXd Id4(4,4); Id4.setIdentity();
	Mout.setLocal(loc, kroneckerProduct(SpinBase::Scomp(SOP,D),Id4));
	return Mout;
}

MpoQ<2> KondoModel::
Ssub (size_t L, size_t loc, SPINOP_LABEL SOP)
{
	assert(loc<L);
	stringstream ss;
	ss << SOP << "(" << loc << ")";
	MpoQ<2> Mout(L, locBasis(), {0,0}, KondoModel::NMlabel, ss.str());
	MatrixXd IdImp(MpoQ<2,double>::qloc[loc].size()/4, MpoQ<2,double>::qloc[loc].size()/4); IdImp.setIdentity();
	Mout.setLocal(loc, kroneckerProduct(IdImp, FermionBase::Scomp(SOP)));
	return Mout;
}

MpoQ<2> KondoModel::
SimpSimp (size_t L, size_t loc1, SPINOP_LABEL SOP1, size_t loc2, SPINOP_LABEL SOP2)
{
	assert(loc1<L and loc2<L);
	stringstream ss;
	ss << SOP1 << "(" << loc1 << ")" << SOP2 << "(" << loc2 << ")";
	MpoQ<2> Mout(L, locBasis(), {0,0}, KondoModel::NMlabel, ss.str());
	MatrixXd Id4(4,4); Id4.setIdentity();
	Mout.setLocal({loc1, loc2}, {kroneckerProduct(SpinBase::Scomp(SOP1,D),Id4), 
	                             kroneckerProduct(SpinBase::Scomp(SOP2,D),Id4)}
	             );
	return Mout;
}


MpoQ<2> KondoModel::
SsubSsub (size_t L, size_t loc1, SPINOP_LABEL SOP1, size_t loc2, SPINOP_LABEL SOP2)
{
	assert(loc1<L and loc2<L);
	stringstream ss;
	ss << SOP1 << "(" << loc1 << ")" << SOP2 << "(" << loc2 << ")";
	MpoQ<2> Mout(L, locBasis(), {0,0}, KondoModel::NMlabel, ss.str());
	MatrixXd IdImp1(MpoQ<2>::qloc[loc1].size()/4, MpoQ<2>::qloc[loc1].size()/4); IdImp1.setIdentity();
	MatrixXd IdImp2(MpoQ<2>::qloc[loc2].size()/4, MpoQ<2>::qloc[loc2].size()/4); IdImp2.setIdentity();
	Mout.setLocal({loc1, loc2}, {kroneckerProduct(IdImp1,FermionBase::Scomp(SOP1)), 
	                             kroneckerProduct(IdImp2,FermionBase::Scomp(SOP2))}
	             );
	return Mout;
}

MpoQ<2> KondoModel::
SimpSsub (size_t L, size_t loc1, SPINOP_LABEL SOP1, size_t loc2, SPINOP_LABEL SOP2)
{
	assert(loc1<L and loc2<L);
	stringstream ss;
	ss << SOP1 << "(" << loc1 << ")" << SOP2 << "(" << loc2 << ")";
	MpoQ<2> Mout(L, locBasis(), {0,0}, KondoModel::NMlabel, ss.str());
	MatrixXd Id4(4,4); Id4.setIdentity();
	MatrixXd IdImp(MpoQ<2>::qloc[loc2].size()/4, MpoQ<2>::qloc[loc2].size()/4); IdImp.setIdentity();
	Mout.setLocal({loc1, loc2}, {kroneckerProduct(SpinBase::Scomp(SOP1,D),Id4), 
	                             kroneckerProduct(IdImp,FermionBase::Scomp(SOP2))}
	             );
	return Mout;
}

MpoQ<2> KondoModel::
SimpSsubSimpSimp (size_t L, size_t loc1, SPINOP_LABEL SOP1, size_t loc2, SPINOP_LABEL SOP2, size_t loc3, SPINOP_LABEL SOP3, size_t loc4, SPINOP_LABEL SOP4)
{
	assert(loc1<L and loc2<L and loc3<L and loc4<L);
	stringstream ss;
	ss << SOP1 << "(" << loc1 << ")" << SOP2 << "(" << loc2 << ")" << SOP3 << "(" << loc3 << ")" << SOP4 << "(" << loc4 << ")";
	MpoQ<2> Mout(L, locBasis(), {0,0}, KondoModel::NMlabel, ss.str());
	MatrixXd Id4(4,4); Id4.setIdentity();
	MatrixXd IdImp(MpoQ<2>::qloc[loc2].size()/4, MpoQ<2>::qloc[loc2].size()/4); IdImp.setIdentity();
	Mout.setLocal({loc1, loc2, loc3, loc4}, {kroneckerProduct(SpinBase::Scomp(SOP1,D),Id4), 
	                                         kroneckerProduct(IdImp,FermionBase::Scomp(SOP2)),
	                                         kroneckerProduct(SpinBase::Scomp(SOP3,D),Id4),
	                                         kroneckerProduct(SpinBase::Scomp(SOP4,D),Id4)}
	             );
	return Mout;
}

MpoQ<2> KondoModel::
SimpSsubSimpSsub (size_t L, size_t loc1, SPINOP_LABEL SOP1, size_t loc2, SPINOP_LABEL SOP2, size_t loc3, SPINOP_LABEL SOP3, size_t loc4, SPINOP_LABEL SOP4)
{
	assert(loc1<L and loc2<L and loc3<L and loc4<L);
	stringstream ss;
	ss << SOP1 << "(" << loc1 << ")" << SOP2 << "(" << loc2 << ")" << SOP3 << "(" << loc3 << ")" << SOP4 << "(" << loc4 << ")";
	MpoQ<2> Mout(L, locBasis(), {0,0}, KondoModel::NMlabel, ss.str());
	MatrixXd Id4(4,4); Id4.setIdentity();
	MatrixXd IdImp(MpoQ<2>::qloc[loc2].size()/4, MpoQ<2>::qloc[loc2].size()/4); IdImp.setIdentity();
	Mout.setLocal({loc1, loc2, loc3, loc4}, {kroneckerProduct(SpinBase::Scomp(SOP1,D),Id4), 
				kroneckerProduct(IdImp,FermionBase::Scomp(SOP2)),
				kroneckerProduct(SpinBase::Scomp(SOP3,D),Id4),
				kroneckerProduct(IdImp,FermionBase::Scomp(SOP4))}
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
	qarrayIterator (const vector<vector<qarray<2> > > &qloc_input, int l_frst, int l_last)
	{
		int Nimps = 0;
		size_t D = 2;
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
				if (qloc_input[l].size()/4 > 1)
				{
					++Nimps;
					D = qloc_input[l].size()/4;
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


bool KondoModel::
validate (qarray<2> qnum) const
{
	int Sx2 = static_cast<int>(D-1); // necessary because of size_t
	return (qnum[0]+Sx2*imploc.size())%2 == qnum[1]%2;
}

};

#endif

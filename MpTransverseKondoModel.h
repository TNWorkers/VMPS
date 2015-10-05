#ifndef STRAWBERRY_TRANSVERSEKONDOMODEL
#define STRAWBERRY_TRANSVERSEKONDOMODEL

#include "MpKondoModel.h"

namespace VMPS
{

/**MPO representation of 
\f$
H = - \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} - J \sum_{i \in I} \mathbf{S}_i \cdot \mathbf{s}_i - \sum_{i \in I} B_i^z S_i^z - \sum_{i \in I} B_i^x S_i^x
\f$.
The set of impurities \f$I\f$ is completely free to choose.
\note \f$J<0\f$ : antiferromagnetic
\note The local magnetic fields act on the impurities only.*/
class TransverseKondoModel : public MpoQ<1,double>
{
public:
	
	/**Constructs a Kondo Impurity Model (aka a diluted Kondo Model) using initializer lists for the set of impurities.
	\param L_input : chain length
	\param J_input : \f$J\f$
	\param imploc_input : list with locations of the impurities
	\param Bzloc_input : list with locations of the local magnetic fields in z-direction
	\param Bxloc_input : list with locations of the local magnetic fields in x-direction
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	\param D_input : \f$2S+1\f$ (impurity spin)*/
	TransverseKondoModel (size_t L_input, double J_input, 
	                      initializer_list<size_t> imploc_input, initializer_list<double> Bzloc_input={}, initializer_list<double> Bxloc_input={}, 
	                      bool CALC_SQUARE=true, size_t D_input=2);
	
	/**Constructs a Kondo Impurity Model (aka a diluted Kondo Model) using vectors for the set of impurities.
	\param L_input : chain length
	\param J_input : \f$J\f$
	\param imploc_input : list with locations of the impurities
	\param Bzloc_input : list with locations of the local magnetic fields in z-direction
	\param Bxloc_input : list with locations of the local magnetic fields in x-direction
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	\param D_input : \f$2S+1\f$ (impurity spin)*/
	TransverseKondoModel (size_t L_input, double J_input, 
	                      vector<size_t> imploc_input, vector<double> Bzloc_input, vector<double> Bxloc_input, 
	                      bool CALC_SQUARE=true, size_t D_input=2);
	
	/**Labels the conserved quantum numbers as "N", "M".*/
	static const std::array<string,1> Nlabel;
	
	static const std::array<qarray<1>,4> qsub;
	static const vector<qarray<1> > qimp (size_t D)
	{
		vector<qarray<1> > vout;
		for (int i=0; i<D; ++i)
		{
			vout.push_back(qarray<1>{0});
			vout.push_back(qarray<1>{1});
			vout.push_back(qarray<1>{1});
			vout.push_back(qarray<1>{2});
		}
		return vout;
	};
	
	///@{
	/**Typedef for convenient reference (no need to specify \p Nq, \p Scalar all the time).*/
	typedef MpsQ<1,double>                           StateXd;
	typedef MpsQ<1,complex<double> >                 StateXcd;
	typedef DmrgSolverQ<1,TransverseKondoModel>      Solver;
	typedef MpsQCompressor<1,double,double>          CompressorXd;
	typedef MpsQCompressor<1,complex<double>,double> CompressorXcd;
	typedef MpoQ<1>                                  Operator;
	///@}
	
	///@{
	/**Operator for the impurity spin.*/
	MpoQ<1> Simp (size_t L, size_t loc, SPINOP_LABEL Sa);
	
	/**Operator for the substrate spin.*/
	MpoQ<1> Ssub (size_t L, size_t loc, SPINOP_LABEL Sa);
	
	/**Operator for the impurity-substrate correlations.*/
	MpoQ<1> SimpSsub (size_t L, size_t loc1, SPINOP_LABEL SOP1, size_t loc2, SPINOP_LABEL SOP2);
	
	/**Operator for the impurity-impurity correlations.*/
	MpoQ<1> SimpSimp (size_t L, size_t loc1, SPINOP_LABEL SOP1, size_t loc2, SPINOP_LABEL SOP2);
	
	/***/
	MpoQ<1> hopping (size_t L, size_t loc);
	///@}
	
private:
	
	double J=-1.;
	size_t D=2;
	
	vector<double> Bzloc, Bxloc;
	vector<size_t> imploc;
};

const std::array<string,1> TransverseKondoModel::Nlabel{"N"};


const std::array<qarray<1>,4> TransverseKondoModel::qsub
{
	qarray<1>{0}, qarray<1>{1}, qarray<1>{1}, qarray<1>{2}
};

TransverseKondoModel::
TransverseKondoModel (size_t L_input, double J_input, vector<size_t> imploc_input, vector<double> Bzloc_input, vector<double> Bxloc_input, bool CALC_SQUARE, size_t D_input)
:MpoQ<1,double>(), J(J_input), imploc(imploc_input), D(D_input)
{
	// if Bzloc_input empty, set it to zero
	if (Bzloc_input.size() == 0)
	{
		Bzloc.assign(imploc.size(),0.);
	}
	else
	{
		assert(imploc_input.size() == Bzloc_input.size());
		Bzloc = Bzloc_input;
	}
	
	if (Bxloc_input.size() == 0)
	{
		Bxloc.assign(imploc.size(),0.);
	}
	else
	{
		assert(imploc_input.size() == Bxloc_input.size());
		Bxloc = Bxloc_input;
	}
	
	// assign stuff
	this->N_sites = L_input;
	this->Qtot = {0};
	this->qlabel = Nlabel;
	this->label = "TransverseKondoModel";
	this->format = noFormat;
	this->Daux = 6;
	this->qloc.resize(this->N_sites);
	
	// make a pretty label
	stringstream ss;
	ss << "(J=" << J << ",imps={";
	for (auto i=0; i<imploc.size(); ++i)
	{
		ss << imploc[i];
		if (i!=imploc.size()-1) {ss << ",";}
	}
	ss << "}";
	ss << ",Bz={";
	for (auto i=0; i<Bzloc.size(); ++i)
	{
		ss << Bzloc[i];
		if (i!=Bzloc.size()-1) {ss << ",";}
	}
	ss << "}";
	ss << ",Bx={";
	for (auto i=0; i<Bxloc.size(); ++i)
	{
		ss << Bxloc[i];
		if (i!=Bxloc.size()-1) {ss << ",";}
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
			this->qloc[l] = qimp(D);
			
			size_t i = it-imploc.begin();
			if (l==0)
			{
				G[l].setRowVector(6,8);
				G[l] = KondoModel::Generator(J,Bzloc[i],Bxloc[i],-1.,0.,0.,D).row(5);
				if (CALC_SQUARE == true)
				{
					Gsq[l].setRowVector(6*6,8);
					Gsq[l] = tensor_product(G[l],G[l]);
				}
			}
			else if (l==this->N_sites-1)
			{
				G[l].setColVector(6,8);
				G[l] = KondoModel::Generator(J,Bzloc[i],Bxloc[i],-1.,0.,0.,D).col(0);
				if (CALC_SQUARE == true)
				{
					Gsq[l].setColVector(6*6,8);
					Gsq[l] = tensor_product(G[l],G[l]);
				}
			}
			else
			{
				G[l].setMatrix(6,8);
				G[l] = KondoModel::Generator(J,Bzloc[i],Bxloc[i],-1.,0.,0.,D);
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
			this->qloc[l] = vector<qarray<1> >(begin(qsub),end(qsub));
			
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

TransverseKondoModel::
TransverseKondoModel (size_t L_input, double J_input, initializer_list<size_t> imploc_input, initializer_list<double> Bzloc_input, initializer_list<double> Bxloc_input, bool CALC_SQUARE, size_t D_input)
:TransverseKondoModel(L_input, J_input, vector<size_t>(begin(imploc_input),end(imploc_input)), vector<double>(begin(Bzloc_input),end(Bzloc_input)), vector<double>(begin(Bxloc_input),end(Bxloc_input)), CALC_SQUARE, D_input)
{}

MpoQ<1> TransverseKondoModel::
Simp (size_t L, size_t loc, SPINOP_LABEL Sa)
{
	assert(loc<L);
	stringstream ss;
	ss << Sa << "imp(" << loc << ")";
	MpoQ<1> Mout(L, locBasis(), {0}, Nlabel, ss.str());
	MatrixXd Id4(4,4); Id4.setIdentity();
	Mout.setLocal(loc, kroneckerProduct(SpinBase::Scomp(Sa,D),Id4));
	return Mout;
}

MpoQ<1> TransverseKondoModel::
Ssub (size_t L, size_t loc, SPINOP_LABEL Sa)
{
	assert(loc<L);
	stringstream ss;
	ss << Sa << "sub(" << loc << ")";
	MpoQ<1> Mout(L, locBasis(), {0}, Nlabel, ss.str());
	MatrixXd IdImp(qloc[loc].size()/4, qloc[loc].size()/4); IdImp.setIdentity();
	Mout.setLocal(loc, kroneckerProduct(IdImp, FermionBase::Scomp(Sa)));
	return Mout;
}

MpoQ<1> TransverseKondoModel::
SimpSsub (size_t L,size_t loc1, SPINOP_LABEL SOP1, size_t loc2, SPINOP_LABEL SOP2)
{
	assert(loc1<L and loc2<L);
	stringstream ss;
	ss << SOP1 << "(" << loc1 << ")" << SOP2 << "(" << loc2 << ")";
	MpoQ<1> Mout(L, locBasis(), {0}, Nlabel, ss.str());
	MatrixXd Id4(4,4); Id4.setIdentity();
	MatrixXd IdImp(MpoQ<1>::qloc[loc2].size()/4, MpoQ<1>::qloc[loc2].size()/4); IdImp.setIdentity();
	Mout.setLocal({loc1, loc2}, {kroneckerProduct(SpinBase::Scomp(SOP1,D),Id4), 
	                             kroneckerProduct(IdImp,FermionBase::Scomp(SOP2))}
	             );
	return Mout;
}

MpoQ<1> TransverseKondoModel::
SimpSimp (size_t L,size_t loc1, SPINOP_LABEL SOP1, size_t loc2, SPINOP_LABEL SOP2)
{
	assert(loc1<L and loc2<L);
	stringstream ss;
	ss << SOP1 << "(" << loc1 << ")" << SOP2 << "(" << loc2 << ")";
	MpoQ<1> Mout(L, locBasis(), {0}, Nlabel, ss.str());
	MatrixXd Id4(4,4); Id4.setIdentity();
	MatrixXd IdImp(MpoQ<1>::qloc[loc2].size()/4, MpoQ<1>::qloc[loc2].size()/4); IdImp.setIdentity();
	Mout.setLocal({loc1, loc2}, {kroneckerProduct(SpinBase::Scomp(SOP1,D),Id4), 
	                             kroneckerProduct(SpinBase::Scomp(SOP2,D),Id4)}
	             );
	return Mout;
}

MpoQ<1> TransverseKondoModel::
hopping (size_t L, size_t loc)
{
	assert(loc<L);
	stringstream ss;
	ss << "hopping" << "(" << loc << ")";
	vector<SuperMatrix<double> > G(L);
	for (size_t l=0; l<L; ++l)
	{
		auto Gloc = HubbardModel::Generator(0,0);
		auto it = find(imploc.begin(),imploc.end(),l);
		if (it != imploc.end())
		{
			Gloc = KondoModel::Generator(0,0,0,-1.,0,0,D);
		}
		
		if (l==0)
		{
			G[l].setRowVector(6,Gloc.D());
			if (l == loc or l==loc-1)
			{
				G[l] = Gloc.row(5);
			}
			else
			{
				G[l].setZero();
				G[l](0,5).setIdentity();
			}
		}
		else if (l == L-1)
		{
			G[l].setColVector(6,Gloc.D());
			if (l == loc or l==loc+1)
			{
				G[l] = Gloc.col(0);
			}
			else
			{
				G[l].setZero();
				G[l](0,0).setIdentity();
			}
		}
		else
		{
			G[l].setMatrix(6,Gloc.D());
			if (l==loc or l==loc-1 or l==loc+1)
			{
				G[l] = Gloc;
				if (l == loc-1)
				{
					G[l](1,0).setZero();
					G[l](2,0).setZero();
					G[l](3,0).setZero();
					G[l](4,0).setZero();
				}
				else if (l == loc+1)
				{
					G[l](5,1).setZero();
					G[l](5,2).setZero();
					G[l](5,3).setZero();
					G[l](5,4).setZero();
				}
			}
			else
			{
				G[l].setZero();
				G[l](0,0).setIdentity();
				G[l](5,5).setIdentity();
			}
		}
	}
	MpoQ<1> Mout(L, G, locBasis(), {0}, Nlabel, ss.str());
	return Mout;
}

};

#endif

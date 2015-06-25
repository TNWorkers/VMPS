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
template<size_t D=2>
class TransverseKondoModel : public MpoQ<1,double>
{
public:
	
	/**Constructs a Kondo Impurity Model (aka a diluted Kondo Model) using initializer lists for the set of impurities.
	\param L_input : chain length
	\param J_input : \f$J\f$
	\param imploc_input : list with locations of the impurities
	\param Bzloc_input : list with locations of the local magnetic fields in z-direction
	\param Bxloc_input : list with locations of the local magnetic fields in x-direction
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$*/
	TransverseKondoModel (size_t L_input, double J_input, 
	                      initializer_list<size_t> imploc_input, initializer_list<double> Bzloc_input={}, initializer_list<double> Bxloc_input={}, 
	                      bool CALC_SQUARE=true);
	
	/**Constructs a Kondo Impurity Model (aka a diluted Kondo Model) using vectors for the set of impurities.
	\param L_input : chain length
	\param J_input : \f$J\f$
	\param imploc_input : list with locations of the impurities
	\param Bzloc_input : list with locations of the local magnetic fields in z-direction
	\param Bxloc_input : list with locations of the local magnetic fields in x-direction
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$*/
	TransverseKondoModel (size_t L_input, double J_input, 
	                      vector<size_t> imploc_input, vector<double> Bzloc_input, vector<double> Bxloc_input, 
	                      bool CALC_SQUARE=true);
	
	/**Labels the conserved quantum numbers as "N", "M".*/
	static const std::array<string,1> Nlabel;
	
	static const std::array<qarray<1>,4>   qsub;
	static const std::array<qarray<1>,4*D> qimp;
	
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
	///@}
	
private:
	
	double J, Bz;
	
	vector<double> Bzloc, Bxloc;
	vector<size_t> imploc;
};

template<size_t D> const std::array<string,1> TransverseKondoModel<D>::Nlabel{"N"};

template<size_t D>
const std::array<qarray<1>,4> TransverseKondoModel<D>::qsub
{
	qarray<1>{0}, qarray<1>{1}, qarray<1>{1}, qarray<1>{2}
};

template<>
const std::array<qarray<1>,8> TransverseKondoModel<2>::qimp
{
	qarray<1>{0}, qarray<1>{1}, qarray<1>{1}, qarray<1>{2},
	qarray<1>{0}, qarray<1>{1}, qarray<1>{1}, qarray<1>{2}
};

template<>
const std::array<qarray<1>,12> TransverseKondoModel<3>::qimp
{
	qarray<1>{0}, qarray<1>{1}, qarray<1>{1}, qarray<1>{2},
	qarray<1>{0}, qarray<1>{1}, qarray<1>{1}, qarray<1>{2},
	qarray<1>{0}, qarray<1>{1}, qarray<1>{1}, qarray<1>{2}
};

template<size_t D>
TransverseKondoModel<D>::
TransverseKondoModel (size_t L_input, double J_input, vector<size_t> imploc_input, vector<double> Bzloc_input, vector<double> Bxloc_input, bool CALC_SQUARE)
:MpoQ<1,double>(), J(J_input), imploc(imploc_input)
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
	this->N_sv = this->Daux;
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
			this->qloc[l] = vector<qarray<1> >(begin(qimp),end(qimp));
			
			size_t i = it-imploc.begin();
			if (l==0)
			{
				G[l].setRowVector(6,8);
				G[l] = KondoModel<D>::Generator(J,Bzloc[i],Bxloc[i]).row(5);
				if (CALC_SQUARE == true)
				{
					Gsq[l].setRowVector(6*6,8);
					Gsq[l] = tensor_product(G[l],G[l]);
				}
			}
			else if (l==this->N_sites-1)
			{
				G[l].setColVector(6,8);
				G[l] = KondoModel<D>::Generator(J,Bzloc[i],Bxloc[i]).col(0);
				if (CALC_SQUARE == true)
				{
					Gsq[l].setColVector(6*6,8);
					Gsq[l] = tensor_product(G[l],G[l]);
				}
			}
			else
			{
				G[l].setMatrix(6,8);
				G[l] = KondoModel<D>::Generator(J,Bzloc[i],Bxloc[i]);
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

template<size_t D>
TransverseKondoModel<D>::
TransverseKondoModel (size_t L_input, double J_input, initializer_list<size_t> imploc_input, initializer_list<double> Bzloc_input, initializer_list<double> Bxloc_input, bool CALC_SQUARE)
:TransverseKondoModel(L_input, J_input, vector<size_t>(begin(imploc_input),end(imploc_input)), vector<double>(begin(Bzloc_input),end(Bzloc_input)), vector<double>(begin(Bxloc_input),end(Bxloc_input)), CALC_SQUARE)
{}

template<size_t D>
MpoQ<1> TransverseKondoModel<D>::
Simp (size_t L, size_t loc, SPINOP_LABEL Sa)
{
	assert(loc<L);
	stringstream ss;
	ss << Sa << "Imp(" << loc << ")";
	MpoQ<1> Mout(L, locBasis(), {0}, Nlabel, ss.str());
	MatrixXd Id4(4,4); Id4.setIdentity();
	Mout.setLocal(loc, kroneckerProduct(SpinBase<D>::Scomp(Sa),Id4));
	return Mout;
}

template<size_t D>
MpoQ<1> TransverseKondoModel<D>::
Ssub (size_t L, size_t loc, SPINOP_LABEL Sa)
{
	assert(loc<L);
	stringstream ss;
	ss << Sa << "Sub(" << loc << ")";
	MpoQ<1> Mout(L, locBasis(), {0}, Nlabel, ss.str());
	
	MatrixXd IdImp(qloc[loc].size()/4, qloc[loc].size()/4); IdImp.setIdentity();
	
	if (Sa == SX)
	{
		Mout.setLocal(loc, kroneckerProduct(IdImp, HubbardModel::Sx));
	}
	else if (Sa == iSY)
	{
		Mout.setLocal(loc, kroneckerProduct(IdImp,HubbardModel::iSy));
	}
	else if (Sa == SZ)
	{
		Mout.setLocal(loc, kroneckerProduct(IdImp, HubbardModel::Sz));
	}
	return Mout;
}

};

#endif

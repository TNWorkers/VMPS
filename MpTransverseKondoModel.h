#ifndef STRAWBERRY_TRANSVERSEKONDOMODEL
#define STRAWBERRY_TRANSVERSEKONDOMODEL

#include "MpKondoModel.h"
#include "MpHeisenbergModel.h"

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
	                      double K_input=0., size_t D_input=2, bool CALC_SQUARE=true);
	
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
	                      double K_input=0., size_t D_input=2, bool CALC_SQUARE=true);
	
	/**Labels the conserved quantum number as "N".*/
	static const std::array<string,1> Nlabel;
	
	static const std::array<qarray<1>,4> qssN;
	
	vector<qarray<1> > qsub (size_t N_legs);
	vector<qarray<1> > qimp (size_t N_legs, size_t D);
	
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
	MpoQ<1> Simp (SPINOP_LABEL Sa, size_t locx, size_t locy=0);
	
	/**Operator for the substrate spin.*/
	MpoQ<1> Ssub (SPINOP_LABEL Sa, size_t locx, size_t locy=0);
	
	/**Operator for impurity-substrate correlations.*/
	MpoQ<1> SimpSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	
	/**Operator for impurity-impurity correlations.*/
	MpoQ<1> SimpSimp (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	
	/**Operator for substrate-substrate correlations.*/
	MpoQ<1> SsubSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	
	/***/
	MpoQ<1> hopping (size_t locx, size_t locy=0);
	///@}
	
private:
	
	double J=-1.;
	size_t D=2;
	double K=0;
	
	vector<double> Bzloc, Bxloc;
	vector<size_t> imploc;
	
	FermionBase F;
	SpinBase S;
};

const std::array<string,1> TransverseKondoModel::Nlabel{"N"};
const std::array<qarray<1>,4> TransverseKondoModel::qssN {qarray<1>{0}, qarray<1>{1}, qarray<1>{1}, qarray<1>{2}};

vector<qarray<1> > TransverseKondoModel::
qsub (size_t N_legs)
{
	vector<qarray<1> > vout(pow(4,N_legs));
	
	NestedLoopIterator Nelly(N_legs,4);
	for (Nelly=Nelly.begin(); Nelly!=Nelly.end(); ++Nelly)
	{
		vout[*Nelly][0] = qssN[Nelly(0)][0];
		for (int leg=1; leg<N_legs; ++leg)
		{
			vout[*Nelly][0] += qssN[Nelly(leg)][0];
		}
	}
	
	return vout;
}

vector<qarray<1> > TransverseKondoModel::
qimp (size_t N_legs, size_t D)
{
	size_t dimS = static_cast<size_t>(pow(D,N_legs));
	size_t dimF = static_cast<size_t>(pow(4,N_legs));
	
	vector<qarray<1> > vout(dimS*dimF);
	
	vector<qarray<1> > vS = HeisenbergModel::qloc(N_legs,D);
	vector<qarray<1> > vF = qsub(N_legs);
	
	NestedLoopIterator Nelly(2,{dimS,dimF});
	for (Nelly=Nelly.begin(); Nelly!=Nelly.end(); ++Nelly)
	{
		vout[*Nelly] = vF[Nelly(1)]; // only count N
	}
	
	return vout;
};

TransverseKondoModel::
TransverseKondoModel (size_t L_input, double J_input, vector<size_t> imploc_input, vector<double> Bzloc_input, vector<double> Bxloc_input, double K_input, size_t D_input, bool CALC_SQUARE)
:MpoQ<1,double>(), J(J_input), imploc(imploc_input), D(D_input), K(K_input)
{
	F = FermionBase(1);
	S = SpinBase(1,D);
	
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
	
	// if Bxloc_input empty, set it to zero
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
	this->N_legs = 1;
	this->Qtot = {0};
	this->qlabel = Nlabel;
	this->label = "TransverseKondoModel";
	this->format = noFormat;
	this->qloc.resize(this->N_sites);
	
	// make a pretty label
	stringstream ss;
	ss << "(J=" << J << ",K=" << K << ",imps={";
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
			this->qloc[l] = qimp(N_legs,D);
			
			size_t i = it-imploc.begin();
			if (l==0)
			{
				G[l].setRowVector(6,8);
				G[l] = ::Generator(KondoModel::set_operators(F,S, J,Bzloc[i],MatrixXd::Identity(1,1),0.,Bxloc[i],0.,0.,0.,K)).row(5);
				if (CALC_SQUARE == true)
				{
					Gsq[l].setRowVector(6*6,8);
					Gsq[l] = tensor_product(G[l],G[l]);
				}
			}
			else if (l==this->N_sites-1)
			{
				G[l].setColVector(6,8);
				G[l] = ::Generator(KondoModel::set_operators(F,S, J,Bzloc[i],MatrixXd::Identity(1,1),0.,Bxloc[i],0.,0.,0.,K)).col(0);
				if (CALC_SQUARE == true)
				{
					Gsq[l].setColVector(6*6,8);
					Gsq[l] = tensor_product(G[l],G[l]);
				}
			}
			else
			{
				G[l].setMatrix(6,8);
				G[l] = ::Generator(KondoModel::set_operators(F,S, J,Bzloc[i],MatrixXd::Identity(1,1),0.,Bxloc[i],0.,0.,0.,K));
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
			this->qloc[l] = qsub(N_legs);
			
			if (l==0)
			{
				G[l].setRowVector(6,4);
				G[l] = ::Generator(HubbardModel::set_operators(F, 0.)).row(5);
				if (CALC_SQUARE == true)
				{
					Gsq[l].setRowVector(6*6,4);
					Gsq[l] = tensor_product(G[l],G[l]);
				}
			}
			else if (l==this->N_sites-1)
			{
				G[l].setColVector(6,4);
				G[l] = ::Generator(HubbardModel::set_operators(F, 0.)).col(0);
				if (CALC_SQUARE == true)
				{
					Gsq[l].setColVector(6*6,4);
					Gsq[l] = tensor_product(G[l],G[l]);
				}
			}
			else
			{
				G[l].setMatrix(6,4);
				G[l] = ::Generator(HubbardModel::set_operators(F, 0.));
				if (CALC_SQUARE == true)
				{
					Gsq[l].setMatrix(6*6,4);
					Gsq[l] = tensor_product(G[l],G[l]);
				}
			}
		}
	}
	
	this->Daux = 6;
	
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
TransverseKondoModel (size_t L_input, double J_input, initializer_list<size_t> imploc_input, initializer_list<double> Bzloc_input, initializer_list<double> Bxloc_input, double K_input, size_t D_input, bool CALC_SQUARE)
:TransverseKondoModel(L_input, J_input, vector<size_t>(begin(imploc_input),end(imploc_input)), 
                      vector<double>(begin(Bzloc_input),end(Bzloc_input)), 
                      vector<double>(begin(Bxloc_input),end(Bxloc_input)), K_input, D_input, CALC_SQUARE)
{}

MpoQ<1> TransverseKondoModel::
Simp (SPINOP_LABEL Sa, size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << Sa << "_imp(" << locx << "," << locy << ")";
	MpoQ<1> Mout(N_sites, N_legs, locBasis(), {0}, Nlabel, ss.str());
	MatrixXd IdSub(F.dim(),F.dim()); IdSub.setIdentity();
	Mout.setLocal(locx, kroneckerProduct(S.Scomp(Sa,locy),IdSub));
	return Mout;
}

MpoQ<1> TransverseKondoModel::
Ssub (SPINOP_LABEL Sa, size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << Sa << "_sub(" << locx << "," << locy << ")";
	MpoQ<1> Mout(N_sites, N_legs, locBasis(), {0}, Nlabel, ss.str());
	MatrixXd IdImp(qloc[locx].size()/F.dim(), qloc[locx].size()/F.dim()); IdImp.setIdentity();
	Mout.setLocal(locx, kroneckerProduct(IdImp, F.Scomp(Sa,locy)));
	return Mout;
}

MpoQ<1> TransverseKondoModel::
SimpSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2)
{
	assert(locx1<N_sites and locx2<N_sites and locy1<N_legs and locy2<N_legs);
	stringstream ss;
	ss << SOP1 << "(" << locx1 << "," << locy1 << ")" << SOP2 << "(" << locx2 << "," << locy2 << ")";
	MpoQ<1> Mout(N_sites, N_legs, locBasis(), {0}, Nlabel, ss.str());
	MatrixXd IdSub(F.dim(),F.dim()); IdSub.setIdentity();
	MatrixXd IdImp(MpoQ<1>::qloc[locx2].size()/F.dim(), MpoQ<1>::qloc[locx2].size()/F.dim()); IdImp.setIdentity();
	Mout.setLocal({locx1,locx2}, {kroneckerProduct(S.Scomp(SOP1,locy1),IdSub), 
	                              kroneckerProduct(IdImp,F.Scomp(SOP2,locy2))}
	             );
	return Mout;
}

MpoQ<1> TransverseKondoModel::
SimpSimp (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2)
{
	assert(locx1<N_sites and locx2<N_sites and locy1<N_legs and locy2<N_legs);
	stringstream ss;
	ss << SOP1 << "(" << locx1 << "," << locy1 << ")" << SOP2 << "(" << locx2 << "," << locy2 << ")";
	MpoQ<1> Mout(N_sites, N_legs, locBasis(), {0}, Nlabel, ss.str());
	MatrixXd IdSub(F.dim(),F.dim()); IdSub.setIdentity();
	Mout.setLocal({locx1,locx2}, {kroneckerProduct(S.Scomp(SOP1,locy1),IdSub), 
	                              kroneckerProduct(S.Scomp(SOP2,locy2),IdSub)}
	             );
	return Mout;
}

MpoQ<1> TransverseKondoModel::
SsubSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2)
{
	assert(locx1<N_sites and locx2<N_sites and locy1<N_legs and locy2<N_legs);
	stringstream ss;
	ss << SOP1 << "(" << locx1 << "," << locy1 << ")" << SOP2 << "(" << locx2 << "," << locy2 << ")";
	MpoQ<1> Mout(N_sites, N_legs, locBasis(), {0}, Nlabel, ss.str());
	MatrixXd IdImp1(MpoQ<1>::qloc[locx1].size()/F.dim(), MpoQ<1>::qloc[locx1].size()/F.dim()); IdImp1.setIdentity();
	MatrixXd IdImp2(MpoQ<1>::qloc[locx2].size()/F.dim(), MpoQ<1>::qloc[locx2].size()/F.dim()); IdImp2.setIdentity();
	Mout.setLocal({locx1,locx2}, {kroneckerProduct(IdImp1,F.Scomp(SOP1,locy1)), 
	                              kroneckerProduct(IdImp2,F.Scomp(SOP2,locy2))}
	             );
	return Mout;
}

MpoQ<1> TransverseKondoModel::
hopping (size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << "hopping" << "(" << locx << "," << locy << ")";
	vector<SuperMatrix<double> > G(N_sites);
	for (size_t l=0; l<N_sites; ++l)
	{
//		auto Gloc = HubbardModel::Generator(0,0);
		auto it = find(imploc.begin(),imploc.end(),l);
		HamiltonianTermsXd Terms;
		if (it != imploc.end())
		{
//			Gloc = KondoModel::Generator(0,0,0,-1.,0,0,D);
			Terms = KondoModel::set_operators(F,S, 0.,0.,MatrixXd::Identity(1,1),0.,0.,0.,0.);
		}
		else
		{
			Terms = HubbardModel::set_operators(F, 0.);
		}
		auto Gloc = ::Generator(Terms);
		
		if (l==0)
		{
			G[l].setRowVector(6,Gloc.D());
			if (l == locx or l==locx-1)
			{
				G[l] = Gloc.row(5);
			}
			else
			{
				G[l].setZero();
				G[l](0,5).setIdentity();
			}
		}
		else if (l == N_sites-1)
		{
			G[l].setColVector(6,Gloc.D());
			if (l == locx or l==locx+1)
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
			if (l==locx or l==locx-1 or l==locx+1)
			{
				G[l] = Gloc;
				if (l == locx-1)
				{
					G[l](1,0).setZero();
					G[l](2,0).setZero();
					G[l](3,0).setZero();
					G[l](4,0).setZero();
				}
				else if (l == locx+1)
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
	MpoQ<1> Mout(N_sites, N_legs, G, locBasis(), {0}, Nlabel, ss.str());
	
	return Mout;
}

};

#endif

#ifndef DOUBLEBANDHUBBARD
#define DOUBLEBANDHUBBARD

#include "HubbardModel.h"

class DoubleBandHubbardModel : public OccNumVecSpaceFloor
{
//friend class DoubleBandPeierls;
//template<typename Scalar> friend void HxV_PotShPlus (const DoubleBandHubbardModel &H, const Matrix<Scalar,Dynamic,1> &Vin, Matrix<Scalar,Dynamic,1> &Vout);
//template<typename Scalar> friend void HxV (const DoubleBandHubbardModel &H, const Matrix<Scalar,Dynamic,1> &Vin, Matrix<Scalar,Dynamic,1> &Vout);
//template<typename Scalar> friend void HxV (const DoubleBandHubbardModel &H, Matrix<Scalar,Dynamic,1> &Vinout);
//template<typename Scalar> friend void HxV_PotShPlus (const DoubleBandPeierls &H, const Matrix<Scalar,Dynamic,1> &Vin, VectorXcd &Vout);
//template<typename> friend class PolynomialBath;

public:
	
	template<typename Utype>
	DoubleBandHubbardModel (int L_input, int Nup1_input, int Ndn1_input, double U1_input, double E1_input, 
	                                     int Nup2_input, int Ndn2_input, Utype U2_input,
	                        double U01_input, double V11_input=0.,
	                        BOUNDARY_CONDITION BC_input = BC_PERIODIC,
	                        MEM_MANAGEMENT MM_input = MM_DYN,
	                        DIM dim_input = DIM1);
	
	DoubleBandHubbardModel (int L_input, int Nup1_input, int Ndn1_input, InteractionParams params1_input, 
	                                     int Nup2_input, int Ndn2_input, InteractionParams params2_input, 
	                        double U01_input, double V11_input=0.,
	                        BOUNDARY_CONDITION BC_input = BC_PERIODIC,
	                        MEM_MANAGEMENT MM_input = MM_DYN,
	                        DIM dim_input = DIM1);
	
	//--------<info>--------
	void print_basis() const;
	string info (size_t N_indent=1) const;
	double memory (MEMUNIT memunit=GB) const;
	//--------</info>--------
	
	//--------<access>--------
//	double trace() const;
	int double_occupation01 (size_t stateNr) const;
	int double_occupation01 (size_t stateNr1, size_t stateNr2) const;
	std::shared_ptr<OccNumBasis> basis_ptr (int b, SPIN_INDEX sigma) const {return band[b].basis_ptr(sigma);}
	
	size_t subdim (int b, SPIN_INDEX sigma) const {return band[b].subdim(sigma);}
	size_t subdim (int b) const                   {return band[b].dim();}
	int    subN   (int b, SPIN_INDEX sigma) const {return band[b].subN(sigma);}
	
	const VectorXd       &U01vector() const                          {return storedU01vector;}
	const SparseMatrixXd &subHmatrix (int b, SPIN_INDEX sigma) const {return band[b].subHmatrix(sigma);}
	const SparseMatrixXd &subHmatrix (int b) const                   {return band[b].Hmatrix();}
	const VectorXd       &subUvector (int b) const                   {return band[b].Uvector();}
	
	const SparseMatrixXd &Hforward() const {return storedHforward;}
	
	bool check_COULOMB (int b) const {return band[b].check_COULOMB();}
	bool check_COULOMB01()     const {return COULOMB_CHECK01;}
	
	SparseMatrixXd d (int i, int b=1) const;
	SparseMatrixXd n (int i, int b=1) const;
	//--------</access>--------
	
protected:
	
	std::array<HubbardModel,2> band;
	void make_Hmatrix();
	void make_U01vector();
	
	void construct (int Nup1_input, int Ndn1_input, InteractionParams params1_input, 
	                int Nup2_input, int Ndn2_input, InteractionParams params2_input, 
	                double U01_input, double V11_input,
	                MEM_MANAGEMENT MM_input);
	
	size_t stateNr (size_t i1, size_t i2) const;
	size_t substateNr (size_t stateNr, int band_index) const;
	
	double U01;
	bool COULOMB_CHECK01;
	void calc_HubbardU01 (SparseMatrixXd &M);
	VectorXd storedU01vector;
	
	SparseMatrixXd storedHforward; // heirloom for Peierls
};

template<typename Scalar>
void HxV (const DoubleBandHubbardModel &H, const Matrix<Scalar,Dynamic,1> &Vin, Matrix<Scalar,Dynamic,1> &Vout)
{
	if (H.check_HMATRIX() == false)
	{
		HxV_PotShPlus(H, Vin,Vout);
	}
	else
	{
		Vout.noalias() = H.Hmatrix().selfadjointView<Upper>() * Vin;
	}
}

template<typename Scalar>
void HxV (const DoubleBandHubbardModel &H, Matrix<Scalar,Dynamic,1> &Vinout)
{
	if (H.check_HMATRIX() == false)
	{
		Matrix<Scalar,Dynamic,1> Vtmp;
		HxV_PotShPlus(H, Vinout,Vtmp);
		Vinout = Vtmp;
	}
	else
	{
		Vinout = H.Hmatrix().selfadjointView<Upper>() * Vinout;
	}
}

//--------<constructors>--------
void DoubleBandHubbardModel::
construct (int Nup1_input, int Ndn1_input, InteractionParams params1_input, int Nup2_input, int Ndn2_input, InteractionParams params2_input, double U01_input, double V11_input, MEM_MANAGEMENT MM_input)
{
	U01 = U01_input;
	COULOMB_CHECK01 = (gsl_fcmp(U01,0.,1e-10) == 0) ? false : true;
	infolabel = "DoubleBandHubbardModel";
	
	N_states = gsl_sf_choose(N_sites,Nup1_input)*gsl_sf_choose(N_sites,Ndn1_input) 
	          *gsl_sf_choose(N_sites,Nup2_input)*gsl_sf_choose(N_sites,Ndn2_input);
	
	if ((MM_input==MM_DYN and N_states<LARGE_HUBBARD_SPACE) or MM_input==MM_FULL)
	{
		band[0] = HubbardModel(L_edge, Nup1_input, Ndn1_input, params1_input, BOUNDARIES, MM_FULL,BS_UPPER, spacedim);
		band[1] = HubbardModel(L_edge, Nup2_input, Ndn2_input, params2_input, BOUNDARIES, MM_FULL,BS_UPPER, spacedim);
		if (V11_input!=0.) {band[1].switch_HubbardV(V11_input);}
		make_Hmatrix();
	}
	else if ((MM_input==MM_DYN and N_states>LARGE_HUBBARD_SPACE) or MM_input==MM_SUB)
	{
		band[0] = HubbardModel(L_edge, Nup1_input, Ndn1_input, params1_input, BOUNDARIES, MM_SUB,BS_UPPER, spacedim);
		band[1] = HubbardModel(L_edge, Nup2_input, Ndn2_input, params2_input, BOUNDARIES, MM_SUB,BS_UPPER, spacedim);
		if (V11_input!=0.) {band[1].switch_HubbardV(V11_input);}
		if (COULOMB_CHECK01 == true) {make_U01vector();}
	}
	
	band[0].infolabel = "band 0";
	band[1].infolabel = "band 1";
}

template<typename Utype>
DoubleBandHubbardModel::
DoubleBandHubbardModel (int L_input, int Nup1_input, int Ndn1_input, double U1_input, double E1_input, int Nup2_input, int Ndn2_input, Utype U2_input, double U01_input, double V11_input, BOUNDARY_CONDITION BC_input, MEM_MANAGEMENT MM_input, DIM dim_input)
:OccNumVecSpaceFloor (L_input, Nup1_input+Ndn1_input+Nup2_input+Ndn2_input, BC_input, MM_input, dim_input)
{
	InteractionParams p1, p2;
	
	p1.set_U(U1_input);
	if (E1_input==0.) {p1.set_onsiteEnergies_toZero();}
	else              {p1.set_onsiteEnergies(N_sites,E1_input);}
//	p1.set_hoppings_toNN();
	
	p2.set_U(U2_input);
	p2.set_onsiteEnergies_toZero();
	p2.set_hoppings_toNN();
	
	construct(Nup1_input,Ndn1_input,p1, Nup2_input,Ndn2_input,p2, U01_input,V11_input, MM_input);
}

DoubleBandHubbardModel::
DoubleBandHubbardModel (int L_input, int Nup1_input, int Ndn1_input, InteractionParams params1_input, int Nup2_input, int Ndn2_input, InteractionParams params2_input, double U01_input, double V11_input, BOUNDARY_CONDITION BC_input, MEM_MANAGEMENT MM_input, DIM dim_input)
:OccNumVecSpaceFloor (L_input, Nup1_input+Ndn1_input+Nup2_input+Ndn2_input, BC_input, MM_input, dim_input)
{
	construct(Nup1_input,Ndn1_input,params1_input, Nup2_input,Ndn2_input,params2_input, U01_input,V11_input, MM_input);
}
//--------</constructors>--------

//--------------<construct stuff>--------------
void DoubleBandHubbardModel::
make_Hmatrix()
{
//	direct_sum(band[0].storedHmatrix, band[1].storedHmatrix, storedHmatrix);
	direct_sum(subHmatrix(0), subHmatrix(1), storedHmatrix);
//	cout << "making Hubbard-T01 " << HMATRIX_FORMAT << endl;
	
	band[0].kill_Hmatrix();
	band[1].kill_Hmatrix();
	
	if (COULOMB_CHECK01 == true)
	{
		SparseMatrixXd Mtemp(N_states,N_states);
		calc_HubbardU01(Mtemp);
		storedHmatrix += Mtemp;
	}
	
	storedHmatrix.makeCompressed();
	HMATRIX_CHECK = true;
}

void DoubleBandHubbardModel::
calc_HubbardU01 (SparseMatrixXd &Mout)
{
//	cout << "adding Hubbard-U01 " << HMATRIX_FORMAT << endl;
	Mout.resize(N_states,N_states);
	for (SparseMatrixXd::Index state=0; state<N_states; ++state)
	{
		Mout.insert(state,state) = U01*double_occupation01(state);
	}
}

void DoubleBandHubbardModel::
make_U01vector()
{
//	cout << "making Hubbard-U01vector " << HMATRIX_FORMAT << endl;
	storedU01vector.resize(N_states);
	#pragma omp parallel for
	for (EIGEN_DEFAULT_DENSE_INDEX_TYPE state=0; state<N_states; ++state)
	{
		storedU01vector(state) = U01*double_occupation01(state);
	}
}
//--------------</construct stuff>--------------

//--------<info>--------
string DoubleBandHubbardModel::
info (size_t N_indent) const
{
	string indent(N_indent,' ');
	stringstream ss;
	ss << infolabel << ": U01=" << U01
	   << ", states=" << N_states
	   << ", " << mem_info() << endl;
	ss << indent << "•" << band[0].info() << endl;
	ss << indent << "•" << band[1].info();
	return ss.str();
}

double DoubleBandHubbardModel::
memory (MEMUNIT memunit) const
{
	double res=0.;
	
	res += band[0].memory(memunit);
	res += band[1].memory(memunit);
	
	res += calc_memory(storedHmatrix,memunit);
	res += calc_memory(storedU01vector,memunit);
	
	res += calc_memory(storedHforward,memunit);
	
	return res;
}

void DoubleBandHubbardModel::
print_basis() const
{
	for (size_t i1=0; i1<band[0].dim(); ++i1)
	for (size_t i2=0; i2<band[1].dim(); ++i2)
	{
		cout << stateNr(i1,i2)
		<< "\t\033[22;34m0↑\033[0m:" << band[0].basis_state(i1,UP)
		<< " \033[22;34m0↓\033[0m:" << band[0].basis_state(i1,DN)
		<< " \033[22;34md\033[0m=" << band[0].double_occupation(i1)
		<< "\t\033[22;31m1↑\033[0m:" << band[1].basis_state(i2,UP)
		<< " \033[22;31m1↓\033[0m:" << band[1].basis_state(i2,DN)
		<< " \033[22;31md\033[0m=" << band[1].double_occupation(i2)
		<< "\t|d=" << double_occupation01(i1,i2)
		<< endl;
	}
}
//--------</info>--------

//--------<access>--------
//inline double DoubleBandHubbardModel::
//trace() const
//{
//	return band[0].trace()*band[1].dim() + band[0].dim()*band[1].trace() + storedU01vector.sum();
//}

inline size_t DoubleBandHubbardModel::
stateNr (size_t i1, size_t i2) const
{
	return i1*band[1].dim()+i2;
}

inline size_t DoubleBandHubbardModel::
substateNr (size_t stateNr, int band_index) const
{
	return (band_index == 1) ? stateNr/band[1].dim() : stateNr%band[1].dim();
}

inline int DoubleBandHubbardModel::
double_occupation01 (size_t stateNr) const
{
	return double_occupation01(substateNr(stateNr,1), substateNr(stateNr,2));
}

int DoubleBandHubbardModel::
double_occupation01 (size_t stateNr1, size_t stateNr2) const
{
	int out = 0;
	OccNumVector s0UP = band[0].basis_state(stateNr1,UP);
	OccNumVector s0DN = band[0].basis_state(stateNr1,DN);
	OccNumVector s1UP = band[1].basis_state(stateNr2,UP);
	OccNumVector s1DN = band[1].basis_state(stateNr2,DN);
	out += (s0UP & s1UP).count();
	out += (s0UP & s1DN).count();
	out += (s0DN & s1UP).count();
	out += (s0DN & s1DN).count();
	return out;
}

SparseMatrixXd DoubleBandHubbardModel::
d (int i, int b) const
{
	SparseMatrixXd Mout;
	if (b==0)
	{
		MkronI(band[0].d(i), band[1].dim(), Mout);
	}
	else
	{
		IkronM(band[0].dim(), band[1].d(i), Mout);
	}
	return Mout;
}

SparseMatrixXd DoubleBandHubbardModel::
n (int i, int b) const
{
	SparseMatrixXd Mout;
	if (b==0)
	{
		MkronI(band[0].n(i), band[1].dim(), Mout);
	}
	else
	{
		IkronM(band[0].dim(), band[1].n(i), Mout);
	}
	return Mout;
}
//--------</access>--------

template<typename Scalar>
void HxV_PotShPlus (const DoubleBandHubbardModel &H, const Matrix<Scalar,Dynamic,1> &Vin, Matrix<Scalar,Dynamic,1> &Vout)
{
	PotShPlus_algorithm<SparseMatrixXd,Matrix<Scalar,Dynamic,1>,HxV>
	(
		1,
//		H.band[0].spin[UP]->dim(),
//		H.band[0].spin[DN]->dim() * H.band[1].dim(),
//		H.band[0].spin[UP]->storedHmatrix,
		H.subdim(0,UP),
		H.subdim(0,DN) * H.subdim(1),
		H.subHmatrix(0,UP),
		Vin, Vout
	);
	
	Matrix<Scalar,Dynamic,1> Vtmp;
	
	PotShPlus_algorithm<SparseMatrixXd,Matrix<Scalar,Dynamic,1>,HxV>
	(
//		H.band[0].spin[UP]->dim(),
//		H.band[0].spin[DN]->dim(),
//		H.band[1].dim(), 
//		H.band[0].spin[DN]->storedHmatrix,
		H.subdim(0,UP),
		H.subdim(0,DN),
		H.subdim(1),
		H.subHmatrix(0,DN),
		Vin, Vtmp
	);
	Vout += Vtmp;
	
	PotShPlus_algorithm<SparseMatrixXd,Matrix<Scalar,Dynamic,1>,HxV>
	(
//		H.band[0].dim(),
//		H.band[1].spin[UP]->dim(),
//		H.band[1].spin[DN]->dim(), 
//		H.band[1].spin[UP]->storedHmatrix,
		H.subdim(0),
		H.subdim(1,UP),
		H.subdim(1,DN),
		H.subHmatrix(1,UP),
		Vin, Vtmp
	);
	Vout += Vtmp;
	
	PotShPlus_algorithm<SparseMatrixXd,Matrix<Scalar,Dynamic,1>,HxV>
	(
//		H.band[0].dim() * H.band[1].spin[UP]->dim(),
//		H.band[1].spin[DN]->dim(),
//		1, 
//		H.band[1].spin[DN]->storedHmatrix,
		H.subdim(0) * H.subdim(1,UP),
		H.subdim(1,DN),
		1,
		H.subHmatrix(1,DN),
		Vin, Vtmp
	);
	Vout += Vtmp;
	
	// U in band 0
	if (H.check_COULOMB(0) == true)
	{
		PotShPlus_algorithm<VectorXd,Matrix<Scalar,Dynamic,1>,HxV>
		(
//			1,
//			H.band[0].dim(),
//			H.band[1].dim(), 
//			H.band[0].storedUvector,
			1,
			H.subdim(0),
			H.subdim(1),
			H.subUvector(0),
			Vin, Vtmp
		);
		Vout += Vtmp;
	}
	
	// U in band 1
	if (H.check_COULOMB(1) == true)
	{	
		PotShPlus_algorithm<VectorXd,Matrix<Scalar,Dynamic,1>,HxV>
		(
//			H.band[0].dim(),
//			H.band[1].dim(),
//			1, 
//			H.band[1].storedUvector,
			H.subdim(0),
			H.subdim(1),
			1,
			H.subUvector(1),
			Vin, Vtmp
		);
		Vout += Vtmp;
	}
	
	// interband U
	if (H.check_COULOMB01() == true)
	{
//		Vout.noalias() += H.storedU01vector.asDiagonal() * Vin;
		Vout.noalias() += H.U01vector().asDiagonal() * Vin;
	}
}

#endif

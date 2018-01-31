#ifndef SPINOR
#define SPINOR

#include <array>

#include "OccNumVecSpaceFloor.h"
#include "HoppingParticles.h"
#include "TensorProducts.h"

class HubbardModel : public OccNumVecSpaceFloor
{
//template<typename Scalar> friend void HxV (const HubbardModel &H, const Matrix<Scalar,Dynamic,1> &Vin, Matrix<Scalar,Dynamic,1> &Vout);
//template<typename Scalar> friend void HxV (const HubbardModel &H, Matrix<Scalar,Dynamic,1> &Vinout);
//friend class DoubleBandHubbardModel;
//template<typename Scalar> friend void HxV_PotShPlus (const DoubleBandHubbardModel &H, const Matrix<Scalar,Dynamic,1> &Vin, Matrix<Scalar,Dynamic,1> &Vout);
//template<typename Scalar> friend void HxV (const DoubleBandHubbardModel &H, const Matrix<Scalar,Dynamic,1> &Vin, Matrix<Scalar,Dynamic,1> &Vout);
//template<typename Scalar> friend void HxV (const DoubleBandHubbardModel &H, Matrix<Scalar,Dynamic,1> &Vinout);
friend class DoubleBandPeierls;
//template<typename Scalar> friend void HxV_PotShPlus (const DoubleBandPeierls &H, const Matrix<Scalar,Dynamic,1> &Vin, VectorXcd &Vout);

public:
	
	HubbardModel() {};
	
	HubbardModel (int L_input, int Nup_input, int Ndn_input, double U_input, 
	              BOUNDARY_CONDITION BC_input=BC_PERIODIC, 
	              MEM_MANAGEMENT MM_input = MM_DYN,
	              BOND_STORAGE BS_input = BS_UPPER,
	              DIM dim_input=DIM1);
	
	HubbardModel (int L_input, int Nup_input, int Ndn_input, InteractionParams params_input, 
	              BOUNDARY_CONDITION BC_input = BC_PERIODIC, 
	              MEM_MANAGEMENT MM_input = MM_DYN,
	              BOND_STORAGE BS_input = BS_UPPER,
	              DIM dim_input = DIM1);
	
	HubbardModel (int L_input, int Nup_input, int Ndn_input, double U_input, 
	              const SparseMatrixXd &BondMatrix_input, 
	              BOUNDARY_CONDITION BC_input=BC_PERIODIC, 
	              MEM_MANAGEMENT MM_input = MM_DYN,
	              BOND_STORAGE BS_input = BS_UPPER,
	              DIM dim_input=DIM1);
	
	//--------<info>--------
	void print_basis() const;
	string info() const;
	double memory (MEMUNIT memunit=GB) const;
	//--------</info>--------

	//--------<access>--------
//	double trace() const;
	int double_occupation (size_t stateNr) const;
	OccNumVector double_occupation_local (size_t stateNr) const;
	OccNumVector basis_state (size_t stateNr, SPIN_INDEX sigma) const;
	std::shared_ptr<OccNumBasis> basis_ptr (SPIN_INDEX sigma) const {return spin[sigma]->basis_ptr();}
	
	size_t subdim (SPIN_INDEX sigma) const {return spin[sigma]->dim();}
	int    subN   (SPIN_INDEX sigma) const {return spin[sigma]->N();}
	
	const SparseMatrixXd &subHmatrix (SPIN_INDEX sigma) const {return spin[sigma]->Hmatrix();}
	const VectorXd &Uvector() const                           {return storedUvector;}
	bool  check_COULOMB() const                               {return COULOMB_CHECK;};
	
	void scale (double factor=1., double offset=0.);
	//--------</access>--------

	//--------<operators>--------
	#include "HubbardOperatorsList.h"
	//--------</operators>--------
	
	//--------<extensions>--------
	void switch_HubbardV (double V);
	void switch_corrhop (double T2, BOND_STORAGE BS_input);
	//--------</extensions>--------
	
protected:
	
	std::array<std::shared_ptr<HoppingParticles>,2> spin;
	
	void construct (int Nup_input, int Ndn_input, const InteractionParams params_input, MEM_MANAGEMENT MM_input, BOND_STORAGE BS_input);
	void make_Tmatrix();
	void make_Hmatrix();
	void make_Uvector();
	
	size_t stateNr (size_t iup, size_t idn) const;
	size_t substateNr (size_t stateNr, SPIN_INDEX sigma) const;

	double U, V;
	vector<double> Uvec;
	bool COULOMB_CHECK;
	void calc_HubbardU (SparseMatrixXd &Mout) const;
	VectorXd storedUvector;
};

template<typename Scalar>
void HxV (const HubbardModel &H, const Matrix<Scalar,Dynamic,1> &Vin, Matrix<Scalar,Dynamic,1> &Vout)
{
	Stopwatch<> Chronos;
	if (H.check_HMATRIX() == false)
	{
		HxV_PotShPlus(H, Vin,Vout);
//		lout << "HxV(Hubbard,PotSh+): " << Chronos.time() << " s" << endl;
	}
	else
	{
		Vout.noalias() = H.Hmatrix().selfadjointView<Upper>() * Vin;
//		lout << "HxV(Hubbard,fullMatrix): " << Chronos.time() << " s" << endl;
	}
}

template<typename Scalar>
void HxV (const HubbardModel &H, Matrix<Scalar,Dynamic,1> &Vinout)
{
//	if (H.HMATRIX_CHECK == false)
	if (H.check_HMATRIX() == false)
	{
//		size_t dimUP = H.spin[UP]->dim();
//		size_t dimDN = H.spin[DN]->dim();
//		
//		Matrix<Scalar,Dynamic,1> Vtmp1;
//		Matrix<Scalar,Dynamic,1> Vtmp2;
//		
//		PotShPlus_algorithm(1,dimUP,dimDN, H.spin[UP]->storedHmatrix, Vinout,Vtmp1);
//		PotShPlus_algorithm(dimUP,dimDN,1, H.spin[DN]->storedHmatrix, Vinout,Vtmp2);
//		
//		if (H.COULOMB_CHECK==true)
//		{
//			Vinout = H.storedUvector.asDiagonal()*Vinout;
//		}
//		Vinout.noalias() += Vtmp1 + Vtmp2;
		Matrix<Scalar,Dynamic,1> Vtmp;
		HxV(H, Vinout,Vtmp);
		Vinout = Vtmp;
	}
	else
	{
//		Vinout = H.storedHmatrix.selfadjointView<Upper>() * Vinout;
		Vinout = H.Hmatrix().selfadjointView<Upper>() * Vinout;
	}
}

template<typename Scalar>
void polyIter (const HubbardModel &H, const Matrix<Scalar,Dynamic,1> &Vin1, double polyB, const Matrix<Scalar,Dynamic,1> &Vin2, Matrix<Scalar,Dynamic,1> &Vout)
{
	Stopwatch<> Chronos;
	HxV(H,Vin1,Vout);
	Vout -= polyB * Vin2;
	lout << Chronos.info("polyIter(HubbardModel)") << endl;
}

//template<typename Scalar>
//void chebIter (const HubbardModel &H, const Matrix<Scalar,Dynamic,1> &Vin1, const Matrix<Scalar,Dynamic,1> &Vin2, Matrix<Scalar,Dynamic,1> &Vout)
//{
//	HxV(H,Vin1,Vout);
//	Vout -= Vin2;
//}

//--------------<constructors>--------------
void HubbardModel::
construct (int Nup_input, int Ndn_input, InteractionParams params_input, MEM_MANAGEMENT MM_input, BOND_STORAGE BS_input)
{
	V = 0.;
	COULOMB_CHECK = params_input.COULOMB_CHECK;
	
	spin[UP] = std::make_shared<HoppingParticles>(L_edge, Nup_input, params_input, BOUNDARIES, BS_input, spacedim, FERMIONS);
	spin[DN] = (Nup_input==Ndn_input) ? spin[UP] 
	         : std::make_shared<HoppingParticles>(L_edge, Ndn_input, params_input, BOUNDARIES, BS_input, spacedim, FERMIONS);
	
	infolabel = "HubbardModel";
	N_states = spin[UP]->dim() * spin[DN]->dim();
	
	if (COULOMB_CHECK == true)
	{
		U    = params_input.U;
		Uvec = params_input.Uvec;
	}
	
	if ((MM_input==MM_DYN and N_states<LARGE_HUBBARD_SPACE) or MM_input==MM_FULL)
	{
		make_Hmatrix();
	}
	else if ((MM_input==MM_DYN and N_states>LARGE_HUBBARD_SPACE) or MM_input==MM_SUB)
	{
		if (COULOMB_CHECK == true) {make_Uvector();}
	}
}

HubbardModel::
HubbardModel (int L_input, int Nup_input, int Ndn_input, double U_input, 
              BOUNDARY_CONDITION BC_input, MEM_MANAGEMENT MM_input, BOND_STORAGE BS_input, DIM dim_input)
:OccNumVecSpaceFloor(L_input, Nup_input+Ndn_input, BC_input, MM_input, dim_input, FERMIONS)
{
	InteractionParams p;
	p.set_U(U_input);
	p.set_onsiteEnergies_toZero();
	p.set_hoppings_toNN();
	construct(Nup_input, Ndn_input, p, MM_input, BS_input);
}

HubbardModel::
HubbardModel (int L_input, int Nup_input, int Ndn_input, InteractionParams params_input, 
              BOUNDARY_CONDITION BC_input, MEM_MANAGEMENT MM_input, BOND_STORAGE BS_input, DIM dim_input)
:OccNumVecSpaceFloor(L_input, Nup_input+Ndn_input, BC_input, MM_input, dim_input, FERMIONS)
{
	construct(Nup_input, Ndn_input, params_input, MM_input, BS_input);
}

HubbardModel::
HubbardModel (int L_input, int Nup_input, int Ndn_input, double U_input, const SparseMatrixXd &BondMatrix_input, 
              BOUNDARY_CONDITION BC_input, MEM_MANAGEMENT MM_input, BOND_STORAGE BS_input, DIM dim_input)
:OccNumVecSpaceFloor(L_input, Nup_input+Ndn_input, BC_input, MM_input, dim_input, FERMIONS)
{
	InteractionParams p;
	p.set_U(U_input);
	p.set_onsiteEnergies_toZero();
	p.set_hoppings_toNN();
	
	// construct with BondMatrix:
	V = 0.;
	COULOMB_CHECK = p.COULOMB_CHECK;
	
	spin[UP] = std::make_shared<HoppingParticles>(L_edge, Nup_input, BondMatrix_input, BOUNDARIES, BS_input, spacedim, FERMIONS);
	spin[DN] = (Nup_input==Ndn_input) ? spin[UP] 
	         : std::make_shared<HoppingParticles>(L_edge, Ndn_input, BondMatrix_input, BOUNDARIES, BS_input, spacedim, FERMIONS);
	
	infolabel = "HubbardModel (custom hopping)";
	N_states = spin[UP]->dim() * spin[DN]->dim();
	
	if (COULOMB_CHECK == true)
	{
		U    = p.U;
		Uvec = p.Uvec;
	}
	
	if ((MM_input==MM_DYN and N_states<LARGE_HUBBARD_SPACE) or MM_input==MM_FULL)
	{
		make_Hmatrix();
	}
	else if ((MM_input==MM_DYN and N_states>LARGE_HUBBARD_SPACE) or MM_input==MM_SUB)
	{
		if (COULOMB_CHECK == true) {make_Uvector();}
	}
}
//--------------</constructors>--------------

//--------------<construct stuff>--------------
inline void HubbardModel::
make_Tmatrix()
{
	direct_sum(spin[UP]->Hmatrix(), spin[DN]->Hmatrix(), storedHmatrix);
//	cout << "making Hubbard-T " << HMATRIX_FORMAT << endl;
}

void HubbardModel::
make_Hmatrix()
{
	make_Tmatrix();
	
	if (COULOMB_CHECK == true)
	{
//		cout << "adding Hubbard-U " << HMATRIX_FORMAT << endl;
		SparseMatrixXd Mtemp(N_states,N_states);
		calc_HubbardU(Mtemp);
		storedHmatrix += Mtemp;
	}
	
	storedHmatrix.makeCompressed();
	HMATRIX_CHECK = true;
}

void HubbardModel::
calc_HubbardU (SparseMatrixXd &Mout) const
{
	Mout.resize(N_states,N_states);
	// homogeneous U
	if (Uvec.size() == 0)
	{
		for (SparseMatrixXd::Index state=0; state<N_states; ++state)
		{
			Mout.insert(state,state) = U*double_occupation(state);
		}
	}
	// inhomogeneous U
	else
	{
		for (SparseMatrixXd::Index state=0; state<N_states; ++state)
		{
			OccNumVector d = double_occupation_local(state);
			size_t i = d.find_first();
			double Usum = Uvec[i];
			while (i != OccNumVector::npos)
			{
				i = d.find_next(i);
				Usum += Uvec[i];
			}
			Mout.insert(state,state) = Usum;
		}
	}
}

void HubbardModel::
make_Uvector()
{
//	cout << "making Hubbard-Uvector " << HMATRIX_FORMAT << endl;
	storedUvector.resize(N_states);
	if (Uvec.size() == 0)
	{
//		#pragma omp parallel for
		for (EIGEN_DEFAULT_DENSE_INDEX_TYPE state=0; state<N_states; ++state)
		{
			storedUvector(state) = U*double_occupation(state);
		}
	}
	else
	{
//		#pragma omp parallel for
		for (EIGEN_DEFAULT_DENSE_INDEX_TYPE state=0; state<N_states; ++state)
		{
			OccNumVector d = double_occupation_local(state);
			size_t i = d.find_first();
			double Usum = Uvec[i];
			while (i != OccNumVector::npos)
			{
				i = d.find_next(i);
				Usum += Uvec[i];
			}
			storedUvector(state) = Usum;
		}
	}
}
//--------------</construct stuff>--------------

//--------------<info>--------------
string HubbardModel::
info() const
{
	stringstream ss;
	ss << infolabel << ": ";

	ss << "U=";
	if (COULOMB_CHECK==true)
	{
		if (Uvec.size() == 0)
		{
			ss << U << ", ";
		}
		else
		{
			for (int l=0; l<N_sites; ++l)
			{
				ss << round(Uvec[l],1);
				if (l != N_sites-1) {ss << ",";}
				else {ss << " ";}
			}
		}
	}
	else {ss << 0 << ", ";}
	if (V!=0.) {ss << "V=" << V << ", ";}

	ss << spin[UP]->hopping_info() << ", "
	<< states_sites_info() << ", "
	<< "N↑=" << spin[UP]->N() << ", "
	<< "N↓=" << spin[DN]->N() << ", ";

	ss << mem_info();
	
	return ss.str();
}

double HubbardModel::
memory (MEMUNIT memunit) const
{
	double res=0.;
	
	// memory for spin subspaces, normally negligible
	res += spin[UP]->memory(memunit);
	if (spin[UP]->N() != spin[DN]->N())
	{
		res += spin[DN]->memory(memunit);
	}
	
	if (storedHmatrix.rows()>0)
	{
		res += calc_memory(storedHmatrix,memunit);
	}
	if (storedUvector.rows()>0)
	{
		res += calc_memory(storedUvector,memunit);
	}
	
	return res;
}

void HubbardModel::
print_basis() const
{
	for (size_t iup=0; iup<spin[UP]->dim(); ++iup)
	for (size_t idn=0; idn<spin[DN]->dim(); ++idn)
	{
		cout << stateNr(iup,idn) 
		<< "\tσ=↑: " << spin[UP]->basis_state(iup)
		<< " σ=↓: " << spin[DN]->basis_state(idn)
		<< " d=" << double_occupation(stateNr(iup,idn)) 
		<< endl;
	}
}
//--------------</info>--------------

//--------------<access>--------------
inline size_t HubbardModel::
stateNr (size_t iup, size_t idn) const
{
	return iup*spin[DN]->dim()+idn;
}

inline size_t HubbardModel::
substateNr (size_t stateNr, SPIN_INDEX sigma) const
{
	return (sigma == UP) ? stateNr/spin[DN]->dim() : stateNr%spin[DN]->dim();
}

inline int HubbardModel::
double_occupation (size_t stateNr) const
{
//	return (spin[UP]->basis[substateNr(stateNr,UP)] & spin[DN]->basis[substateNr(stateNr,DN)]).count();
	return (basis_state(stateNr,UP) & basis_state(stateNr,DN)).count();
}

inline OccNumVector HubbardModel::
double_occupation_local (size_t stateNr) const
{
	return basis_state(stateNr,UP) & basis_state(stateNr,DN);
}

inline OccNumVector HubbardModel::
basis_state (size_t stateNr, SPIN_INDEX sigma) const
{
//	return spin[sigma]->basis[substateNr(stateNr,sigma)];
	return spin[sigma]->basis_state(substateNr(stateNr,sigma));
}

void HubbardModel::
scale (double factor, double offset)
{
	if (check_HMATRIX() == true)
	{
		OccNumVecSpaceFloor::scale(factor,offset);
	}
	else
	{
		if (factor != 1.)
		{
			spin[UP]->scale(factor);
			if (subN(UP) != subN(DN)) {spin[DN]->scale(factor);}
			storedUvector *= factor;
		}
		if (offset != 0.)
		{
			if (check_COULOMB() == false)
			{
				storedUvector.resize(N_states);
				storedUvector.setConstant(offset);
				COULOMB_CHECK = true;
			}
			else
			{
				storedUvector.array() += offset;
			}
		}
	}
}
//--------------</access>--------------

//--------------<operators>--------------
#include "HubbardOperators.h"
//--------------</operators>--------------

//--------------<extensions>--------------
void HubbardModel::
switch_HubbardV (double V_input)
{
	V = V_input;
	Geometry Euklid;
	SparseMatrixXd BondMatrix = Euklid.BondMatrix(L_edge, spacedim, BOUNDARIES, 1., BS_FULL);
	
	SparseMatrixXd Mout(N_states,N_states);
//	Mout.reserve(N_states);
	vector<triplet> tripletList(N_states);
	
	#pragma omp parallel for
	for (size_t stateNr=0; stateNr<N_states; ++stateNr)
	{
		int substateNr_UP = substateNr(stateNr,UP);
		int substateNr_DN = substateNr(stateNr,DN);
		
		double res = 0.;
		
		for (int isite=0; isite<N_sites; ++isite)
		for (SparseMatrixXd::InnerIterator it(BondMatrix,isite); it and it.row()<=it.col(); ++it)
		{
			int jsite = it.row();
			int ni_UP = spin[UP]->basis_state(substateNr_UP)[isite];
			int ni_DN = spin[DN]->basis_state(substateNr_DN)[isite];
			int nj_UP = spin[UP]->basis_state(substateNr_UP)[jsite];
			int nj_DN = spin[DN]->basis_state(substateNr_DN)[jsite];
			res += V*(ni_UP+ni_DN)*(nj_UP+nj_DN);
		}
//		if (fabs(res)>1e-14) {Mout.insert(stateNr,stateNr) = res;}
		tripletList[stateNr] = triplet(stateNr,stateNr,res);
	}
	
	Mout.setFromTriplets(tripletList.begin(), tripletList.end());
	
	if (storedUvector.rows()>0)
	{
		storedUvector += Mout.diagonal();
	}
	else
	{
		storedHmatrix += Mout;
	}
}

void HubbardModel::
switch_corrhop (double T2, BOND_STORAGE BS_input)
{
	cout << "CORRHOP SWITCHING T2=" << T2 << endl;
	SparseMatrixXd Tcorr = tensor_product(spin[UP]->hopping_element(0,1,T2,false), (spin[DN]->n(0)+spin[DN]->n(1)).eval());
	if (BS_input != BS_FORWARDS)
	{
		Tcorr += tensor_product(spin[UP]->hopping_element(1,0,T2,false), (spin[DN]->n(0)+spin[DN]->n(1)).eval());
	}
	
	for (int l=1; l<N_sites-1; ++l)
	{
		Tcorr += tensor_product(spin[UP]->hopping_element(l,l+1,T2,false), (spin[DN]->n(l)+spin[DN]->n(l+1)).eval());
		if (BS_input != BS_FORWARDS)
		{
			Tcorr += tensor_product(spin[UP]->hopping_element(l+1,l,T2,false), (spin[DN]->n(l)+spin[DN]->n(l+1)).eval());
		}
	}
	if (BOUNDARIES == BC_PERIODIC)
	{
		Tcorr += tensor_product(spin[UP]->hopping_element(N_sites-1,0,T2,false), (spin[DN]->n(N_sites-1)+spin[DN]->n(0)).eval());
		if (BS_input != BS_FORWARDS)
		{
			Tcorr += tensor_product(spin[UP]->hopping_element(0,N_sites-1,T2,false), (spin[DN]->n(N_sites-1)+spin[DN]->n(0)).eval());
		}
	}
	
	for (int l=0; l<N_sites-1; ++l)
	{
		Tcorr += tensor_product((spin[UP]->n(l)+spin[UP]->n(l+1)).eval(), spin[DN]->hopping_element(l,l+1,T2,false));
		if (BS_input != BS_FORWARDS)
		{
			Tcorr += tensor_product((spin[UP]->n(l)+spin[UP]->n(l+1)).eval(), spin[DN]->hopping_element(l+1,l,T2,false));
		}
	}
	if (BOUNDARIES == BC_PERIODIC)
	{
		Tcorr += tensor_product((spin[UP]->n(N_sites-1)+spin[UP]->n(0)).eval(), spin[DN]->hopping_element(N_sites-1,0,T2,false));
		if (BS_input != BS_FORWARDS)
		{
			Tcorr += tensor_product((spin[UP]->n(N_sites-1)+spin[UP]->n(0)).eval(), spin[DN]->hopping_element(0,N_sites-1,T2,false));
		}
	}
	
	storedHmatrix += Tcorr;
}
//--------------</extensions>--------------

template<typename Scalar>
void HxV_PotShPlus (const HubbardModel &H, const Matrix<Scalar,Dynamic,1> &Vin, Matrix<Scalar,Dynamic,1> &Vout)
{
//	Stopwatch<> Chronos;
	size_t dimUP = H.subdim(UP);
	size_t dimDN = H.subdim(DN);
	Vout.resize(Vin.rows());
	Vout.setZero();
	
	PotShPlus_algorithm<SparseMatrixXd,Matrix<Scalar,Dynamic,1>,HxV>
	(
		1,
		dimUP,
		dimDN,
		H.subHmatrix(UP),
		Vin, Vout
	);
	
	Matrix<Scalar,Dynamic,1> Vtmp;
	PotShPlus_algorithm<SparseMatrixXd,Matrix<Scalar,Dynamic,1>,HxV>
	(
		dimUP,
		dimDN,
		1,
		H.subHmatrix(DN),
		Vin, Vtmp
	);
	Vout += Vtmp;
	
	if (H.check_COULOMB() == true)
	{
		Vout.noalias() += H.Uvector().asDiagonal() * Vin;
	}
//	lout << Chronos.info("HxV_PotShPlus") << endl;
}

//inline double HubbardModel::
//trace() const
//{
//	return spin[0]->trace()*spin[1]->dim() + spin[0]->dim()*spin[1]->trace() + storedUvector.sum();
//}

#endif

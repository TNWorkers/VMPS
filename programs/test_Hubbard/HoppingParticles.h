#ifndef HOPPINGPARTICLES
#define HOPPINGPARTICLES

#ifndef MAX_HOPPING_CHUNK_SIZE
#define MAX_HOPPING_CHUNK_SIZE 1e7
#endif

#include <gsl/gsl_sf_gamma.h> // needed for "L choose N"
#include <gsl/gsl_math.h>
//#include <algorithm> // needed for min, max

#include "InteractionParams.h"
#include "OccNumVecSpaceFloor.h"
#include "Chunkomatic.h"

// recursively writes (n choose k) combinations in colex-order into basis
void colex (size_t n, size_t k, OccNumVector state, vector<OccNumVector> &basis)
{
	if (n==0) {basis.push_back(state);}
	else
	{
		if (k<n) {colex(n-1,k,state,basis);}
		if (k>0) {state[n-1].flip(); colex(n-1,k-1,state,basis);}
	}
}

// forward declarations for future friendship:
//class HubbardModel; 
//class DoubleBandHubbardModel;
//class DoubleBandPeierls;

class HoppingParticles : public OccNumVecSpaceFloor
{
//friend class HubbardModel;
//template<typename Scalar> friend void HxV (const HubbardModel &H, const Matrix<Scalar,Dynamic,1> &Vin, Matrix<Scalar,Dynamic,1> &Vout);
//template<typename Scalar> friend void HxV (const HubbardModel &H, Matrix<Scalar,Dynamic,1> &Vinout);
//friend class DoubleBandHubbardModel;
//template<typename Scalar> friend void HxV_PotShPlus (const DoubleBandHubbardModel &H, const Matrix<Scalar,Dynamic,1> &Vin, Matrix<Scalar,Dynamic,1> &Vout);
//template<typename Scalar> friend void HxV (const DoubleBandHubbardModel &H, const Matrix<Scalar,Dynamic,1> &Vin, Matrix<Scalar,Dynamic,1> &Vout);
//template<typename Scalar> friend void HxV (const DoubleBandHubbardModel &H, Matrix<Scalar,Dynamic,1> &Vinout);
//friend class DoubleBandPeierls;
//template<typename Scalar> friend void HxV_PotShPlus (const DoubleBandPeierls &H, const Matrix<Scalar,Dynamic,1> &Vin, VectorXcd &Vout);
//template<typename,typename> friend class PolynomialBath;

public:
	
	HoppingParticles(){};
	
	HoppingParticles (int L_input, int N_input, double hopping_value=-1., 
	                  BOUNDARY_CONDITION BC_input = BC_PERIODIC, 
	                  BOND_STORAGE BS_input = BS_UPPER, 
	                  DIM dim_input = DIM1, 
	                  SPIN_STATISTICS SS_input = FERMIONS);
	
	HoppingParticles (int L_input, int N_input, InteractionParams params_input, 
	                  BOUNDARY_CONDITION BC_input = BC_PERIODIC, 
	                  BOND_STORAGE BS_input = BS_UPPER, 
	                  DIM dim_input = DIM1, 
	                  SPIN_STATISTICS SS_input = FERMIONS);
	
	HoppingParticles (int L_input, int N_input, const SparseMatrixXd &BondMatrix_input, 
	                  BOUNDARY_CONDITION BC_input = BC_PERIODIC, 
	                  BOND_STORAGE BS_input = BS_UPPER, 
	                  DIM dim_input = DIM1, 
	                  SPIN_STATISTICS SS_input = FERMIONS);
	
	void switch_Vnn (double Vnn_input);
	void switch_Vdiag (double Vdiag_input);
	
	//--------<info>--------
	string info() const;
	string hopping_info() const;
	double memory (MEMUNIT memunit=GB) const;
	//--------</info>--------
	
	//--------<operators>--------
	void n (int isite, SparseMatrixXd &Mout);
//	void Nop (SparseMatrixXd &Mout);
	void hopping_element (int origin, int destination, double hopping_value, bool MAKE_HERMITIAN, SparseMatrixXd &M);
	// writes a single element of the hopping matrix between sites origin and destination into a sparse matrix
	
	SparseMatrixXd n (int isite);
//	SparseMatrixXd Nop();
	SparseMatrixXd hopping_element (int origin, int destination, double hopping_value, bool MAKE_HERMITIAN);
	//--------</operators>--------
	
	// static functions for access from DestructoTron:
	static short int parity (const OccNumVector &V, int site1, int site2, SPIN_STATISTICS SS_input);
	static SparseMatrixXd::Index get_stateNr (const OccNumBasis &basis, const OccNumVector &V, SparseMatrixXd::Index guess);
	
private:
	
	void construct();
	void create_basis();
	
	SparseMatrixXd BondMatrix;
	VectorXd hoppings;
	VectorXd onsiteEnergies;
	bool ONSITE_CHECK;
	BOND_STORAGE HOPPING_FORMAT;
	double Vnn, Vdiag;
	
	void make_Tmatrix (SparseMatrixXd::Index state_min, SparseMatrixXd::Index state_max);
};

//--------------<constructors>--------------
void HoppingParticles::
construct()
{
	infolabel = (SPINSTAT==FERMIONS)? "HoppingFermions" : "HoppingBosons";
	N_states = gsl_sf_choose(N_sites,N_particles); // Hilbert space dimension is "L choose N"
	create_basis();
	
	storedHmatrix.resize(N_states,N_states);
	storedHmatrix.setZero();
	
	// on-site energies
	if (ONSITE_CHECK==true)
	{
		for (int i=0; i<onsiteEnergies.rows(); ++i)
		{
			BondMatrix.insert(i,i) += onsiteEnergies(i);
		}
	}
	
	// assemble hopping matrix in chunks to preserve memory
	SparseMatrixXd::Index N_chunks = max(static_cast<int>(ceil(N_states/MAX_HOPPING_CHUNK_SIZE)),1);
	Chunkomatic<SparseMatrixXd::Index> LeChuck(N_states,N_chunks);
	for (LeChuck=LeChuck.begin(); LeChuck<LeChuck.end(); ++LeChuck)
	{
		make_Tmatrix(LeChuck.value1(),LeChuck.value2());
	}

	HMATRIX_CHECK = true;
	storedHmatrix.makeCompressed();
}

// construct with L,N,hopping
HoppingParticles::
HoppingParticles (int L_input, int N_input, double hopping_value, BOUNDARY_CONDITION BC_input, BOND_STORAGE BS_input, DIM dim_input, SPIN_STATISTICS SS_input)
:OccNumVecSpaceFloor(L_input, N_input, BC_input, MM_FULL, dim_input, SS_input)
{
	HOPPING_FORMAT = BS_input;
	// set the hopping
	if (hopping_value != 0.) {hoppings.resize(1); hoppings(0) = hopping_value;}
	Geometry Euklid;
	BondMatrix = Euklid.BondMatrix(L_edge, spacedim, BOUNDARIES, hoppings, BS_input);
	// no on-site energies
	ONSITE_CHECK = false;
	construct();
}

// construct with params
HoppingParticles::
HoppingParticles (int L_input, int N_input, InteractionParams params_input, BOUNDARY_CONDITION BC_input, BOND_STORAGE BS_input, DIM dim_input, SPIN_STATISTICS SS_input)
:OccNumVecSpaceFloor(L_input, N_input, BC_input, MM_FULL, dim_input, SS_input)
{
	HOPPING_FORMAT = BS_input;
	// extract hoppings
	Geometry Euklid;
	BondMatrix = Euklid.BondMatrix(L_edge, spacedim, BOUNDARIES, params_input.hoppings, BS_input);
	// extract on-site energies
	ONSITE_CHECK = params_input.ONSITE_CHECK;
	onsiteEnergies = params_input.onsiteEnergies;
	hoppings = params_input.hoppings;
	construct();
}

// construct with L,N,HoppingMatrix
HoppingParticles::
HoppingParticles (int L_input, int N_input, const SparseMatrixXd &BondMatrix_input, BOUNDARY_CONDITION BC_input, BOND_STORAGE BS_input, DIM dim_input, SPIN_STATISTICS SS_input)
:OccNumVecSpaceFloor(L_input, N_input, BC_input, MM_FULL, dim_input, SS_input)
{
	HOPPING_FORMAT = BS_input;
	if (HOPPING_FORMAT == BS_UPPER)
	{
		BondMatrix = BondMatrix_input.triangularView<Upper>();
	}
//	else if (HOPPING_FORMAT == LOWER)
//	{
//		BondMatrix = BondMatrix_input.triangularView<Lower>();
//	}
	else
	{
		BondMatrix = BondMatrix_input;
	}
	VectorXd BondDiagonal = BondMatrix.diagonal(); // necessary, otherwise compiler warning: address of temporary
	ONSITE_CHECK = (BondDiagonal.minCoeff()==0. and BondDiagonal.maxCoeff()==0.)? false : true;
	construct();
}
//--------------</constructors>--------------

//--------------<construct stuff>--------------
void HoppingParticles::
create_basis()
{
	basis.reserve(N_states);
	OccNumVector init(N_sites);
	colex(N_sites,N_particles,init,basis);
}

SparseMatrixXd::Index HoppingParticles::
get_stateNr (const OccNumBasis &basis, const OccNumVector &V, SparseMatrixXd::Index guess)
{
	if (guess<basis.size() and V==basis[guess]) {return guess;} // test if the new state is just next in list and return if true
	else
	{
		SparseMatrixXd::Index index_min = 0;
		SparseMatrixXd::Index index_max = basis.size()-1;
		SparseMatrixXd::Index index_current;
		// otherwise run bisectional search
		while(1)
		{
			index_current = index_min + (index_max-index_min)/2;
			if      (V<basis[index_current]) {index_max=index_current-1;}
			else if (V>basis[index_current]) {index_min=index_current+1;}
			else {break;}
		}
		return index_current;
	}
}

short int HoppingParticles::
parity (const OccNumVector &V, int site1, int site2, SPIN_STATISTICS SS_input)
{
	short int out=1;
	if (SS_input==FERMIONS and site1!=site2)
	{
		for (int i=min(site1,site2)+1; i<max(site1,site2); ++i) // check all sites between site1 & site2
		{
			if (V[i]==1) {out*=-1;} // switch sign for every electron found between site1 & site2
		}
	}
	return out;
}

void HoppingParticles::
make_Tmatrix (SparseMatrixXd::Index state_min, SparseMatrixXd::Index state_max)
{
	vector<triplet> tripletList;
	
	size_t tripletListSize=0;
	for (SparseMatrixXd::Index istate=state_min; istate<state_max; ++istate) // for all states between state_min & state_max
	{
		for (size_t isite=basis[istate].find_first(); isite<N_sites; isite=basis[istate].find_next(isite)) // for all particles
		for (SparseMatrixXd::InnerIterator it(BondMatrix,isite); it; ++it) // iterate over whole bond matrix
		{
			int jsite = it.row(); // BondMatrix: col=isite, row=it.row(), value=it.value()
			if (basis[istate][jsite]==0) {++tripletListSize;}
			else if (isite==jsite)       {++tripletListSize;}
		}
	}
	tripletList.reserve(tripletListSize);

	for (SparseMatrixXd::Index istate=state_min; istate<state_max; ++istate) // for all states between state_min & state_max
	{
		for (size_t isite=basis[istate].find_first(); isite<N_sites; isite=basis[istate].find_next(isite)) // for all particles
		for (SparseMatrixXd::InnerIterator it(BondMatrix,isite); it; ++it) // iterate over whole bond matrix
		{
			OccNumVector state_vector_new = basis[istate]; // take the state
			int jsite = it.row(); // BondMatrix: col=isite, row=it.row(), value=it.value()
			if (state_vector_new[jsite]==0) // if site to be hopped on is empty
			{
				state_vector_new[isite].flip(); // perform hopping from isite
				state_vector_new[jsite].flip(); // to jsite
				SparseMatrixXd::Index jstate = get_stateNr(basis, state_vector_new, istate+1); // get index of new state from basis
				double val = it.value() * parity(basis[istate], isite, jsite, SPINSTAT);
				tripletList.push_back(triplet(jstate,istate,val));
			}
			else if (isite==jsite)
			{
				tripletList.push_back(triplet(istate,istate,it.value()));
			}
		}
	}

	if (state_min==0 and state_max==N_states)
	{
		storedHmatrix.setFromTriplets(tripletList.begin(), tripletList.end());
	}
	else
	{
		SparseMatrixXd Mout(N_states,N_states);
		Mout.setFromTriplets(tripletList.begin(), tripletList.end());
		storedHmatrix += Mout;
	}
}

void HoppingParticles::
switch_Vnn (double Vnn_input)
{
	Vnn = Vnn_input;
	Geometry Euklid;
	SparseMatrixXd BondMatrix = Euklid.BondMatrix(L_edge, spacedim, BOUNDARIES, 1., BS_FULL);
	
	SparseMatrixXd Mout(N_states,N_states);
	vector<triplet> tripletList(N_states);
	
	#pragma omp parallel for
	for (size_t stateNr=0; stateNr<N_states; ++stateNr)
	{
		double res = 0.;
		for (int isite=0; isite<N_sites; ++isite)
		for (SparseMatrixXd::InnerIterator it(BondMatrix,isite); it and it.row()<=it.col(); ++it)
		{
			int jsite = it.row();
			res += basis[stateNr][isite] * basis[stateNr][jsite] * Vnn;
		}
		tripletList[stateNr] = triplet(stateNr,stateNr,res);
	}
	
	Mout.setFromTriplets(tripletList.begin(), tripletList.end());
	storedHmatrix += Mout;
}

void HoppingParticles::
switch_Vdiag (double Vdiag_input)
{
	Vdiag = Vdiag_input;
	vector<triplet> tripletList(N_states);
	
	#pragma omp parallel for
	for (size_t stateNr=0; stateNr<N_states; ++stateNr)
	{
		tripletList[stateNr] = triplet(stateNr,stateNr, basis[stateNr].count()*Vdiag);
	}
	
	SparseMatrixXd Mout(N_states,N_states);
	Mout.setFromTriplets(tripletList.begin(), tripletList.end());
	storedHmatrix += Mout;
}
//--------------</construct stuff>--------------


//--------------<info>--------------
string HoppingParticles::
info() const
{
	stringstream ss;
	ss << infolabel << ": "
	<< hopping_info() << ", "
	<< states_sites_info() << ", "
	<< "N=" << N_particles << ", "
	<< mem_info();
	return ss.str();
}

string HoppingParticles::
hopping_info() const
{
	stringstream ss;
	if (hoppings.rows()==1)
	{
		ss << "hopping=" << hoppings(0);
	}
	else if (hoppings.rows()==0)
	{
		if (BondMatrix.nonZeros()==0)
		{
			ss << "hopping=0";
		}
		else
		{
			ss << "hoppings=" << MatrixXd(BondMatrix).minCoeff() << "..." << MatrixXd(BondMatrix).maxCoeff();
		}
	}
	else
	{
		ss << "hoppings=("<<hoppings.transpose()<<")";
	}
	ss << ", ";
	
	if (onsiteEnergies.rows()>0)
	{
		if (onsiteEnergies.minCoeff() == onsiteEnergies.maxCoeff())
		{
			ss << "on-site=const("<<onsiteEnergies(0)<<")";		
		}
		else
		{
			ss << "on-site=("<<onsiteEnergies.transpose()<<")";
		}
	}
	else
	{
		ss << "on-site=0";
	}
	
	return ss.str();
}

double HoppingParticles::
memory (MEMUNIT memunit) const
{
	double res = 0;
	if (storedHmatrix.rows()>0)
	{
		res += calc_memory(storedHmatrix,memunit);
	}
	res += 5.*calc_memory<double>(basis.size(),memunit);
	
	return res;
};
//--------------</info>--------------


//void HoppingParticles::
//split_hoppings() // for next neighbours only!
//{
//	Geometry Euklid;
//	SparseMatrixXd BondMatrix_forw = Euklid.BondMatrix(L_edge,spacedim,BOUNDARIES,-1.,FORWARDS);
//	SparseMatrixXd BondMatrix_back = Euklid.BondMatrix(L_edge,spacedim,BOUNDARIES,-1.,BACKWARDS);
//	
//	Chunkomatic LeChuck(N_states,max((int)ceil(N_states/1e6),1));
//	
//	storedHmatrix.resize(N_states,N_states);    storedHmatrix.setZero();
//	storedHmatrixAlt.resize(N_states,N_states); storedHmatrixAlt.setZero();
//	
//	for (LeChuck=LeChuck.begin(); LeChuck<LeChuck.end(); ++LeChuck)
//	{
//		storedHmatrix    += hopping_matrix(LeChuck.value1(),LeChuck.value2(), BondMatrix_forw, false);
//	}
//	
//	for (LeChuck=LeChuck.begin(); LeChuck<LeChuck.end(); ++LeChuck)
//	{
//		storedHmatrixAlt += hopping_matrix(LeChuck.value1(),LeChuck.value2(), BondMatrix_back, false);
//	}
//}

//--------------<operators>--------------
void HoppingParticles::
hopping_element (int isite, int jsite, double hopping_value, bool MAKE_HERMITIAN, SparseMatrixXd &M)
{
	assert(isite>=0 and isite<N_sites);
	assert(jsite>=0 and jsite<N_sites);
	
	M.resize(N_states,N_states);
	vector<triplet> tripletList;
	
	for (SparseMatrixXd::Index istate=0; istate<N_states; ++istate)
	{
		OccNumVector state_vector_new = basis[istate];
//		cout << state_vector_new << endl;
		if (state_vector_new[isite]==1 and state_vector_new[jsite]==0)
		{
			state_vector_new[isite].flip();
			state_vector_new[jsite].flip();
//			cout << "-> " << state_vector_new << endl;
			SparseMatrixXd::Index jstate = get_stateNr(basis, state_vector_new, istate);
			double val = hopping_value * parity(basis[istate], isite, jsite, SPINSTAT);
			
			tripletList.push_back(triplet(jstate,istate,val));
			if (MAKE_HERMITIAN==true)
			{
				tripletList.push_back(triplet(istate,jstate,val));
			}
			//Mout.insert(jstate,istate) = hopping_value * parity(basis[istate], isite, jsite, SPINSTAT);
		}
	}
	M.setFromTriplets(tripletList.begin(), tripletList.end());
	M.makeCompressed();
}

void HoppingParticles::
n (int isite, SparseMatrixXd &Mout)
{
	Mout.resize(N_states,N_states);
	for (SparseMatrixXd::Index state=0; state<N_states; ++state)
	{
		Mout.insert(state,state) = static_cast<double>(basis[state][isite]);
	}
	Mout.makeCompressed();
//	vector<triplet> tripletList(N_states);
//	#pragma omp parallel for
//	for (size_t state=0; state<N_states; ++state)
//	{
//		tripletList[state] = triplet(state, state, static_cast<double>(basis[state][isite]));
//	}
//	M.setFromTriplets(tripletList.begin(), tripletList.end());
//	M.makeCompressed();
}

//void HoppingParticles::
//Nop (SparseMatrixXd &Mout)
//{
//	Mout.resize(N_states,N_states);
//	for (size_t state=0; state<N_states; ++state)
//	{
//		Mout.insert(state, state, static_cast<double>(basis[state].count());
//	}
//	Mout.makeCompressed();
////	vector<triplet> tripletList(N_states);
////	#pragma omp parallel for
////	for (size_t state=0; state<N_states; ++state)
////	{
////		tripletList[state] = triplet(state, state, static_cast<double>(basis[state].count()));
////	}
////	M.setFromTriplets(tripletList.begin(), tripletList.end());
////	M.makeCompressed();
//}

inline SparseMatrixXd HoppingParticles::
hopping_element (int isite, int jsite, double hopping_value, bool MAKE_HERMITIAN)
{
	SparseMatrixXd Mtemp;
	hopping_element(isite,jsite,hopping_value,MAKE_HERMITIAN,Mtemp);
	return Mtemp;
}

inline SparseMatrixXd HoppingParticles::
n (int isite)
{
	SparseMatrixXd Mtemp;
	n(isite,Mtemp);
	return Mtemp;
}

//inline SparseMatrixXd HoppingParticles::
//Nop()
//{
//	SparseMatrixXd Mtemp;
//	Nop(Mtemp);
//	return Mtemp;
//}
//--------------</operators>--------------

#endif

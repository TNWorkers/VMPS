#ifndef OCCNUMVECSPACEFLOOR
#define OCCNUMVECSPACEFLOOR

#include <stdio.h>
#include <sstream>
#include <memory>
using namespace std;

#include <Eigen/Dense>
#include <Eigen/SparseCore>
using namespace Eigen;

#include <gsl/gsl_math.h>

#include "HilbertTypedefs.h"
#include "Geometry.h"
#include "MemCalc.h"
#include "StringStuff.h"
#include "HxV.h"
#include "Stopwatch.h"

template <typename T> void NullDeleter(T*){} // null deleter for shared pointers

class OccNumVecSpaceFloor
{
//template<typename Scalar> friend void HxV (const OccNumVecSpaceFloor &H, const Matrix<Scalar,Dynamic,1> &Vin, Matrix<Scalar,Dynamic,1> &Vout);
//template<typename Scalar> friend void HxV (const OccNumVecSpaceFloor &H, Matrix<Scalar,Dynamic,1> &Vinout);

public:

	OccNumVecSpaceFloor() {};
	OccNumVecSpaceFloor (int L_edge_input, int N_input,
	                     BOUNDARY_CONDITION BC_input = BC_PERIODIC,
	                     MEM_MANAGEMENT MM_input = MM_FULL,
	                     DIM spacedim_input = DIM1,
	                     SPIN_STATISTICS SS_input = FERMIONS);
	
	//--------<info>--------
	virtual string info() const;
	string mem_info (MEMUNIT memunit=GB) const;
	string states_sites_info() const;
	virtual double memory (MEMUNIT memunit=GB) const;
	virtual double alpha() const; // α = #non-zeros per row/col
	virtual void print_basis() const;
	string infolabel;
	//--------</info>--------
	
	//--------<access>--------
	size_t dim()            const {return N_states;};
	DIM D()                 const {return spacedim;};
	int L()                 const {return L_edge;};
	int volume()            const {return N_sites;};
	int N()                 const {return N_particles;};
	BOUNDARY_CONDITION BC() const {return BOUNDARIES;};
	SPIN_STATISTICS SS()    const {return SPINSTAT;};
	size_t rows()           const {return N_states;};
	size_t cols()           const {return N_states;};
	
//	virtual double trace() const;
	virtual double norm() const;
	virtual void scale (double factor=1., double offset=0.);
	
	const SparseMatrixXd &Hmatrix() const {return storedHmatrix;};
	bool check_HMATRIX() const {return HMATRIX_CHECK;}
	
	OccNumVector basis_state (size_t state_nr) {return basis[state_nr];};
	std::shared_ptr<OccNumBasis> basis_ptr() const {return pointer_to_basis;}
	//--------</access>--------
	
	//--------<diagonalization>--------
	void diagonalize() const;
	VectorXd eigenvalues() const;
	MatrixXd eigenvectors() const;
	int ground_state_degeneracy (double eps=1e-8) const;
	//--------</diagonalization>--------
	
	void kill_basis();
	void kill_Hmatrix();
	
protected:
	
	SparseMatrixXd storedHmatrix;
	bool HMATRIX_CHECK;
	MEM_MANAGEMENT HMATRIX_FORMAT;
	
	size_t N_states;
	int N_sites;
	int L_edge;
	DIM spacedim;
	int N_particles;
	BOUNDARY_CONDITION BOUNDARIES;
	SPIN_STATISTICS SPINSTAT;

	vector<OccNumVector> basis;
	
	std::shared_ptr<OccNumBasis> pointer_to_basis;
	
	mutable SelfAdjointEigenSolver<MatrixXd> Eigensolver;
	mutable bool DIAGONALIZED_CHECK;
};

template<typename Scalar>
inline void HxV (const OccNumVecSpaceFloor &H, const Matrix<Scalar,Dynamic,1> &Vin, Matrix<Scalar,Dynamic,1> &Vout)
{
//	Stopwatch<> Chronos;
	Vout.noalias() = H.Hmatrix().selfadjointView<Upper>() * Vin;
//	lout << Chronos.info("HxV(OccNumVecSpaceFloor,fullMatrix)") << endl;
}

template<typename Scalar>
inline void HxV (const OccNumVecSpaceFloor &H, Matrix<Scalar,Dynamic,1> &Vinout)
{
	Vinout = H.Hmatrix().selfadjointView<Upper>() * Vinout;
}

template<typename Scalar>
void polyIter (const OccNumVecSpaceFloor &H, const Matrix<Scalar,Dynamic,1> &Vin1, double polyB, const Matrix<Scalar,Dynamic,1> &Vin2, Matrix<Scalar,Dynamic,1> &Vout)
{
	Stopwatch<> Chronos;
	HxV(H,Vin1,Vout);
	Vout -= polyB * Vin2;
	lout << Chronos.info("polyIter(OccNumVecSpaceFloor)") << endl;
}

//--------------<constructor>--------------
OccNumVecSpaceFloor::
OccNumVecSpaceFloor(int L_edge_input, int N_input, BOUNDARY_CONDITION BC_input, MEM_MANAGEMENT MM_input, DIM spacedim_input, SPIN_STATISTICS SS_input)
:L_edge(L_edge_input), N_particles(N_input), BOUNDARIES(BC_input), HMATRIX_FORMAT(MM_input), spacedim(spacedim_input), SPINSTAT(SS_input), HMATRIX_CHECK(false)
{
	N_states = 0;
	infolabel = "OccNumVecSpace";
	N_sites = pow(L_edge_input,static_cast<unsigned int>(spacedim));
	DIAGONALIZED_CHECK = false;
	pointer_to_basis   = std::shared_ptr<OccNumBasis>   (&basis,         NullDeleter<OccNumBasis>);
}

void OccNumVecSpaceFloor::
kill_basis()
{
	basis.clear();
	vector<OccNumVector>().swap(basis);
}

void OccNumVecSpaceFloor::
kill_Hmatrix()
{
	storedHmatrix.resize(0,0);
}
//--------------</constructor>--------------

//--------------<info>--------------
string OccNumVecSpaceFloor::
info() const
{
	stringstream ss;
	ss << infolabel << ": " << states_sites_info() << ", ";
	ss << "N=" << N_particles << ", "
	<< mem_info();
	return ss.str();
}

string OccNumVecSpaceFloor::
states_sites_info() const
{
	stringstream ss;
	ss << "states=" << N_states << ", " << "sites=" << N_sites << " (";
	if (static_cast<int>(spacedim)>1)
	{
		ss << "=" << L_edge << "^" << spacedim << ", ";
	}
	if      (BOUNDARIES == BC_PERIODIC) {ss << "periodic)";}
	else if (BOUNDARIES == BC_DANGLING) {ss << "dangling)";}
	return ss.str();
}

string OccNumVecSpaceFloor::
mem_info (MEMUNIT memunit) const
{
	stringstream ss;
	ss << "mem(H)=" << round(memory(memunit),3) << "GB "
	   << "(α=" << round(alpha(),3) << "), "
	   << "mem(Ψ∊ℝ)=" << round(calc_memory<double>(N_states,memunit),3) << "GB";
	return ss.str();
}

inline double OccNumVecSpaceFloor::
alpha() const
{
	return static_cast<double>(storedHmatrix.nonZeros())/N_states;
}

void OccNumVecSpaceFloor::
print_basis() const
{
	for (size_t i=0; i<N_states; ++i) {cout << i << "\t" << basis[i] << endl;}
}

inline double OccNumVecSpaceFloor::
memory (MEMUNIT memunit) const
{
	return calc_memory(storedHmatrix, memunit);
}
//--------------</info>--------------

//--------------<calculate stuff>--------------
//double OccNumVecSpaceFloor::
//trace() const
//{
//	double out = 0.;
//	for (size_t k=0; k<storedHmatrix.outerSize(); ++k)
//	{
//		out += storedHmatrix.coeff(k,k);
//	}
//	return out;
//}

double OccNumVecSpaceFloor::
norm() const
{
	double res = 2.*pow(storedHmatrix.triangularView<StrictlyUpper>().norm(),2);
	res += pow(storedHmatrix.diagonal().norm(),2);
	return sqrt(res);
}

void OccNumVecSpaceFloor::
scale (double factor, double offset)
{
	if (factor != 1.)
	{
		storedHmatrix *= factor;
	}
	if (offset != 0.)
	{
		SparseMatrixXd Hoffset(N_states,N_states);
		for (int i=0; i<N_states; ++i)
		{
			Hoffset.insert(i,i) = offset;
		}
		storedHmatrix += Hoffset;
	}
}

void OccNumVecSpaceFloor::
diagonalize() const
{
	assert(HMATRIX_CHECK == true and "No Hmatrix constructed!");
	
	if (DIAGONALIZED_CHECK == false)
	{
		if (N_states>5000)
		{
			cout << "WARNING: diagonalizing a " << N_states << "x" << N_states << " matrix; estimated time: " << 8.3e-16*pow(N_states,4.) << " hours" << endl;
		}
		
		MatrixXd Hdense(storedHmatrix);
		Hdense += Hdense.triangularView<StrictlyUpper>().transpose();
		Eigensolver.compute(Hdense);
		DIAGONALIZED_CHECK = true;
	}
}

inline VectorXd OccNumVecSpaceFloor::
eigenvalues() const
{
	if (DIAGONALIZED_CHECK == false) {diagonalize();}
	return Eigensolver.eigenvalues();
}

inline MatrixXd OccNumVecSpaceFloor::
eigenvectors() const
{
	if (DIAGONALIZED_CHECK == false) {diagonalize();}
	return Eigensolver.eigenvectors();
}

int OccNumVecSpaceFloor::
ground_state_degeneracy (double eps) const
{
	if (DIAGONALIZED_CHECK == false) {diagonalize();}
	
	double E0 = Eigensolver.eigenvalues().minCoeff();
	int out=0;
	for (size_t i=0; i<N_states; ++i)
	{
		if (gsl_fcmp(Eigensolver.eigenvalues()(i),E0, eps) == 0) {++out;}
	}
	return out;
}
//--------------<calculate stuff>--------------

//--------------<external wrappers>--------------
//inline double trace (const OccNumVecSpaceFloor &H)
//{
//	return H.trace();
//}

inline double norm (const OccNumVecSpaceFloor &H)
{
	return H.norm();
}

inline size_t dim (const OccNumVecSpaceFloor &H)
{
	return H.dim();
}
//--------------</external wrappers>--------------

#endif

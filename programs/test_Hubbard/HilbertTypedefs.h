#ifndef HILBERTTYPEDEFS
#define HILBERTTYPEDEFS

#ifndef LARGE_HUBBARD_SPACE
#define LARGE_HUBBARD_SPACE 12e6 // Hub(14):11.8e6, Hub(16):165.6e6
#endif

#include <Eigen/Dense>
#include <Eigen/SparseCore>
using namespace Eigen;

#include <boost/dynamic_bitset.hpp>
//#include <boost/shared_ptr.hpp>

#include <vector>
#include <iostream>
//#include <iterator>
//using namespace std;

enum DIM {DIM0=0, DIM1=1, DIM2=2, DIM3=3};
#ifndef SPIN_INDEX_ENUM
#define SPIN_INDEX_ENUM
	enum SPIN_INDEX {UP=false, DN=true, NOSPIN=2, UPDN=3};
	SPIN_INDEX operator! (const SPIN_INDEX sigma)
	{
		assert(sigma==UP or sigma==DN);
		return (sigma==UP) ? DN : UP;
	}
	//string spin_index_strings[] = {"UP","DN","NO","UPDN"};
#endif
enum PARTICLE_TYPE      {HOLES=false, PARTICLES=true};
enum DECONSTRUCTION     {CREATE=false, ANNIHILATE=true};
enum SPIN_STATISTICS    {FERMIONS, BOSONS};
enum BOUNDARY_CONDITION {BC_PERIODIC, BC_DANGLING};

std::ostream& operator<< (std::ostream& s, BOUNDARY_CONDITION BC)
{
	if      (BC==BC_PERIODIC) {s << "PBC";}
	else if (BC==BC_DANGLING) {s << "OBC";}
	return s;
}

enum BOND_STORAGE {BS_FORWARDS, BS_UPPER, BS_FULL, BS_NO};
enum MEM_MANAGEMENT {MM_FULL, MM_SUB, MM_DYN, MM_IDLE};

std::ostream& operator<< (std::ostream& s, MEM_MANAGEMENT MM)
{
	if      (MM==MM_FULL) {s << "MM_FULL";} // full Hmatrix
	else if (MM==MM_SUB)  {s << "MM_SUB";} // only subspaces
	else if (MM==MM_DYN)  {s << "MM_DYN";} // decide dynamically
	else if (MM==MM_IDLE) {s << "MM_IDLE";} // do nothing (useful for inheritance)
	return s;
}

#ifndef EIGEN_DEFAULT_SPARSE_INDEX_TYPE
#define EIGEN_DEFAULT_SPARSE_INDEX_TYPE int
#endif
typedef Eigen::SparseMatrix<double,ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE> SparseMatrixXd;
typedef Eigen::SparseVector<double,ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE> SparseVectorXd;
typedef Eigen::SparseMatrix<complex<double>,ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE> SparseMatrixXcd;
typedef Eigen::SparseVector<complex<double>,ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE> SparseVectorXcd;

typedef Eigen::Triplet<double> triplet;

typedef boost::dynamic_bitset<unsigned char> OccNumVector;
typedef std::vector<OccNumVector> OccNumBasis;

#endif

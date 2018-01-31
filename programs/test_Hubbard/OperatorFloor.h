#ifndef OPERATORFLOOR
#define OPERATORFLOOR

#include <Eigen/Dense>
#include <Eigen/SparseCore>
using namespace Eigen;

#include "HilbertTypedefs.h"
#include "MemCalc.h"

class OperatorFloor
{
//template<typename,typename> friend class PolynomialBath;
//template<typename Scalar> friend void OxV (const OperatorFloor &H, const Matrix<Scalar,Dynamic,1> &Vin, Matrix<Scalar,Dynamic,1> &Vout);
//template<typename Scalar> friend void OxV (const OperatorFloor &H, Matrix<Scalar,Dynamic,1> &Vinout);

public:

	OperatorFloor(){};
	OperatorFloor (const SparseMatrixXd &M_input);

	inline const SparseMatrixXd &Operator() const {return storedOperator;};
	
	double memory (MEMUNIT memunit=GB) const;

	size_t rows() const {return storedOperator.rows();};
	size_t cols() const {return storedOperator.cols();};

protected:

	SparseMatrixXd storedOperator;

};

OperatorFloor::
OperatorFloor (const SparseMatrixXd &M_input)
:storedOperator(M_input)
{}

inline double OperatorFloor::
memory (MEMUNIT memunit) const
{
	return calc_memory(storedOperator, memunit);
}

template<typename Scalar>
inline void OxV (const OperatorFloor &O, const Matrix<Scalar,Dynamic,1> &Vin, Matrix<Scalar,Dynamic,1> &Vout)
{
	Vout.noalias() = O.Operator() * Vin;
}

template<typename Scalar>
inline void OxV (const OperatorFloor &O, Matrix<Scalar,Dynamic,1> &Vinout)
{
	Vinout = O.Operator() * Vinout;
}

#endif

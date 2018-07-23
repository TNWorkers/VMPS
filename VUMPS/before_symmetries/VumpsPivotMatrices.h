#ifndef VUMPSPIVOTSTUFF
#define VUMPSPIVOTSTUFF

#include "tensors/Biped.h"
#include "pivot/DmrgPivotVector.h"

//-----------<definitions>-----------

/**Structure to update \f$A_C\f$ (eq. 11) with a 2-site Hamiltonian. Contains \f$A_L\f$, \f$A_L\f$ and \f$H_L\f$ (= \p L), \f$H_R\f$ (= \p R).
\ingroup VUMPS
*/
template<typename Symmetry, typename Scalar, typename MpoScalar=double>
struct PivumpsMatrix1
{
	PivumpsMatrix1(){};
	
	// Produces an error with boost::multi_array!
//	PivumpsMatrix (const Matrix<Scalar,Dynamic,Dynamic> &L_input,
//	               const Matrix<Scalar,Dynamic,Dynamic> &R_input,
//	               const boost::multi_array<MpoScalar,4> &h_input,
//	               const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &AL_input,
//	               const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &AR_input)
//	:L(L_input), R(R_input), AL(AL_input), AR(AR_input)
//	{
//		size_t D = h_input.shape()[0];
//		h.resize(boost::extents[D][D][D][D]);
//		h = h_input;
//	}
	
	Matrix<Scalar,Dynamic,Dynamic> L;
	Matrix<Scalar,Dynamic,Dynamic> R;
	
	boost::multi_array<MpoScalar,4> h;
	
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > AL;
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > AR;
};

/**Structure to update \f$C\f$ (eq. 16) with a 2-site Hamiltonian. Contains \f$A_L\f$, \f$A_L\f$ and \f$H_L\f$ (= \p L), \f$H_R\f$ (= \p R).
\ingroup VUMPS
*/
template<typename Symmetry, typename Scalar, typename MpoScalar=double>
struct PivumpsMatrix0
{
	PivumpsMatrix0(){};
	
	PivumpsMatrix0 (const PivumpsMatrix1<Symmetry,Scalar,MpoScalar> &H)
	:L(H.L), R(H.R), h(H.h), AL(H.AL), AR(H.AR)
	{}
	
	Matrix<Scalar,Dynamic,Dynamic> L;
	Matrix<Scalar,Dynamic,Dynamic> R;
	
	boost::multi_array<MpoScalar,4> h;
	
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > AL;
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > AR;
};
//-----------</definitions>-----------

/**Performs the local update of \f$A_C\f$ (eq. 11) with a 2-site Hamiltonian.*/
template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivumpsMatrix1<Symmetry,Scalar,MpoScalar> &H, const PivotVector<Symmetry,Scalar> &Vin, PivotVector<Symmetry,Scalar> &Vout)
{
	size_t D = H.h.shape()[0];
	
	Vout = Vin;
	for (size_t s=0; s<D; ++s)
	{
		Vout.data[s].block[0].setZero();
	}
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	{
		if (H.h[s1][s2][s3][s4] != 0.)
		{
			Vout.data[s3].block[0] += H.h[s1][s2][s3][s4] * H.AL[s1].block[0].adjoint() * H.AL[s2].block[0] * Vin.data[s4].block[0];
		}
	}
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	{
		if (H.h[s1][s2][s3][s4] != 0.)
		{
			Vout.data[s1].block[0] += H.h[s1][s2][s3][s4] * Vin.data[s2].block[0] * H.AR[s4].block[0] * H.AR[s3].block[0].adjoint();
		}
	}
	
	for (size_t s=0; s<D; ++s)
	{
		Vout.data[s].block[0] += H.L * Vin.data[s].block[0];
		Vout.data[s].block[0] += Vin.data[s].block[0] * H.R;
	}
}

/**Performs \p HxV in place.*/
template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivumpsMatrix1<Symmetry,Scalar,MpoScalar> &H, PivotVector<Symmetry,Scalar> &Vinout)
{
	PivotVector<Symmetry,Scalar> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

/**Performs the local update of \f$C\f$ (eq. 16) with an explicit 2-site Hamiltonian.*/
template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivumpsMatrix0<Symmetry,Scalar,MpoScalar> &H, const PivotVector<Symmetry,Scalar> &Vin, PivotVector<Symmetry,Scalar> &Vout)
{
	size_t D = H.h.shape()[0];
	
	Vout = Vin;
	Vout.data[0].setZero();
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	{
		if (H.h[s1][s2][s3][s4] != 0.)
		{
			Vout.data[0].block[0] += H.h[s1][s2][s3][s4] * H.AL[s1].block[0].adjoint() * H.AL[s2].block[0] 
			                                             * Vin.data[0].block[0] 
			                                             * H.AR[s4].block[0] * H.AR[s3].block[0].adjoint();
		}
	}
	
	Vout.data[0].block[0] += H.L * Vin.data[0].block[0];
	Vout.data[0].block[0] += Vin.data[0].block[0] * H.R;
}

/**Performs \p HxV in place.*/
template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivumpsMatrix0<Symmetry,Scalar,MpoScalar> &H, PivotVector<Symmetry,Scalar> &Vinout)
{
	PivotVector<Symmetry,Scalar> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
inline size_t dim (const PivumpsMatrix1<Symmetry,Scalar,MpoScalar> &H)
{
	return 0;
}

// How to calculate the Frobenius norm of this?
template<typename Symmetry, typename Scalar, typename MpoScalar>
inline double norm (const PivumpsMatrix1<Symmetry,Scalar,MpoScalar> &H)
{
	return 0;
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
inline size_t dim (const PivumpsMatrix0<Symmetry,Scalar,MpoScalar> &H)
{
	return 0;
}

// How to calculate the Frobenius norm of this?
template<typename Symmetry, typename Scalar, typename MpoScalar>
inline double norm (const PivumpsMatrix0<Symmetry,Scalar,MpoScalar> &H)
{
	return 0;
}

#endif

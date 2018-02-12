#ifndef VUMPSPIVOTSTUFF
#define VUMPSPIVOTSTUFF

#include "tensors/Biped.h"
#include "pivot/DmrgPivotStuff1.h"
#include "pivot/DmrgPivotStuff0.h"

//-----------<definitions>-----------

/**Structure to update \f$A_C\f$ (eq. 11) with 2-site Hamiltonian. Contains \f$A_L\f$, \f$A_L\f$ and \f$H_L\f$ (= \p L), \f$H_R\f$ (= \p R).
\ingroup VUMPS
*/
template<typename Symmetry, typename Scalar, typename MpoScalar=double>
struct PivumpsMatrix
{
	PivumpsMatrix(){};
	
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
//-----------</definitions>-----------

template<typename Symmetry, typename Scalar, typename MpoScalar>
inline size_t dim (const PivumpsMatrix<Symmetry,Scalar,MpoScalar> &H)
{
	return 0;
}

/**Performs the local update of \f$A_C\f$ (eq. 11) with a 2-site Hamiltonian.*/
template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivumpsMatrix<Symmetry,Scalar,MpoScalar> &H, const PivotVector1<Symmetry,Scalar> &Vin, PivotVector1<Symmetry,Scalar> &Vout)
{
	size_t D = H.h.shape()[0];
	
	Vout = Vin;
	for (size_t s=0; s<D; ++s)
	{
		Vout.A[s].block[0].setZero();
	}
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	{
		if (H.h[s1][s2][s3][s4] != 0.)
		{
			Vout.A[s3].block[0] += H.h[s1][s2][s3][s4] * H.AL[s1].block[0].adjoint() * H.AL[s2].block[0] * Vin.A[s4].block[0];
		}
	}
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	{
		if (H.h[s1][s2][s3][s4] != 0.)
		{
			Vout.A[s1].block[0] += H.h[s1][s2][s3][s4] * Vin.A[s2].block[0] * H.AR[s4].block[0] * H.AR[s3].block[0].adjoint();
		}
	}
	
	for (size_t s=0; s<D; ++s)
	{
		Vout.A[s].block[0] += H.L * Vin.A[s].block[0];
		Vout.A[s].block[0] += Vin.A[s].block[0] * H.R;
	}
}

/**Performs \p HxV in place.*/
template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivumpsMatrix<Symmetry,Scalar,MpoScalar> &H, PivotVector1<Symmetry,Scalar> &Vinout)
{
	PivotVector1<Symmetry,Scalar> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

/**Performs the local update of \f$C\f$ (eq. 16) with an explicit 2-site Hamiltonian.*/
template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivumpsMatrix<Symmetry,Scalar,MpoScalar> &H, const PivotVector0<Symmetry,Scalar> &Vin, PivotVector0<Symmetry,Scalar> &Vout)
{
	size_t D = H.h.shape()[0];
	
	Vout = Vin;
	Vout.C.setZero();
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	{
		if (H.h[s1][s2][s3][s4] != 0.)
		{
			Vout.C.block[0] += H.h[s1][s2][s3][s4] * H.AL[s1].block[0].adjoint() * H.AL[s2].block[0] 
			                                       * Vin.C.block[0] 
			                                       * H.AR[s4].block[0] * H.AR[s3].block[0].adjoint();
		}
	}
	
	Vout.C.block[0] += H.L * Vin.C.block[0];
	Vout.C.block[0] += Vin.C.block[0] * H.R;
}

/**Performs \p HxV in place.*/
template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivumpsMatrix<Symmetry,Scalar,MpoScalar> &H, PivotVector0<Symmetry,Scalar> &Vinout)
{
	PivotVector0<Symmetry,Scalar> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

#endif

#ifndef VANILLA_VUMPSTRANSFERMATRIX
#define VANILLA_VUMPSTRANSFERMATRIX

#include "boost/multi_array.hpp"

#include "VUMPS/VumpsTypedefs.h"
#include "tensors/Biped.h"

/**
Operators \f$T_L\f$, \f$T_R\f$ for solving the linear systems eq. 14; or \f$1-T_L+|R)(1|\f$, \f$1-T_R+|1)(R|\f$ for solving eq. C25ab. Due to the similar structure of the equations, no different data structures are required. 
\ingroup VUMPS
*/
template<typename Symmetry, typename Scalar>
struct TransferMatrix
{
	TransferMatrix(){};
	
	/**Constructor for a 1-site unit cell.*/
	TransferMatrix (GAUGE::OPTION gauge_input, 
	                const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Abra_input, 
	                const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Aket_input, 
	                const Matrix<Scalar,Dynamic,Dynamic> &LReigen_input, 
	                vector<Scalar> Wvec_input,
	                vector<size_t> D_input)
	:Abra(Abra_input), Aket(Aket_input), gauge(gauge_input), LReigen(LReigen_input), Wvec(Wvec_input), D(D_input)
	{
		assert(Aket.size() == Abra.size());
		
		if (Wvec.size() == 0)
		{
			Wvec.resize(Aket.size());
			for (size_t s=0; s<Aket.size(); ++s)
			{
				Wvec[s] = 1.;
			}
		}
	}
	
	/**Constructor for a 2-site unit cell.*/
	TransferMatrix (GAUGE::OPTION gauge_input, 
	                const vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > &ApairBra_input, 
	                const vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > &ApairKet_input, 
	                const Matrix<Scalar,Dynamic,Dynamic> &LReigen_input, 
	                boost::multi_array<double,4> Warray_input,
	                vector<size_t> D_input)
	:ApairBra(ApairBra_input), ApairKet(ApairKet_input), gauge(gauge_input), LReigen(LReigen_input), D(D_input)
	{
		assert(ApairKet.size() == ApairBra.size());
		assert(D_input.size() == 2);
		
		Warray.resize(boost::extents[D[0]][D[0]][D[1]][D[1]]);
		Warray = Warray_input;
	}
	
	/**Constructor for a 4-site unit cell.*/
	TransferMatrix (GAUGE::OPTION gauge_input, 
	                const boost::multi_array<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> >,4> &AquartettBra_input, 
	                const boost::multi_array<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> >,4> &AquartettKet_input, 
	                const Matrix<Scalar,Dynamic,Dynamic> &LReigen_input, 
	                boost::multi_array<double,8> Warray4_input,
	                vector<size_t> D_input)
	:AquartettBra(AquartettBra_input), AquartettKet(AquartettKet_input), gauge(gauge_input), LReigen(LReigen_input), D(D_input)
	{
		assert(ApairKet.size() == ApairBra.size());
		assert(D_input.size() == 4);
		Warray4.resize(boost::extents[D[0]][D[0]][D[1]][D[1]][D[2]][D[2]][D[3]][D[3]]);
		Warray4 = Warray4_input;
	}
	
	/**Gauge (L or R).*/
	GAUGE::OPTION gauge;
	
	/**Local dimensions within the unit cell.*/
	vector<size_t> D;
	
	///\{
	/** 1-cell data*/
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > Aket;
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > Abra;
	vector<Scalar> Wvec;
	///\}
	
	///\{
	/** 2-cell data*/
	vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > ApairKet;
	vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > ApairBra;
	boost::multi_array<double,4> Warray;
	///\}
	
	///\{
	/** 4-cell data*/
	boost::multi_array<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> >,4> AquartettKet;
	boost::multi_array<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> >,4> AquartettBra;
	boost::multi_array<double,8> Warray4;
	///\}
	
	/**Left and right eigenvectors \f$(L|\f$, \f$|R)\f$.*/
	Matrix<Scalar,Dynamic,Dynamic> LReigen;
};

/**
Vector \f$(H_L|\f$, \f$|H_R)\f$ that is obtained in the linear systems eq. 14 or \f$(L_a|\f$, \f$|R_a)\f$ that is obtained in eq. C25ab.
\ingroup VUMPS
*/
template<typename Scalar>
struct TransferVector
{
	Matrix<Scalar,Dynamic,Dynamic> A;
	GAUGE::OPTION gauge;
	
	///\{
	/**Linear algebra in the correspodnding vector space.*/
	TransferVector<Scalar>& operator+= (const TransferVector<Scalar> &Vrhs);
	TransferVector<Scalar>& operator-= (const TransferVector<Scalar> &Vrhs);
	template<typename OtherScalar> TransferVector<Scalar>& operator*= (const OtherScalar &alpha);
	template<typename OtherScalar> TransferVector<Scalar>& operator/= (const OtherScalar &alpha);
	///\}
};

template<typename Scalar>
inline void setZero (Matrix<Scalar,Dynamic,Dynamic> &M)
{
	M.setZero();
}

/**Matrix-vector multiplication in eq. 14 or 25ab
* Note:
* - if \p H.LReigen.rows()==0, only \p T is used (eq. 14)
* - if \p H.LReigen.rows()!=0, \p 1-T+|1)(LReigen| is used (eq. 25ab)
*/
template<typename Symmetry, typename Scalar1, typename Scalar2>
void HxV (const TransferMatrix<Symmetry,Scalar1> &H, const TransferVector<Scalar2> &Vin, TransferVector<Scalar2> &Vout)
{
	Vout = Vin;
	
	if (H.LReigen.rows() == 0)
	{
		setZero(Vout.A);
	}
	
	double factor = (H.LReigen.rows()==0)? +1.:-1.;
	
	// 1-cell
	if (H.Aket.size() != 0)
	{
		if (H.gauge == GAUGE::R)
		{
			for (size_t s=0; s<H.D[0]; ++s)
			{
				Vout.A += factor * H.Wvec[s] * H.Aket[s].block[0] * Vin.A * H.Abra[s].block[0].adjoint();
			}
		}
		else if (H.gauge == GAUGE::L)
		{
			for (size_t s=0; s<H.D[0]; ++s)
			{
				Vout.A += factor * H.Wvec[s] * H.Abra[s].block[0].adjoint() * Vin.A * H.Aket[s].block[0];
			}
		}
	}
	// 2-cell
	else if (H.ApairKet.size() != 0)
	{
		if (H.gauge == GAUGE::R)
		{
			for (size_t s1=0; s1<H.D[0]; ++s1)
			for (size_t s2=0; s2<H.D[0]; ++s2)
			for (size_t s3=0; s3<H.D[1]; ++s3)
			for (size_t s4=0; s4<H.D[1]; ++s4)
			{
				if (H.Warray[s1][s2][s3][s4] != 0.)
				{
					Vout.A += factor * H.Warray[s1][s2][s3][s4] * H.ApairKet[s2][s4].block[0] * Vin.A * H.ApairBra[s1][s3].block[0].adjoint();
				}
			}
		}
		else if (H.gauge == GAUGE::L)
		{
			for (size_t s1=0; s1<H.D[0]; ++s1)
			for (size_t s2=0; s2<H.D[0]; ++s2)
			for (size_t s3=0; s3<H.D[1]; ++s3)
			for (size_t s4=0; s4<H.D[1]; ++s4)
			{
				if (H.Warray[s1][s2][s3][s4] != 0.)
				{
					Vout.A += factor * H.Warray[s1][s2][s3][s4] * H.ApairBra[s1][s3].block[0].adjoint() * Vin.A * H.ApairKet[s2][s4].block[0];
				}
			}
		}
	}
	// 4-cell
	else if (H.AquartettKet.size() != 0)
	{
		if (H.gauge == GAUGE::R)
		{
			for (size_t s1=0; s1<H.D[0]; ++s1)
			for (size_t s2=0; s2<H.D[0]; ++s2)
			for (size_t s3=0; s3<H.D[1]; ++s3)
			for (size_t s4=0; s4<H.D[1]; ++s4)
			for (size_t s5=0; s5<H.D[2]; ++s5)
			for (size_t s6=0; s6<H.D[2]; ++s6)
			for (size_t s7=0; s7<H.D[3]; ++s7)
			for (size_t s8=0; s8<H.D[3]; ++s8)
			{
				if (H.Warray4[s1][s2][s3][s4][s5][s6][s7][s8] != 0.)
				{
					Vout.A += factor * H.Warray4[s1][s2][s3][s4][s5][s6][s7][s8] * 
					          H.AquartettKet[s2][s4][s6][s8].block[0] * Vin.A * H.AquartettBra[s1][s3][s5][s7].block[0].adjoint();
				}
			}
		}
		else if (H.gauge == GAUGE::L)
		{
			for (size_t s1=0; s1<H.D[0]; ++s1)
			for (size_t s2=0; s2<H.D[0]; ++s2)
			for (size_t s3=0; s3<H.D[1]; ++s3)
			for (size_t s4=0; s4<H.D[1]; ++s4)
			for (size_t s5=0; s5<H.D[2]; ++s5)
			for (size_t s6=0; s6<H.D[2]; ++s6)
			for (size_t s7=0; s7<H.D[3]; ++s7)
			for (size_t s8=0; s8<H.D[3]; ++s8)
			{
				if (H.Warray4[s1][s2][s3][s4][s5][s6][s7][s8] != 0.)
				{
					Vout.A += factor * 
					        H.Warray4[s1][s2][s3][s4][s5][s6][s7][s8] * 
					        H.AquartettBra[s1][s3][s5][s7].block[0].adjoint() * 
					        Vin.A * 
					        H.AquartettKet[s2][s4][s6][s8].block[0];
				}
			}
		}
	}
	
	if (H.LReigen.rows() != 0)
	{
		if (H.gauge == GAUGE::R)
		{
			Vout.A += (H.LReigen * Vin.A).trace() * Matrix<Scalar2,Dynamic,Dynamic>::Identity(Vin.A.rows(),Vin.A.cols());
		}
		else if (H.gauge == GAUGE::L)
		{
			Vout.A += (Vin.A * H.LReigen).trace() * Matrix<Scalar2,Dynamic,Dynamic>::Identity(Vin.A.rows(),Vin.A.cols());
		}
	}
}

template<typename Symmetry, typename Scalar1, typename Scalar2>
void HxV (const TransferMatrix<Symmetry,Scalar1> &H, TransferVector<Scalar2> &Vinout)
{
	TransferVector<Scalar2> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

template<typename Symmetry, typename Scalar>
inline size_t dim (const TransferMatrix<Symmetry,Scalar> &H)
{
	if (H.Aket.size() != 0)
	{
		return H.Aket[0].block[0].cols() * H.Abra[0].block[0].rows();
	}
	else if (H.ApairKet.size() != 0)
	{
		return H.ApairKet[0][0].block[0].cols() * H.ApairBra[0][0].block[0].rows();
	}
	else if (H.AquartettKet.size() != 0)
	{
		return H.AquartettKet[0][0][0][0].block[0].cols() * H.AquartettBra[0][0][0][0].block[0].rows();
	}
}

template<typename Scalar>
inline Scalar squaredNorm (const TransferVector<Scalar> &V)
{
//	return V.A.squaredNorm();
	return dot(V,V);
}

template<typename Scalar>
inline Scalar norm (const TransferVector<Scalar> &V)
{
	return sqrt(squaredNorm(V));
//	return V.A.norm();
}

template<typename Scalar>
inline void normalize (TransferVector<Scalar> &V)
{
	V /= norm(V);
}

template<typename Scalar>
inline Scalar infNorm (const TransferVector<Scalar> &V1, const TransferVector<Scalar> &V2)
{
	return (V1-V2).template lpNorm<Eigen::Infinity>();
}

template<typename Scalar>
void swap (TransferVector<Scalar> &V1, TransferVector<Scalar> &V2)
{
	V1.A.swap(V2.A);
}

template<typename Scalar>
inline Scalar dot (const TransferVector<Scalar> &V1, const TransferVector<Scalar> &V2)
{
	return (V1.A.adjoint()*V2.A).trace();
}

//-----------<vector arithmetics>-----------
template<typename Scalar>
TransferVector<Scalar>& TransferVector<Scalar>::
operator+= (const TransferVector<Scalar> &Vrhs)
{
	A += Vrhs.A;
	return *this;
}

template<typename Scalar>
TransferVector<Scalar>& TransferVector<Scalar>::
operator-= (const TransferVector<Scalar> &Vrhs)
{
	A -= Vrhs.A;
	return *this;
}

template<typename Scalar>
template<typename OtherScalar>
TransferVector<Scalar>& TransferVector<Scalar>::
operator*= (const OtherScalar &alpha)
{
	A *= alpha;
	return *this;
}

template<typename Scalar>
template<typename OtherScalar>
TransferVector<Scalar>& TransferVector<Scalar>::
operator/= (const OtherScalar &alpha)
{
	A /= alpha;
	return *this;
}

template<typename Scalar, typename OtherScalar>
TransferVector<Scalar> operator* (const OtherScalar &alpha, TransferVector<Scalar> V)
{
	return V *= alpha;
}

template<typename Scalar, typename OtherScalar>
TransferVector<Scalar> operator/ (TransferVector<Scalar> V, const OtherScalar &alpha)
{
	return V /= alpha;
}

template<typename Scalar, typename OtherScalar>
TransferVector<Scalar> operator+ (const TransferVector<Scalar> &V1, const TransferVector<Scalar> &V2)
{
	TransferVector<Scalar> Vout = V1;
	Vout.A += V2.A;
	return Vout;
}

template<typename Scalar, typename OtherScalar>
TransferVector<Scalar> operator- (const TransferVector<Scalar> &V1, const TransferVector<Scalar> &V2)
{
	TransferVector<Scalar> Vout = V1;
	Vout.A -= V2.A;
	return Vout;
}

template<typename Scalar>
inline void setZero (TransferVector<Scalar> &V)
{
	V.A.setZero();
}
//-----------</vector arithmetics>-----------

#include "RandomVector.h"

template<typename Scalar>
struct GaussianRandomVector<TransferVector<Scalar>,Scalar>
{
	static void fill (size_t N, TransferVector<Scalar> &Vout)
	{
		for (size_t i=0; i<Vout.A.rows(); ++i)
		for (size_t j=0; j<Vout.A.cols(); ++j)
		{
			Vout.A(i,j) = threadSafeRandUniform<Scalar>(-1.,1.);
		}
		normalize(Vout);
	}
};

#endif

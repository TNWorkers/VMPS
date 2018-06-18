#ifndef VANILLA_VUMPSTRANSFERMATRIX
#define VANILLA_VUMPSTRANSFERMATRIX

#include "boost/multi_array.hpp"

#include "VUMPS/VumpsTypedefs.h"
#include "tensors/Biped.h"
#include "tensors/Multipede.h"
#include "tensors/DmrgContractions.h"

/**
Operators \f$T_L\f$, \f$T_R\f$ for solving the linear systems eq. 14; or \f$1-T_L+|R)(1|\f$, \f$1-T_R+|1)(R|\f$ for solving eq. C25ab. Due to the similar structure of the equations, no different data structures are required. 
\ingroup VUMPS
*/
template<typename Symmetry, typename Scalar>
struct TransferMatrix
{
	TransferMatrix(){};
	
	TransferMatrix (GAUGE::OPTION gauge_input, 
	                const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Abra_input, 
	                const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Aket_input, 
	                const Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &LReigen_input, 
	                vector<vector<vector<SparseMatrix<Scalar> > > > W_input,
	                const vector<qarray<Symmetry::Nq> > &qloc_input,
	                const vector<qarray<Symmetry::Nq> > &qOp_input,
	                size_t ab_input)
	:gauge(gauge_input), Abra(Abra_input), Aket(Aket_input), LReigen(LReigen_input), W(W_input), qloc(qloc_input), qOp(qOp_input), ab(ab_input)
	{}
	
	/**Constructor for a 1-site unit cell.*/
//	TransferMatrix (GAUGE::OPTION gauge_input, 
//	                const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Abra_input, 
//	                const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Aket_input, 
//	                const Matrix<Scalar,Dynamic,Dynamic> &LReigen_input, 
//	                vector<Scalar> Wvec_input,
//	                vector<size_t> D_input)
//	:Abra(Abra_input), Aket(Aket_input), gauge(gauge_input), LReigen(LReigen_input), Wvec(Wvec_input), D(D_input)
//	{
//		assert(Aket.size() == Abra.size());
//		
//		if (Wvec.size() == 0)
//		{
//			Wvec.resize(Aket.size());
//			for (size_t s=0; s<Aket.size(); ++s)
//			{
//				Wvec[s] = 1.;
//			}
//		}
//	}
//	
//	/**Constructor for a 2-site unit cell.*/
//	TransferMatrix (GAUGE::OPTION gauge_input, 
//	                const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &ApairBra_input, 
//	                const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &ApairKet_input, 
//	                const Matrix<Scalar,Dynamic,Dynamic> &LReigen_input, 
//	                boost::multi_array<double,4> Warray_input,
//	                vector<size_t> D_input)
//	:ApairBra(ApairBra_input), ApairKet(ApairKet_input), gauge(gauge_input), LReigen(LReigen_input), D(D_input)
//	{
//		assert(ApairKet.size() == ApairBra.size());
//		assert(D_input.size() == 2);
//		
//		Warray.resize(boost::extents[D[0]][D[0]][D[1]][D[1]]);
//		Warray = Warray_input;
//	}
//	
//	/**Constructor for a 4-site unit cell.*/
//	TransferMatrix (GAUGE::OPTION gauge_input, 
//	                const boost::multi_array<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> >,4> &AquartettBra_input, 
//	                const boost::multi_array<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> >,4> &AquartettKet_input, 
//	                const Matrix<Scalar,Dynamic,Dynamic> &LReigen_input, 
//	                boost::multi_array<double,8> Warray4_input,
//	                vector<size_t> D_input)
//	:AquartettBra(AquartettBra_input), AquartettKet(AquartettKet_input), gauge(gauge_input), LReigen(LReigen_input), D(D_input)
//	{
//		assert(ApairKet.size() == ApairBra.size());
//		assert(D_input.size() == 4);
//		Warray4.resize(boost::extents[D[0]][D[0]][D[1]][D[1]][D[2]][D[2]][D[3]][D[3]]);
//		Warray4 = Warray4_input;
//	}
	
	/**Gauge (L or R).*/
	GAUGE::OPTION gauge;
	
	/**Local dimensions within the unit cell.*/
//	vector<size_t> D;
	
	///\{
	/** 1-cell data*/
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > Aket;
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > Abra;
	vector<vector<vector<SparseMatrix<Scalar> > > > W;
	///\}
	
//	///\{
//	/** 2-cell data*/
//	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > ApairKet;
//	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > ApairBra;
//	boost::multi_array<double,4> Warray;
//	///\}
//	
//	///\{
//	/** 4-cell data*/
//	boost::multi_array<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> >,4> AquartettKet;
//	boost::multi_array<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> >,4> AquartettBra;
//	boost::multi_array<double,8> Warray4;
//	///\}
	
	/**Left and right eigenvectors \f$(L|\f$, \f$|R)\f$.*/
//	Matrix<Scalar,Dynamic,Dynamic> LReigen;
	
	Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > LReigen;
	
	size_t ab;
	
	vector<qarray<Symmetry::Nq> > qloc;
	vector<qarray<Symmetry::Nq> > qOp;
};

/**
Vector \f$(H_L|\f$, \f$|H_R)\f$ that is obtained in the linear systems eq. 14 or \f$(L_a|\f$, \f$|R_a)\f$ that is obtained in eq. C25ab.
\ingroup VUMPS
*/
template<typename Symmetry, typename Scalar>
struct TransferVector
{
//	Matrix<Scalar,Dynamic,Dynamic> A;
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > data;
	size_t ab;
	
	///\{
	/**Linear algebra in the correspodnding vector space.*/
	TransferVector<Symmetry,Scalar>& operator+= (const TransferVector<Symmetry,Scalar> &Vrhs);
	TransferVector<Symmetry,Scalar>& operator-= (const TransferVector<Symmetry,Scalar> &Vrhs);
	template<typename OtherScalar> TransferVector<Symmetry,Scalar>& operator*= (const OtherScalar &alpha);
	template<typename OtherScalar> TransferVector<Symmetry,Scalar>& operator/= (const OtherScalar &alpha);
	///\}
};

//template<typename Symmetry, typename Scalar>
//inline void setZero (Matrix<Scalar,Dynamic,Dynamic> &M)
//{
//	M.setZero();
//}

/**Matrix-vector multiplication in eq. 14 or 25ab
* Note:
* - if \p H.LReigen.rows()==0, only \p T is used (eq. 14)
* - if \p H.LReigen.rows()!=0, \p 1-T+|1)(LReigen| is used (eq. 25ab)
*/
template<typename Symmetry, typename Scalar1, typename Scalar2>
void HxV (const TransferMatrix<Symmetry,Scalar1> &H, const TransferVector<Symmetry,Scalar2> &Vin, TransferVector<Symmetry,Scalar2> &Vout)
{
	Vout.ab = H.ab;
	TransferVector<Symmetry,Scalar2> TxV;
	TxV.ab  = H.ab;
	
	if (H.gauge == GAUGE::R)
	{
		contract_R(Vin.data, H.Abra, H.W, H.Aket, H.qloc, H.qOp, TxV.data, false, make_pair(CONTRACT_LR_MODE::FIXED,H.ab));
	}
	else if (H.gauge == GAUGE::L)
	{
		contract_L(Vin.data, H.Abra, H.W, H.Aket, H.qloc, H.qOp, TxV.data, false, make_pair(CONTRACT_LR_MODE::FIXED,H.ab));
	}
	
	Scalar2 LdotR;
	if (H.gauge == GAUGE::R)
	{
		LdotR = contract_LR(H.LReigen, Vin.data);
	}
	else if (H.gauge == GAUGE::L)
	{
		LdotR = contract_LR(Vin.data, H.LReigen);
	}
	
	for (size_t q=0; q<TxV.data.dim; ++q)
	{
		qarray3<Symmetry::Nq> quple = {TxV.data.in(q), TxV.data.out(q), TxV.data.mid(q)};
		auto it = Vin.data.dict.find(quple);
		
		Matrix<Scalar2,Dynamic,Dynamic> Mtmp;
		if (it != Vin.data.dict.end())
		{
			Mtmp = Vin.data.block[it->second][H.ab][0] 
			     - TxV.data.block[q][H.ab][0] 
			     + LdotR * Matrix<Scalar2,Dynamic,Dynamic>::Identity(Vin.data.block[it->second][H.ab][0].rows(),
			                                                         Vin.data.block[it->second][H.ab][0].cols());
		}
		
		if (Mtmp.size() != 0)
		{
			auto ip = Vout.data.dict.find(quple);
			if (ip != Vout.data.dict.end())
			{
				if (Vout.data.block[ip->second][H.ab][0].rows() != Mtmp.rows() or 
					Vout.data.block[ip->second][H.ab][0].cols() != Mtmp.cols())
				{
					Vout.data.block[ip->second][H.ab][0] = Mtmp;
				}
				else
				{
					Vout.data.block[ip->second][H.ab][0] += Mtmp;
				}
			}
			else
			{
				boost::multi_array<Matrix<Scalar2,Dynamic,Dynamic>,LEGLIMIT> Mtmpvec(boost::extents[H.W[0][0][0].cols()][1]);
				Mtmpvec[H.ab][0] = Mtmp;
				Vout.data.push_back(quple, Mtmpvec);
			}
		}
	}
	
//	Vout = Vin;
//	
//	if (H.LReigen.rows() == 0)
//	{
//		setZero(Vout.A);
//	}
//	
//	double factor = (H.LReigen.rows()==0)? +1.:-1.;
//	
//	auto index = [&H] (size_t s1, size_t s3) -> size_t {return s1*H.D[0]+s3;};
//	
//	// 1-cell
//	if (H.Aket.size() != 0)
//	{
//		if (H.gauge == GAUGE::R)
//		{
//			for (size_t s=0; s<H.D[0]; ++s)
//			{
//				Vout.A += factor * H.Wvec[s] * H.Aket[s].block[0] * Vin.A * H.Abra[s].block[0].adjoint();
//			}
//		}
//		else if (H.gauge == GAUGE::L)
//		{
//			for (size_t s=0; s<H.D[0]; ++s)
//			{
//				Vout.A += factor * H.Wvec[s] * H.Abra[s].block[0].adjoint() * Vin.A * H.Aket[s].block[0];
//			}
//		}
//	}
//	// 2-cell
//	else if (H.ApairKet.size() != 0)
//	{
//		if (H.gauge == GAUGE::R)
//		{
//			for (size_t s1=0; s1<H.D[0]; ++s1)
//			for (size_t s2=0; s2<H.D[0]; ++s2)
//			for (size_t s3=0; s3<H.D[1]; ++s3)
//			for (size_t s4=0; s4<H.D[1]; ++s4)
//			{
//				if (H.Warray[s1][s2][s3][s4] != 0.)
//				{
//					Vout.A += factor * H.Warray[s1][s2][s3][s4] * H.ApairKet[index(s2,s4)].block[0] * 
//					                                              Vin.A * 
//					                                              H.ApairBra[index(s1,s3)].block[0].adjoint();
//				}
//			}
//		}
//		else if (H.gauge == GAUGE::L)
//		{
//			for (size_t s1=0; s1<H.D[0]; ++s1)
//			for (size_t s2=0; s2<H.D[0]; ++s2)
//			for (size_t s3=0; s3<H.D[1]; ++s3)
//			for (size_t s4=0; s4<H.D[1]; ++s4)
//			{
//				if (H.Warray[s1][s2][s3][s4] != 0.)
//				{
//					Vout.A += factor * H.Warray[s1][s2][s3][s4] * H.ApairBra[index(s1,s3)].block[0].adjoint() * 
//					                                              Vin.A * 
//					                                              H.ApairKet[index(s2,s4)].block[0];
//				}
//			}
//		}
//	}
//	// 4-cell
//	else if (H.AquartettKet.size() != 0)
//	{
//		if (H.gauge == GAUGE::R)
//		{
//			for (size_t s1=0; s1<H.D[0]; ++s1)
//			for (size_t s2=0; s2<H.D[0]; ++s2)
//			for (size_t s3=0; s3<H.D[1]; ++s3)
//			for (size_t s4=0; s4<H.D[1]; ++s4)
//			for (size_t s5=0; s5<H.D[2]; ++s5)
//			for (size_t s6=0; s6<H.D[2]; ++s6)
//			for (size_t s7=0; s7<H.D[3]; ++s7)
//			for (size_t s8=0; s8<H.D[3]; ++s8)
//			{
//				if (H.Warray4[s1][s2][s3][s4][s5][s6][s7][s8] != 0.)
//				{
//					Vout.A += factor * H.Warray4[s1][s2][s3][s4][s5][s6][s7][s8] * 
//					          H.AquartettKet[s2][s4][s6][s8].block[0] * Vin.A * H.AquartettBra[s1][s3][s5][s7].block[0].adjoint();
//				}
//			}
//		}
//		else if (H.gauge == GAUGE::L)
//		{
//			for (size_t s1=0; s1<H.D[0]; ++s1)
//			for (size_t s2=0; s2<H.D[0]; ++s2)
//			for (size_t s3=0; s3<H.D[1]; ++s3)
//			for (size_t s4=0; s4<H.D[1]; ++s4)
//			for (size_t s5=0; s5<H.D[2]; ++s5)
//			for (size_t s6=0; s6<H.D[2]; ++s6)
//			for (size_t s7=0; s7<H.D[3]; ++s7)
//			for (size_t s8=0; s8<H.D[3]; ++s8)
//			{
//				if (H.Warray4[s1][s2][s3][s4][s5][s6][s7][s8] != 0.)
//				{
//					Vout.A += factor * 
//					        H.Warray4[s1][s2][s3][s4][s5][s6][s7][s8] * 
//					        H.AquartettBra[s1][s3][s5][s7].block[0].adjoint() * 
//					        Vin.A * 
//					        H.AquartettKet[s2][s4][s6][s8].block[0];
//				}
//			}
//		}
//	}
//	
//	if (H.LReigen.rows() != 0)
//	{
//		if (H.gauge == GAUGE::R)
//		{
//			Vout.A += (H.LReigen * Vin.A).trace() * Matrix<Scalar2,Dynamic,Dynamic>::Identity(Vin.A.rows(),Vin.A.cols());
//		}
//		else if (H.gauge == GAUGE::L)
//		{
//			Vout.A += (Vin.A * H.LReigen).trace() * Matrix<Scalar2,Dynamic,Dynamic>::Identity(Vin.A.rows(),Vin.A.cols());
//		}
//	}
}

template<typename Symmetry, typename Scalar1, typename Scalar2>
void HxV (const TransferMatrix<Symmetry,Scalar1> &H, TransferVector<Symmetry,Scalar2> &Vinout)
{
	TransferVector<Symmetry,Scalar2> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

template<typename Symmetry, typename Scalar>
inline size_t dim (const TransferMatrix<Symmetry,Scalar> &H)
{
//	if (H.Aket.size() != 0)
//	{
//		return H.Aket[0].block[0].cols() * H.Abra[0].block[0].rows();
//	}
//	else if (H.ApairKet.size() != 0)
//	{
//		return H.ApairKet[0].block[0].cols() * H.ApairBra[0].block[0].rows();
//	}
//	else if (H.AquartettKet.size() != 0)
//	{
//		return H.AquartettKet[0][0][0][0].block[0].cols() * H.AquartettBra[0][0][0][0].block[0].rows();
//	}
	
	size_t out = 0;
	for (size_t s=0; s<H.qloc.size(); ++s)
	for (size_t q=0; q<H.Aket[s].dim; ++q)
	{
		out += H.Aket[s].block[q].size();
	}
	return out;
}

template<typename Symmetry, typename Scalar>
inline size_t dim (const TransferVector<Symmetry,Scalar> &V)
{
	size_t out = 0;
	for (size_t q=0; q<V.data.dim; ++q)
	{
		out += V.data.block[q][V.ab][0].size();
	}
	return out;
}

template<typename Symmetry, typename Scalar>
inline Scalar squaredNorm (const TransferVector<Symmetry,Scalar> &V)
{
//	return V.A.squaredNorm();
	return dot(V,V);
}

template<typename Symmetry, typename Scalar>
inline Scalar norm (const TransferVector<Symmetry,Scalar> &V)
{
	return sqrt(squaredNorm(V));
//	return V.A.norm();
}

template<typename Symmetry, typename Scalar>
inline void normalize (TransferVector<Symmetry,Scalar> &V)
{
	V /= norm(V);
}

//template<typename Symmetry, typename Scalar>
//inline Scalar infNorm (const TransferVector<Symmetry,Scalar> &V1, const TransferVector<Symmetry,Scalar> &V2)
//{
//	return (V1-V2).template lpNorm<Eigen::Infinity>();
//}

template<typename Symmetry, typename Scalar>
void swap (TransferVector<Symmetry,Scalar> &V1, TransferVector<Symmetry,Scalar> &V2)
{
	V1.A.swap(V2.A);
}

template<typename Symmetry, typename Scalar>
inline Scalar dot (const TransferVector<Symmetry,Scalar> &V1, const TransferVector<Symmetry,Scalar> &V2)
{
//	return (V1.A.adjoint()*V2.A).trace();
	return contract_LR(V1.data,V2.data,true);
}

//-----------<vector arithmetics>-----------
template<typename Symmetry, typename Scalar>
TransferVector<Symmetry,Scalar>& TransferVector<Symmetry,Scalar>::
operator+= (const TransferVector<Symmetry,Scalar> &Vrhs)
{
	for (size_t q=0; q<Vrhs.data.dim; ++q)
	{
		qarray3<Symmetry::Nq> quple = {Vrhs.data.in(q), Vrhs.data.out(q), Vrhs.data.mid(q)};
		auto it = data.dict.find(quple);
		if (it != data.dict.end())
		{
			data.block[it->second][ab][0] += Vrhs.data.block[q][ab][0];
		}
	}
	return *this;
}

template<typename Symmetry, typename Scalar>
TransferVector<Symmetry,Scalar>& TransferVector<Symmetry,Scalar>::
operator-= (const TransferVector<Symmetry,Scalar> &Vrhs)
{
	for (size_t q=0; q<Vrhs.data.dim; ++q)
	{
		qarray3<Symmetry::Nq> quple = {Vrhs.data.in(q), Vrhs.data.out(q), Vrhs.data.mid(q)};
		auto it = data.dict.find(quple);
		if (it != data.dict.end())
		{
			data.block[it->second][ab][0] -= Vrhs.data.block[q][ab][0];
		}
	}
	return *this;
}

template<typename Symmetry, typename Scalar>
template<typename OtherScalar>
TransferVector<Symmetry,Scalar>& TransferVector<Symmetry,Scalar>::
operator*= (const OtherScalar &alpha)
{
	for (size_t q=0; q<data.dim; ++q)
	{
		data.block[q][ab][0] *= alpha;
	}
	return *this;
}

template<typename Symmetry, typename Scalar>
template<typename OtherScalar>
TransferVector<Symmetry,Scalar>& TransferVector<Symmetry,Scalar>::
operator/= (const OtherScalar &alpha)
{
	for (size_t q=0; q<data.dim; ++q)
	{
		data.block[q][ab][0] /= alpha;
	}
	return *this;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
TransferVector<Symmetry,Scalar> operator* (const OtherScalar &alpha, TransferVector<Symmetry,Scalar> V)
{
	return V *= alpha;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
TransferVector<Symmetry,Scalar> operator/ (TransferVector<Symmetry,Scalar> V, const OtherScalar &alpha)
{
	return V /= alpha;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
TransferVector<Symmetry,Scalar> operator+ (const TransferVector<Symmetry,Scalar> &V1, const TransferVector<Symmetry,Scalar> &V2)
{
	TransferVector<Symmetry,Scalar> Vout = V1;
	Vout.data += V2.data;
	return Vout;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
TransferVector<Symmetry,Scalar> operator- (const TransferVector<Symmetry,Scalar> &V1, const TransferVector<Symmetry,Scalar> &V2)
{
	TransferVector<Symmetry,Scalar> Vout = V1;
	Vout.data -= V2.data;
	return Vout;
}

template<typename Symmetry, typename Scalar>
inline void setZero (TransferVector<Symmetry,Scalar> &V)
{
	V.data.setZero();
}
//-----------</vector arithmetics>-----------

#include "RandomVector.h"

template<typename Symmetry, typename Scalar>
struct GaussianRandomVector<TransferVector<Symmetry,Scalar>,Scalar>
{
	static void fill (size_t N, TransferVector<Symmetry,Scalar> &Vout)
	{
		for (size_t q=0; q<Vout.data.dim; ++q)
		for (size_t i=0; i<Vout.data.block[q][Vout.ab][0].rows(); ++i)
		for (size_t j=0; j<Vout.data.block[q][Vout.ab][0].cols(); ++j)
		{
			Vout.data.block[q][Vout.ab][0](i,j) = threadSafeRandUniform<Scalar>(-1.,1.);
		}
		normalize(Vout);
	}
};

#endif

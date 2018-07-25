#ifndef VANILLA_VUMPSTRANSFERMATRIX
#define VANILLA_VUMPSTRANSFERMATRIX

#include "termcolor.hpp"

#include "VUMPS/VumpsTypedefs.h"
#include "tensors/Biped.h"
#include "tensors/Multipede.h"
#include "tensors/DmrgContractions.h"
#include "pivot/DmrgPivotVector.h"

/**
Operators \f$1-T_L+|R)(1|\f$, \f$1-T_R+|1)(R|\f$ for solving eq. C25ab.
\ingroup VUMPS
*/
template<typename Symmetry, typename Scalar>
struct TransferMatrix
{
	TransferMatrix(){};
	
	TransferMatrix (GAUGE::OPTION gauge_input, 
	                const vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > &Abra_input, 
	                const vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > &Aket_input, 
	                const Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &LReigen_input, 
	                const vector<vector<vector<vector<SparseMatrix<Scalar> > > > > &W_input,
	                const vector<vector<qarray<Symmetry::Nq> > > &qloc_input,
	                const vector<vector<qarray<Symmetry::Nq> > > &qOp_input,
	                size_t ab_input)
	:gauge(gauge_input), Abra(Abra_input), Aket(Aket_input), LReigen(LReigen_input), W(W_input), qloc(qloc_input), qOp(qOp_input), ab(ab_input)
	{}
	
	/**Gauge (L or R).*/
	GAUGE::OPTION gauge;
	
	///\{
	vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > Aket;
	vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > Abra;
	vector<vector<vector<vector<SparseMatrix<Scalar> > > > > W;
	///\}
	
	Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > LReigen;
	
	size_t ab;
	
	vector<vector<qarray<Symmetry::Nq> > > qloc;
	vector<vector<qarray<Symmetry::Nq> > > qOp;
};

/**
Vector \f$(L_a|\f$, \f$|R_a)\f$ that is obtained in eq. (C25ab).
\ingroup VUMPS
*/
template<typename Symmetry, typename Scalar>
struct TransferVector
{
	TransferVector(){};
	
	TransferVector (const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &T, const size_t &ab_input, const Scalar &LRdotY)
	:data(T), ab(ab_input)
	{
		for (size_t q=0; q<data.dim; ++q)
		{
			if (data.mid(q) == Symmetry::qvacuum())
			{
				data.block[q][ab][0] -= LRdotY * Matrix<Scalar,Dynamic,Dynamic>::Identity(data.block[q][ab][0].rows(),
				                                                                          data.block[q][ab][0].cols());
			}
			else
			{
				cout << termcolor::red << "ELSE" << termcolor::reset << endl;
			}
		}
	};
	
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

/**Matrix-vector multiplication in eq. (25ab)
\ingroup VUMPS*/
template<typename Symmetry, typename Scalar1, typename Scalar2>
void HxV (const TransferMatrix<Symmetry,Scalar1> &H, const TransferVector<Symmetry,Scalar2> &Vin, TransferVector<Symmetry,Scalar2> &Vout)
{
	Vout = Vin;
	Vout.data.setZero();
	size_t Lcell = H.W.size();
	
	TransferVector<Symmetry,Scalar2> TxV = Vin;
	TxV.data.setZero();
	
	if (H.gauge == GAUGE::R)
	{
		Tripod<Symmetry,Matrix<Scalar2,Dynamic,Dynamic> > Rnext;
		Tripod<Symmetry,Matrix<Scalar2,Dynamic,Dynamic> > R = Vin.data;
		for (int l=Lcell-1; l>=0; --l)
		{
			if (l==0 or l==Lcell-1)
			{
				contract_R(R, H.Abra[l], H.W[l], H.Aket[l], H.qloc[l], H.qOp[l], Rnext, false, make_pair(CONTRACT_LR_MODE::FIXED,H.ab));
			}
			else
			{
				contract_R(R, H.Abra[l], H.W[l], H.Aket[l], H.qloc[l], H.qOp[l], Rnext);
			}
			R.clear();
			R = Rnext;
			Rnext.clear();
		}
		TxV.data = R;
	}
	else if (H.gauge == GAUGE::L)
	{
		Tripod<Symmetry,Matrix<Scalar2,Dynamic,Dynamic> > Lnext;
		Tripod<Symmetry,Matrix<Scalar2,Dynamic,Dynamic> > L = Vin.data;
		for (size_t l=0; l<Lcell; ++l)
		{
			if (l==Lcell-1 or l==0)
			{
				contract_L(L, H.Abra[l], H.W[l], H.Aket[l], H.qloc[l], H.qOp[l], Lnext, false, make_pair(CONTRACT_LR_MODE::FIXED,H.ab));
			}
			else
			{
				contract_L(L, H.Abra[l], H.W[l], H.Aket[l], H.qloc[l], H.qOp[l], Lnext);
			}
			L.clear();
			L = Lnext;
			Lnext.clear();
		}
		TxV.data = L;
	}
	
	Scalar2 LdotR;
	if (H.gauge == GAUGE::R)
	{
		LdotR = contract_LR(H.ab, H.LReigen, Vin.data);
	}
	else if (H.gauge == GAUGE::L)
	{
		LdotR = contract_LR(H.ab, Vin.data, H.LReigen);
	}
	
	for (size_t q=0; q<TxV.data.dim; ++q)
	{
		qarray3<Symmetry::Nq> quple = {TxV.data.in(q), TxV.data.out(q), TxV.data.mid(q)};
		auto it = Vin.data.dict.find(quple);
		
		Matrix<Scalar2,Dynamic,Dynamic> Mtmp;
		if (it != Vin.data.dict.end())
		{
			Mtmp = Vin.data.block[it->second][H.ab][0] - TxV.data.block[q][H.ab][0];
			
			if (quple[2] == Symmetry::qvacuum())
			{
				Mtmp += LdotR * Matrix<Scalar2,Dynamic,Dynamic>::Identity(Vin.data.block[it->second][H.ab][0].rows(),
				                                                          Vin.data.block[it->second][H.ab][0].cols());
			}
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
				cout << termcolor::red << "push_back that shouldn't be" << termcolor::reset << endl;
				boost::multi_array<Matrix<Scalar2,Dynamic,Dynamic>,LEGLIMIT> Mtmpvec(boost::extents[H.W[0][0][0][0].cols()][1]);
				Mtmpvec[H.ab][0] = Mtmp;
				Vout.data.push_back(quple, Mtmpvec);
			}
		}
	}
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
//	size_t out = 0;
//	for (size_t s=0; s<H.Aket.size(); ++s)
//	for (size_t q=0; q<H.Aket[s].dim; ++q)
//	{
//		out += H.Aket[s].block[q].size();
//	}
//	return out;
	return 0;
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
	return dot(V,V);
}

template<typename Symmetry, typename Scalar>
inline Scalar norm (const TransferVector<Symmetry,Scalar> &V)
{
	return sqrt(squaredNorm(V));
}

template<typename Symmetry, typename Scalar>
inline void normalize (TransferVector<Symmetry,Scalar> &V)
{
	V /= norm(V);
}

template<typename Symmetry, typename Scalar>
inline Scalar dot (const TransferVector<Symmetry,Scalar> &V1, const TransferVector<Symmetry,Scalar> &V2)
{
	Scalar res = 0;
	for (size_t q=0; q<V1.data.size(); ++q)
	{
//		assert(V1.data.in(q) == V2.data.in(q));
//		assert(V1.data.out(q) == V2.data.out(q));
//		assert(V1.data.mid(q) == V2.data.mid(q) and V1.data.mid(q) == Symmetry::qvacuum());
////		cout << V1.data.in(q) << ", " << V1.data.out(q) << ", " << V1.data.mid(q) << " | " 
////		     << V2.data.in(q) << ", " << V2.data.out(q) << ", " << V2.data.mid(q) << endl;
		res += (V1.data.block[q][V1.ab][0].adjoint() * V2.data.block[q][V2.ab][0]).trace();
	}
	return res;
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
//		cout << data.in(q) << ", " << data.out(q) << ", " << data.mid(q) << " | " 
//		     << Vrhs.data.in(q) << ", " << Vrhs.data.out(q) << ", " << Vrhs.data.mid(q) << endl;
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

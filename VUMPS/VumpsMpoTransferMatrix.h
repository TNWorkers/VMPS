#ifndef VANILLA_VUMPS_MPO_TRANSFERMATRIX
#define VANILLA_VUMPS_MPO_TRANSFERMATRIX

/// \cond
#include "termcolor.hpp"
/// \endcond

#include "VUMPS/VumpsTypedefs.h"

//include "tensors/Biped.h"
//include "tensors/Multipede.h"
//include "tensors/DmrgContractions.h"
//include "RandomVector.h"

/**
Operators \f$1-T_L+|R><1|\f$, \f$1-T_R+|1><R|\f$ for solving eq. C25ab.
\ingroup VUMPS
*/
template<typename Symmetry, typename Scalar>
struct MpoTransferMatrix
{
	MpoTransferMatrix(){};
	
	MpoTransferMatrix (VMPS::DIRECTION::OPTION DIR_input, 
	                   const vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > &Abra_input, 
	                   const vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > &Aket_input, 
	                   const Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &LReigen_input, 
	                   const vector<vector<vector<vector<Biped<Symmetry,SparseMatrix<Scalar> > > > > > &W_input,
	                   const vector<vector<qarray<Symmetry::Nq> > > &qloc_input,
	                   const vector<vector<qarray<Symmetry::Nq> > > &qOp_input,
	                   size_t ab_input,
					   const std::unordered_map<pair<qarray<Symmetry::Nq>,size_t>,size_t> &basis_order_map_input = {},
					   const vector<pair<qarray<Symmetry::Nq>,size_t> > &basis_order_imput={})
	:DIR(DIR_input), Abra(Abra_input), Aket(Aket_input), LReigen(LReigen_input), W(W_input), qloc(qloc_input), qOp(qOp_input), ab(ab_input), basis_order_map(basis_order_map_input), basis_order(basis_order_imput)
	{}
	
	/**Gauge (L or R).*/
	VMPS::DIRECTION::OPTION DIR;
	
	///\{
	vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > Abra;
	vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > Aket;
	vector<vector<vector<vector<Biped<Symmetry, SparseMatrix<Scalar> > > > > > W;
	///\}
	
	Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > LReigen;
	
	size_t ab;
	std::unordered_map<pair<qarray<Symmetry::Nq>,size_t>,size_t> basis_order_map;
	vector<pair<qarray<Symmetry::Nq>,size_t> > basis_order;
	
	vector<vector<qarray<Symmetry::Nq> > > qloc;
	vector<vector<qarray<Symmetry::Nq> > > qOp;
};

/**
Vector \f$<L_a|\f$, \f$|R_a>\f$ that is obtained in eq. (C25ab).
\ingroup VUMPS
*/
template<typename Symmetry, typename Scalar_>
struct MpoTransferVector
{
	typedef Scalar_ Scalar;
	
	MpoTransferVector(){};
	
	// When called for the VUMPS ground state algorithm, ab_input and LRdotY are set.
	// When called with StructureFactor, they are equal to zero.
	MpoTransferVector (const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &T, const pair<qarray<Symmetry::Nq>,size_t> &ab_input, const Scalar &LRdotY=0.)
	:data(T), ab(ab_input)
	{
		if (LRdotY != 0.)
		{
			for (size_t q=0; q<data.dim; ++q)
			{
				if (data.mid(q) == ab.first)
				{
					data.block[q][ab.second][0] -= LRdotY * Matrix<Scalar,Dynamic,Dynamic>::Identity(data.block[q][ab.second][0].rows(), data.block[q][ab.second][0].cols());
				}
			}
		}
	};
	
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > data;
	pair<qarray<Symmetry::Nq>,size_t> ab;
	
	///\{
	/**Linear algebra in the correspodnding vector space.*/
	MpoTransferVector<Symmetry,Scalar>& operator+= (const MpoTransferVector<Symmetry,Scalar> &Vrhs);
	MpoTransferVector<Symmetry,Scalar>& operator-= (const MpoTransferVector<Symmetry,Scalar> &Vrhs);
	template<typename OtherScalar> MpoTransferVector<Symmetry,Scalar>& operator*= (const OtherScalar &alpha);
	template<typename OtherScalar> MpoTransferVector<Symmetry,Scalar>& operator/= (const OtherScalar &alpha);
	///\}
};

/**Matrix-vector multiplication in eq. (25ab)
\ingroup VUMPS*/
template<typename Symmetry, typename Scalar1, typename Scalar2>
void HxV (const MpoTransferMatrix<Symmetry,Scalar1> &H, const MpoTransferVector<Symmetry,Scalar2> &Vin, MpoTransferVector<Symmetry,Scalar2> &Vout)
{
	Vout = Vin;
	Vout.data.setZero();
	size_t Lcell = H.W.size();
	
	MpoTransferVector<Symmetry,Scalar2> TxV = Vin;
	TxV.data.setZero();
	
	if (H.DIR == VMPS::DIRECTION::RIGHT)
	{
		Tripod<Symmetry,Matrix<Scalar2,Dynamic,Dynamic> > Rnext;
		Tripod<Symmetry,Matrix<Scalar2,Dynamic,Dynamic> > R = Vin.data;
		for (int l=Lcell-1; l>=0; --l)
		{
//			if (l==0 or l==Lcell-1)
			if (l==0)
			{
				contract_R(R, H.Abra[l], H.W[l], H.Aket[l], H.qloc[l], H.qOp[l], Rnext, false, make_pair(CONTRACT_LR_MODE::FIXED_ROWS,H.ab), H.basis_order_map);
			}
			else if (l==Lcell-1)
			{
				contract_R(R, H.Abra[l], H.W[l], H.Aket[l], H.qloc[l], H.qOp[l], Rnext, false, make_pair(CONTRACT_LR_MODE::FIXED_COLS,H.ab), H.basis_order_map);
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
		
//		cout << "MpoTransferVector R:" << endl << TxV.data.print(true,13) << endl;
	}
	else if (H.DIR == VMPS::DIRECTION::LEFT)
	{
		Tripod<Symmetry,Matrix<Scalar2,Dynamic,Dynamic> > Lnext;
		Tripod<Symmetry,Matrix<Scalar2,Dynamic,Dynamic> > L = Vin.data;
		for (size_t l=0; l<Lcell; ++l)
		{
//			if (l==Lcell-1 or l==0)
			if (l==Lcell-1)
			{
				contract_L(L, H.Abra[l], H.W[l], H.Aket[l], H.qloc[l], H.qOp[l], Lnext, false, make_pair(CONTRACT_LR_MODE::FIXED_COLS,H.ab), H.basis_order_map);
			}
			else if (l==0)
			{
				contract_L(L, H.Abra[l], H.W[l], H.Aket[l], H.qloc[l], H.qOp[l], Lnext, false, make_pair(CONTRACT_LR_MODE::FIXED_ROWS,H.ab), H.basis_order_map);
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
		
//		cout << "MpoTransferVector L:" << endl << TxV.data.print(true,13) << endl;
	}

	// find quantum number and inner basis element for H.ab
	pair<qarray<Symmetry::Nq>,size_t> ab_blocked = H.basis_order[H.ab];
	Scalar2 LdotR;
	if (H.DIR == VMPS::DIRECTION::RIGHT)
	{
		LdotR = contract_LR(ab_blocked, H.LReigen, Vin.data);
	}
	else if (H.DIR == VMPS::DIRECTION::LEFT)
	{
		LdotR = contract_LR(ab_blocked, Vin.data, H.LReigen);
	}
	
	for (size_t q=0; q<TxV.data.dim; ++q)
	{
		qarray3<Symmetry::Nq> quple = {TxV.data.in(q), TxV.data.out(q), TxV.data.mid(q)};
		auto it = Vin.data.dict.find(quple);
		
		Matrix<Scalar2,Dynamic,Dynamic> Mtmp;
		if (it != Vin.data.dict.end())
		{
			Mtmp = Vin.data.block[it->second][ab_blocked.second][0] - TxV.data.block[q][ab_blocked.second][0];
			Mtmp += LdotR * Matrix<Scalar2,Dynamic,Dynamic>::Identity(Vin.data.block[it->second][ab_blocked.second][0].rows(),
			                                                          Vin.data.block[it->second][ab_blocked.second][0].cols());
		}
		
		if (Mtmp.size() != 0)
		{
			auto ip = Vout.data.dict.find(quple);
			if (ip != Vout.data.dict.end())
			{
				if (Vout.data.block[ip->second][ab_blocked.second][0].rows() != Mtmp.rows() or 
				    Vout.data.block[ip->second][ab_blocked.second][0].cols() != Mtmp.cols())
				{
					Vout.data.block[ip->second][ab_blocked.second][0] = Mtmp;
				}
				else
				{
					Vout.data.block[ip->second][ab_blocked.second][0] += Mtmp;
				}
			}
			else
			{
				cout << termcolor::red << "push_back that shouldn't be" << termcolor::reset << endl;
				boost::multi_array<Matrix<Scalar2,Dynamic,Dynamic>,LEGLIMIT> Mtmpvec(boost::extents[H.W[0][0][0][0].block[0].cols()][1]);
				Mtmpvec[ab_blocked.second][0] = Mtmp;
				Vout.data.push_back(quple, Mtmpvec);
			}
		}
	}
}

template<typename Symmetry, typename Scalar1, typename Scalar2>
void HxV (const MpoTransferMatrix<Symmetry,Scalar1> &H, MpoTransferVector<Symmetry,Scalar2> &Vinout)
{
	MpoTransferVector<Symmetry,Scalar2> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

template<typename Symmetry, typename Scalar>
inline size_t dim (const MpoTransferMatrix<Symmetry,Scalar> &H)
{
	return 0;
}

template<typename Symmetry, typename Scalar>
inline size_t dim (const MpoTransferVector<Symmetry,Scalar> &V)
{
	size_t out = 0;
	for (size_t q=0; q<V.data.dim; ++q)
	{
		if (V.data.mid(q) != V.ab.first) {continue;}
		out += V.data.block[q][V.ab.second][0].size();
	}
	return out;
}

template<typename Symmetry, typename Scalar>
inline double squaredNorm (const MpoTransferVector<Symmetry,Scalar> &V)
{
	return isReal(dot(V,V));
}

template<typename Symmetry, typename Scalar>
inline double norm (const MpoTransferVector<Symmetry,Scalar> &V)
{
	return sqrt(squaredNorm(V));
}

template<typename Symmetry, typename Scalar>
inline void normalize (MpoTransferVector<Symmetry,Scalar> &V)
{
	V /= norm(V);
}

template<typename Symmetry, typename Scalar>
inline Scalar dot (const MpoTransferVector<Symmetry,Scalar> &V1, const MpoTransferVector<Symmetry,Scalar> &V2)
{
	Scalar res = 0;
	for (size_t q=0; q<V1.data.size(); ++q)
	{
		if (V1.data.mid(q) != V1.ab.first) {continue;}
		// Note: qmid is not necessarily the vacuum for the structure factor (TransferMatrixSF)!
		qarray3<Symmetry::Nq> quple = {V1.data.in(q), V1.data.out(q), V1.data.mid(q)};
		auto it = V2.data.dict.find(quple);
		if (it != V2.data.dict.end())
		{
			if (V2.data.mid(it->second) != V2.ab.first) {continue;}
			res += (V1.data.block[q][V1.ab.second][0].adjoint() * V2.data.block[it->second][V2.ab.second][0]).trace() * Symmetry::coeff_dot(V2.data.out(q));
		}
	}
	return res;
}

template<typename Symmetry, typename Scalar>
MpoTransferVector<Symmetry,Scalar>& MpoTransferVector<Symmetry,Scalar>::
operator+= (const MpoTransferVector<Symmetry,Scalar> &Vrhs)
{
	data.addScale(1.,Vrhs.data);
	return *this;

	// for (size_t q=0; q<Vrhs.data.dim; ++q)
	// {
	// 	qarray3<Symmetry::Nq> quple = {Vrhs.data.in(q), Vrhs.data.out(q), Vrhs.data.mid(q)};
	// 	auto it = data.dict.find(quple);
	// 	if (it != data.dict.end())
	// 	{
	// 		data.block[it->second][ab][0] += Vrhs.data.block[q][ab][0];
	// 	}
	// }
	// return *this;
}

template<typename Symmetry, typename Scalar>
MpoTransferVector<Symmetry,Scalar>& MpoTransferVector<Symmetry,Scalar>::
operator-= (const MpoTransferVector<Symmetry,Scalar> &Vrhs)
{
	data.addScale(-1.,Vrhs.data);
	return *this;

	// for (size_t q=0; q<Vrhs.data.dim; ++q)
	// {
	// 	qarray3<Symmetry::Nq> quple = {Vrhs.data.in(q), Vrhs.data.out(q), Vrhs.data.mid(q)};
	// 	auto it = data.dict.find(quple);
	// 	if (it != data.dict.end())
	// 	{
	// 		data.block[it->second][ab][0] -= Vrhs.data.block[q][ab][0];
	// 	}
	// }
	// return *this;
}

template<typename Symmetry, typename Scalar>
template<typename OtherScalar>
MpoTransferVector<Symmetry,Scalar>& MpoTransferVector<Symmetry,Scalar>::
operator*= (const OtherScalar &alpha)
{
	for (size_t q=0; q<data.dim; ++q)
	{
		if (data.mid(q) != ab.first) {continue;}
		data.block[q][ab.second][0] *= alpha;
	}
	return *this;
}

template<typename Symmetry, typename Scalar>
template<typename OtherScalar>
MpoTransferVector<Symmetry,Scalar>& MpoTransferVector<Symmetry,Scalar>::
operator/= (const OtherScalar &alpha)
{
	for (size_t q=0; q<data.dim; ++q)
	{
		if (data.mid(q) != ab.first) {continue;}
		data.block[q][ab.second][0] /= alpha;
	}
	return *this;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
MpoTransferVector<Symmetry,Scalar> operator* (const OtherScalar &alpha, MpoTransferVector<Symmetry,Scalar> V)
{
	return V *= alpha;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
MpoTransferVector<Symmetry,Scalar> operator/ (MpoTransferVector<Symmetry,Scalar> V, const OtherScalar &alpha)
{
	return V /= alpha;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
MpoTransferVector<Symmetry,Scalar> operator+ (const MpoTransferVector<Symmetry,Scalar> &V1, const MpoTransferVector<Symmetry,Scalar> &V2)
{
	MpoTransferVector<Symmetry,Scalar> Vout = V1;
	Vout.data += V2.data;
	return Vout;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
MpoTransferVector<Symmetry,Scalar> operator- (const MpoTransferVector<Symmetry,Scalar> &V1, const MpoTransferVector<Symmetry,Scalar> &V2)
{
	MpoTransferVector<Symmetry,Scalar> Vout = V1;
	Vout.data -= V2.data;
	return Vout;
}

template<typename Symmetry, typename Scalar>
inline void setZero (MpoTransferVector<Symmetry,Scalar> &V)
{
	V.data.setZero();
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
inline void addScale (const OtherScalar alpha, const MpoTransferVector<Symmetry,Scalar> &Vin, MpoTransferVector<Symmetry,Scalar> &Vout)
{
	Vout.data.addScale(alpha,Vin.data);
	// Vout += alpha * Vin;
}

template<typename Symmetry, typename Scalar>
struct GaussianRandomVector<MpoTransferVector<Symmetry,Scalar>,Scalar>
{
	static void fill (size_t N, MpoTransferVector<Symmetry,Scalar> &Vout)
	{
		for (size_t q=0; q<Vout.data.dim; ++q)
		{
			if (Vout.data.mid(q) != Vout.ab.first) {continue;}
			for (size_t i=0; i<Vout.data.block[q][Vout.ab.second][0].rows(); ++i)
			for (size_t j=0; j<Vout.data.block[q][Vout.ab.second][0].cols(); ++j)
			{
				Vout.data.block[q][Vout.ab.second][0](i,j) = threadSafeRandUniform<Scalar>(-1.,1.);
			}
		}
		normalize(Vout);
	}
};

#endif

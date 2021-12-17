#ifndef VANILLA_VUMPS_TRANSFERMATRIX_AA
#define VANILLA_VUMPS_TRANSFERMATRIX_AA

#include "VUMPS/VumpsTypedefs.h"
//#include "pivot/DmrgPivotVector.h"

/**
Operators \f$T_L\f$, \f$T_R\f$ for solving the linear systems eq. 14.
\ingroup VUMPS
*/
template<typename Symmetry, typename Scalar>
struct TransferMatrix
{
	TransferMatrix(){};
	
	TransferMatrix (VMPS::DIRECTION::OPTION DIR_input)
	:DIR(DIR_input)
	{};
	
	TransferMatrix (VMPS::DIRECTION::OPTION DIR_input, 
	                const vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > &Abra_input, 
	                const vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > &Aket_input, 
	                const vector<vector<qarray<Symmetry::Nq> > > &qloc_input,
	                bool SHIFTED_input = false,
	                qarray<Symmetry::Nq> Qtot_input = Symmetry::qvacuum())
	:DIR(DIR_input), Abra(Abra_input), Aket(Aket_input), qloc(qloc_input), SHIFTED(SHIFTED_input), Qtot(Qtot_input)
	{}
	
	VMPS::DIRECTION::OPTION DIR;
	
	///\{
	vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > Aket;
	vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > Abra;
	///\}
	
	Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > LReigen;
	
	bool SHIFTED = false; // true for solve_linear with 2-site Hamiltonian, code commented out and needs review
	
	vector<vector<qarray<Symmetry::Nq> > > qloc;
	qarray<Symmetry::Nq> Qtot;
	
	Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic>> TopEigvec;
	Scalar TopEigval;
	bool PROJECT_OUT_TOPEIGVEC = false;
};

template<typename Symmetry, typename Scalar>
inline size_t dim (const TransferMatrix<Symmetry,Scalar> &H)
{
	return 0;
}

template<typename Symmetry, typename Scalar_>
struct TransferVector
{
	typedef Scalar_ Scalar;
	
	TransferVector(){};
	
	TransferVector (const Biped<Symmetry,Matrix<Scalar_,Dynamic,Dynamic> > &data_input)
	{
		data = data_input;
	};
	
	Biped<Symmetry,Matrix<Scalar_,Dynamic,Dynamic> > &operator[] (size_t i)             {return data[i];}
	Biped<Symmetry,Matrix<Scalar_,Dynamic,Dynamic> > &operator() (size_t i)             {return data[i];}
	const Biped<Symmetry,Matrix<Scalar_,Dynamic,Dynamic> > &operator[] (size_t i) const {return data[i];}
	const Biped<Symmetry,Matrix<Scalar_,Dynamic,Dynamic> > &operator() (size_t i) const {return data[i];}
	
	TransferVector<Symmetry,Scalar_>& operator+= (const TransferVector<Symmetry,Scalar_> &Vrhs);
	TransferVector<Symmetry,Scalar_>& operator-= (const TransferVector<Symmetry,Scalar_> &Vrhs);
	template<typename OtherScalar> TransferVector<Symmetry,Scalar_>& operator*= (const OtherScalar &alpha);
	template<typename OtherScalar> TransferVector<Symmetry,Scalar_>& operator/= (const OtherScalar &alpha);
	
	Biped<Symmetry,Matrix<Scalar_,Dynamic,Dynamic> > data;
};

template<typename Symmetry, typename Scalar1, typename Scalar2>
void HxV (const TransferMatrix<Symmetry,Scalar1> &H, const TransferVector<Symmetry,Scalar2> &Vin, TransferVector<Symmetry,Scalar2> &Vout)
{
	Vout.data.clear();
	size_t Lcell = H.qloc.size();
	
	assert(H.SHIFTED == false);
	
	if (H.SHIFTED == false)
	{
		if (H.DIR == VMPS::DIRECTION::RIGHT) // contract from right to left
		{
			Biped<Symmetry,Matrix<Scalar2,Dynamic,Dynamic> > Rnext;
			Biped<Symmetry,Matrix<Scalar2,Dynamic,Dynamic> > R = Vin.data;
			for (int l=Lcell-1; l>=0; --l)
			{
				contract_R(R, H.Abra[l], H.Aket[l], H.qloc[l], Rnext); // RANDOMIZE=false, CLEAR=false
				R.clear();
				R = Rnext;
				Rnext.clear();
			}
			Vout.data = R;
		}
		else if (H.DIR == VMPS::DIRECTION::LEFT) // contract from left to right
		{
			Biped<Symmetry,Matrix<Scalar2,Dynamic,Dynamic> > Lnext;
			Biped<Symmetry,Matrix<Scalar2,Dynamic,Dynamic> > L = Vin.data;
			for (size_t l=0; l<Lcell; ++l)
			{
				contract_L(L, H.Abra[l], H.Aket[l], H.qloc[l], Lnext); // RANDOMIZE=false, CLEAR=false
				L.clear();
				L = Lnext;
				Lnext.clear();
			}
			Vout.data = L;
		}
	}
//	else
//	{
//		Vout = Vin;
//		Vout.setZero();
//		
//		PivotVector<Symmetry,Scalar2> TxV = Vin;
//		TxV.setZero();
//		
//		if (H.gauge == GAUGE::R)
//		{
//			Biped<Symmetry,Matrix<Scalar2,Dynamic,Dynamic> > Rnext;
//			Biped<Symmetry,Matrix<Scalar2,Dynamic,Dynamic> > R = Vin.data[0];
//			for (int l=Lcell-1; l>=0; --l)
//			{
//				contract_R(R, H.Abra[l], H.Aket[l], H.qloc[l], Rnext);
//				R.clear();
//				R = Rnext;
//				Rnext.clear();
//			}
//			Vout.data[0] = R;
//		}
//		else if (H.gauge == GAUGE::L)
//		{
//			Biped<Symmetry,Matrix<Scalar2,Dynamic,Dynamic> > Lnext;
//			Biped<Symmetry,Matrix<Scalar2,Dynamic,Dynamic> > L = Vin.data[0];
//			for (size_t l=0; l<Lcell; ++l)
//			{
//				contract_L (L, H.Abra[l], H.Aket[l], H.qloc[l], Lnext);
//				L.clear();
//				L = Lnext;
//				Lnext.clear();
//			}
//			Vout.data[0] = L;
//		}
//		
//		Scalar2 LdotR;
//		if (H.gauge == GAUGE::R)
//		{
////			LdotR = (H.LReigen.contract(Vin.data[0])).trace();
//			LdotR = (H.LReigen.template cast<Matrix<Scalar2,Dynamic,Dynamic> >().contract(Vin.data[0])).trace();
//		}
//		else if (H.gauge == GAUGE::L)
//		{
////			LdotR = (Vin.data[0].contract(H.LReigen)).trace();
//			LdotR = (Vin.data[0].contract(H.LReigen.template cast<Matrix<Scalar2,Dynamic,Dynamic> >())).trace();
//		}
//		
//		for (size_t q=0; q<TxV.data[0].dim; ++q)
//		{
//			qarray2<Symmetry::Nq> quple = {TxV.data[0].in[q], TxV.data[0].out[q]};
//			auto it = Vin.data[0].dict.find(quple);
//			
//			Matrix<Scalar2,Dynamic,Dynamic> Mtmp;
//			if (it != Vin.data[0].dict.end())
//			{
//				Mtmp = Vin.data[0].block[it->second] - TxV.data[0].block[q] +
//				       LdotR * Matrix<Scalar2,Dynamic,Dynamic>::Identity(Vin.data[0].block[it->second].rows(),
//				                                                         Vin.data[0].block[it->second].cols());
//			}
//			
//			if (Mtmp.size() != 0)
//			{
//				auto ip = Vout.data[0].dict.find(quple);
//				if (ip != Vout.data[0].dict.end())
//				{
//					if (Vout.data[0].block[ip->second].rows() != Mtmp.rows() or 
//						Vout.data[0].block[ip->second].cols() != Mtmp.cols())
//					{
//						Vout.data[0].block[ip->second] = Mtmp;
//					}
//					else
//					{
//						Vout.data[0].block[ip->second] += Mtmp;
//					}
//				}
//				else
//				{
//					cout << termcolor::red << "push_back that shouldn't be: TransferMatrix" << termcolor::reset << endl;
//					Vout.data[0].push_back(quple, Mtmp);
//				}
//			}
//		}
//	}
	
	if (H.PROJECT_OUT_TOPEIGVEC)
	{
		Vout.data.addScale(-(H.TopEigval * H.TopEigvec.template cast<Matrix<Scalar2,Dynamic,Dynamic> >().adjoint().contract(Vin.data).trace()), 
		                    H.TopEigvec.template cast<Matrix<Scalar2,Dynamic,Dynamic> >());
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
inline size_t dim (const TransferVector<Symmetry,Scalar> &V)
{
	size_t out = 0;
	for (size_t q=0; q<V.data.dim; ++q)
	{
		out += V.data.block[q].size();
	}
	return out;
}

template<typename Symmetry, typename Scalar>
inline double squaredNorm (const TransferVector<Symmetry,Scalar> &V)
{
	return isReal(dot(V,V));
}

template<typename Symmetry, typename Scalar>
inline double norm (const TransferVector<Symmetry,Scalar> &V)
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
//	cout << "begin dot" << endl;
//	cout << "dot res=" << (V1.data.adjoint() * V2.data).trace() << endl;
//	return (V1.data.adjoint() * V2.data).trace();
	return V1.data.adjoint().contract(V2.data).trace();
}

template<typename Symmetry, typename Scalar>
TransferVector<Symmetry,Scalar>& TransferVector<Symmetry,Scalar>::
operator+= (const TransferVector<Symmetry,Scalar> &Vrhs)
{
	data.addScale(1.,Vrhs.data);
	return *this;
}

template<typename Symmetry, typename Scalar>
TransferVector<Symmetry,Scalar>& TransferVector<Symmetry,Scalar>::
operator-= (const TransferVector<Symmetry,Scalar> &Vrhs)
{
	data.addScale(-1.,Vrhs.data);
	return *this;
}

template<typename Symmetry, typename Scalar>
template<typename OtherScalar>
TransferVector<Symmetry,Scalar>& TransferVector<Symmetry,Scalar>::
operator*= (const OtherScalar &alpha)
{
	for (size_t q=0; q<data.dim; ++q)
	{
		data.block[q] *= alpha;
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
		data.block[q] /= alpha;
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

template<typename Symmetry, typename Scalar>
TransferVector<Symmetry,Scalar> operator+ (const TransferVector<Symmetry,Scalar> &V1, const TransferVector<Symmetry,Scalar> &V2)
{
	TransferVector<Symmetry,Scalar> Vout = V1;
	Vout.data.addScale(+1.,V2.data);
	return Vout;
}

template<typename Symmetry, typename Scalar>
TransferVector<Symmetry,Scalar> operator- (const TransferVector<Symmetry,Scalar> &V1, const TransferVector<Symmetry,Scalar> &V2)
{
	TransferVector<Symmetry,Scalar> Vout = V1;
	Vout.data.addScale(-1.,V2.data);
	return Vout;
}

template<typename Symmetry, typename Scalar>
inline void setZero (TransferVector<Symmetry,Scalar> &V)
{
	V.data.setZero();
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
inline void addScale (const OtherScalar alpha, const TransferVector<Symmetry,Scalar> &Vin, TransferVector<Symmetry,Scalar> &Vout)
{
	Vout.data.addScale(alpha,Vin.data);
}

template<typename Symmetry, typename Scalar>
struct GaussianRandomVector<TransferVector<Symmetry,Scalar>,Scalar>
{
	static void fill (size_t N, TransferVector<Symmetry,Scalar> &Vout)
	{
		for (size_t q=0; q<Vout.data.dim; ++q)
		for (size_t i=0; i<Vout.data.block[q].rows(); ++i)
		for (size_t j=0; j<Vout.data.block[q].cols(); ++j)
		{
			Vout.data.block[q](i,j) = threadSafeRandUniform<Scalar>(-1.,1.);
		}
		normalize(Vout);
	}
};

#endif

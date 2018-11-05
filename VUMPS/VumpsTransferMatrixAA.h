#ifndef VANILLA_VUMPSTRANSFERMATRIXAA
#define VANILLA_VUMPSTRANSFERMATRIXAA

#include "VUMPS/VumpsTypedefs.h"
#include "pivot/DmrgPivotVector.h"

/**
Operators \f$T_L\f$, \f$T_R\f$ for solving the linear systems eq. 14.
\ingroup VUMPS
*/
template<typename Symmetry, typename Scalar>
struct TransferMatrixAA
{
	TransferMatrixAA(){};
	
	TransferMatrixAA (GAUGE::OPTION gauge_input, 
	                  const vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > &Abra_input, 
	                  const vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > &Aket_input, 
	                  const vector<vector<qarray<Symmetry::Nq> > > &qloc_input,
	                  bool SHIFTED_input = false)
	:gauge(gauge_input), Abra(Abra_input), Aket(Aket_input), qloc(qloc_input), SHIFTED(SHIFTED_input)
	{}
	
	GAUGE::OPTION gauge;
	
	///\{
	vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > Aket;
	vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > Abra;
	///\}
	
	Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > LReigen;
	
	bool SHIFTED = false;
	
	vector<vector<qarray<Symmetry::Nq> > > qloc;
};

template<typename Symmetry, typename Scalar>
inline size_t dim (const TransferMatrixAA<Symmetry,Scalar> &H)
{
	return 0;
}

template<typename Symmetry, typename Scalar1, typename Scalar2>
void HxV (const TransferMatrixAA<Symmetry,Scalar1> &H, const PivotVector<Symmetry,Scalar2> &Vin, PivotVector<Symmetry,Scalar2> &Vout)
{
//	Vout.outerResize(Vin);
	size_t Lcell = H.qloc.size();
	
	assert(H.SHIFTED == false);
	
	if (H.SHIFTED == false)
	{
		if (H.gauge == GAUGE::L)
		{
			Biped<Symmetry,Matrix<Scalar2,Dynamic,Dynamic> > Rnext;
			Biped<Symmetry,Matrix<Scalar2,Dynamic,Dynamic> > R = Vin.data[0];
			for (int l=Lcell-1; l>=0; --l)
			{
				contract_R(R, H.Abra[l], H.Aket[l], H.qloc[l], Rnext);
				R.clear();
				R = Rnext;
				Rnext.clear();
			}
			Vout.data[0] = R;
		}
		else if (H.gauge == GAUGE::R)
		{
			Biped<Symmetry,Matrix<Scalar2,Dynamic,Dynamic> > Lnext;
			Biped<Symmetry,Matrix<Scalar2,Dynamic,Dynamic> > L = Vin.data[0];
			for (size_t l=0; l<Lcell; ++l)
			{
				contract_L(L, H.Abra[l], H.Aket[l], H.qloc[l], Lnext);
				L.clear();
				L = Lnext;
				Lnext.clear();
			}
			Vout.data[0] = L;
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
//					cout << termcolor::red << "push_back that shouldn't be: TransferMatrixAA" << termcolor::reset << endl;
//					Vout.data[0].push_back(quple, Mtmp);
//				}
//			}
//		}
//	}
}

template<typename Symmetry, typename Scalar1, typename Scalar2>
void HxV (const TransferMatrixAA<Symmetry,Scalar1> &H, PivotVector<Symmetry,Scalar2> &Vinout)
{
	PivotVector<Symmetry,Scalar2> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

#endif

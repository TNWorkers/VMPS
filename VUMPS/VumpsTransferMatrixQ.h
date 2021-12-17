#ifndef VANILLA_VUMPS_TRANSFERMATRIXQ
#define VANILLA_VUMPS_TRANSFERMATRIXQ

#include "VUMPS/VumpsMpoTransferMatrix.h"

template<typename Symmetry, typename Scalar>
struct TransferMatrixQ
{
	TransferMatrixQ(){};
	
	TransferMatrixQ (VMPS::DIRECTION::OPTION DIR_input, 
	                  const vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > &Abra_input, 
	                  const vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > &Aket_input, 
	                  const vector<vector<qarray<Symmetry::Nq> > > &qloc_input,
	                  const typename Symmetry::qType& Qtot = Symmetry::qvacuum())
	:DIR(DIR_input), Abra(Abra_input), Aket(Aket_input), qloc(qloc_input)
	{
		Id = Mpo<Symmetry,Scalar>::Identity(qloc,Qtot);
	}
	
	VMPS::DIRECTION::OPTION DIR;
	
	Mpo<Symmetry,Scalar> Id;
	
	///\{
	vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > Abra;
	vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > Aket;
	///\}
	
	vector<vector<qarray<Symmetry::Nq> > > qloc;
};

template<typename Symmetry, typename Scalar1, typename Scalar2>
void HxV (const TransferMatrixQ<Symmetry,Scalar1> &H, const MpoTransferVector<Symmetry,Scalar2> &Vin, MpoTransferVector<Symmetry,Scalar2> &Vout)
{
	Vout.data.clear();
	size_t Lcell = H.qloc.size();
	
	Tripod<Symmetry,Matrix<Scalar2,Dynamic,Dynamic> > TxV;
	
	if (H.DIR == VMPS::DIRECTION::RIGHT)
	{
		// Calculate T*|Vin>
		Tripod<Symmetry,Matrix<Scalar2,Dynamic,Dynamic> > Rnext;
		Tripod<Symmetry,Matrix<Scalar2,Dynamic,Dynamic> > R = Vin.data;
		for (int l=Lcell-1; l>=0; --l)
		{
			contract_R(R, H.Abra[l], H.Id.W_at(l), H.Aket[l], H.qloc[l], H.Id.opBasis(l), Rnext);
			R.clear();
			R = Rnext;
			Rnext.clear();
		}
		TxV = R;
	}
	else if (H.DIR == VMPS::DIRECTION::LEFT)
	{
		// Calculate <Vin|*T
		Tripod<Symmetry,Matrix<Scalar2,Dynamic,Dynamic> > Lnext;
		Tripod<Symmetry,Matrix<Scalar2,Dynamic,Dynamic> > L = Vin.data;
		for (size_t l=0; l<Lcell; ++l)
		{
			contract_L(L, H.Abra[l], H.Id.W_at(l), H.Aket[l], H.qloc[l], H.Id.opBasis(l), Lnext);
			L.clear();
			L = Lnext;
			Lnext.clear();
		}
		TxV = L;
	}
	else
	{
		assert(1==0 and "Unknown VMPS::DIRECTION::OPTION in TransferMatrixQ!");
	}
	
	Vout = MpoTransferVector<Symmetry,Scalar2>(TxV, make_pair(TxV.mid(0),0));
}

template<typename Symmetry, typename Scalar1, typename Scalar2>
void HxV (const TransferMatrixQ<Symmetry,Scalar1> &H, MpoTransferVector<Symmetry,Scalar2> &Vinout)
{
	MpoTransferVector<Symmetry,Scalar2> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

template<typename Symmetry, typename Scalar>
inline size_t dim (const TransferMatrixQ<Symmetry,Scalar> &H)
{
	return 0;
}

#endif

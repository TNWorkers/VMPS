#ifndef VANILLA_VUMPS_TRANSFERMATRIX_STRUCTUREFACTOR
#define VANILLA_VUMPS_TRANSFERMATRIX_STRUCTUREFACTOR

#include "VUMPS/VumpsMpoTransferMatrix.h"

template<typename Symmetry, typename Scalar>
struct TransferMatrixSF
{
	TransferMatrixSF(){};
	
	TransferMatrixSF (VMPS::DIRECTION::OPTION DIR_input, 
	                  const vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > &Abra_input, 
	                  const vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > &Aket_input, 
	                  const Biped<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > &Leigen_input, 
	                  const Biped<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > &Reigen_input, 
	                  const vector<vector<qarray<Symmetry::Nq> > > &qloc_input,
	                  double k_input,
					  const typename Symmetry::qType& Q = Symmetry::qvacuum())
	:DIR(DIR_input), Abra(Abra_input), Aket(Aket_input), 
	 qloc(qloc_input), k(k_input)
	{
		Id = Mpo<Symmetry,Scalar>::Identity(qloc, Q);
		Leigen = Tripod<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> >(Leigen_input);
		Reigen = Tripod<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> >(Reigen_input);
	}
	
	VMPS::DIRECTION::OPTION DIR;
	
	double k;
	
	Mpo<Symmetry,Scalar> Id;
	
	///\{
	vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > Abra;
	vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > Aket;
	///\}
	
	Tripod<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > Leigen;
	Tripod<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > Reigen;
	
	vector<vector<qarray<Symmetry::Nq> > > qloc;
};

template<typename Symmetry, typename Scalar>
void HxV (const TransferMatrixSF<Symmetry,Scalar> &H, const MpoTransferVector<Symmetry,complex<Scalar> > &Vin, MpoTransferVector<Symmetry,complex<Scalar> > &Vout)
{
	Vout.data.clear();
	size_t Lcell = H.qloc.size();
	
	Tripod<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > TxV;
	
	if (H.DIR == VMPS::DIRECTION::RIGHT)
	{
		// Calculate T*|Vin>
		Tripod<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > Rnext;
		Tripod<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > R = Vin.data;
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
		Tripod<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > Lnext;
		Tripod<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > L = Vin.data;
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
		assert(1==0 and "Unknown VMPS::DIRECTION::OPTION in TransferMatrixSF!");
	}
	
	// result must be:
	// RIGHT: [1-exp(+i*k)*(T-|R><L|)] * |Vin>
	// LEFT : <Vin| * [1-exp(-i*k)*(T-|R><L|)]
	
	Vout = Vin; // multiply 1
	
	if (H.DIR == VMPS::DIRECTION::RIGHT)
	{
		// subtract exp(+i*k)*T*|Vin>
		Vout.data.addScale(-exp(+1.i*H.k), TxV.template cast<Matrix<complex<Scalar>,Dynamic,Dynamic> >());
		// add <L|Vin>*exp(+i*k)*|R>
		complex<Scalar> LdotV = contract_LR(H.Leigen, Vin.data);
		Vout.data.addScale(+exp(+1.i*H.k)*LdotV, H.Reigen);
	}
	else if (H.DIR == VMPS::DIRECTION::LEFT)
	{
		// subtract exp(-i*k)*<Vin|*T
		Vout.data.addScale(-exp(-1.i*H.k), TxV.template cast<Matrix<complex<Scalar>,Dynamic,Dynamic> >());
		// add <Vin|R>*exp(-i*k)*<L|
		complex<Scalar> VdotR = contract_LR(Vin.data, H.Reigen);
		Vout.data.addScale(+exp(-1.i*H.k)*VdotR, H.Leigen);
	}
}

template<typename Symmetry, typename Scalar1, typename Scalar2>
void HxV (const TransferMatrixSF<Symmetry,Scalar1> &H, MpoTransferVector<Symmetry,Scalar2> &Vinout)
{
	MpoTransferVector<Symmetry,Scalar2> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

template<typename Symmetry, typename Scalar>
inline size_t dim (const TransferMatrixSF<Symmetry,Scalar> &H)
{
	return 0;
}

#endif

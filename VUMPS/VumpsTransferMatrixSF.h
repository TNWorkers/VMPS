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
	                  double k_input)
	:DIR(DIR_input), Abra(Abra_input), Aket(Aket_input), 
	 Leigen(Leigen_input), Reigen(Leigen_input),
	 qloc(qloc_input), k(k_input)
	{
		Id = Mpo<Symmetry,Scalar>::Identity(qloc);
	}
	
	VMPS::DIRECTION::OPTION DIR;
	
	double k;
	
	Mpo<Symmetry,Scalar> Id;
	
	///\{
	vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > Abra;
	vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > Aket;
	///\}
	
	Biped<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > Leigen;
	Biped<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > Reigen;
	
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
		Tripod<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > Rnext;
		Tripod<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > R = Vin.data;
		for (int l=Lcell-1; l>=0; --l)
		{
			contract_R(R, H.Abra[l], H.Id.W_at(l), H.Id.IS_HAMILTONIAN(), H.Aket[l], H.qloc[l], H.Id.opBasis(l), Rnext);
			R.clear();
			R = Rnext;
			Rnext.clear();
		}
		TxV = R;
	}
	else if (H.DIR == VMPS::DIRECTION::LEFT)
	{
		Tripod<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > Lnext;
		Tripod<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > L = Vin.data;
		for (size_t l=0; l<Lcell; ++l)
		{
			contract_L(L, H.Abra[l], H.Id.W_at(l), H.Id.IS_HAMILTONIAN(), H.Aket[l], H.qloc[l], H.Id.opBasis(l), Lnext);
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
	
	complex<Scalar> LdotR;
	if (H.DIR == VMPS::DIRECTION::RIGHT)
	{
		LdotR = contract_LR(0, H.Leigen, Vin.data);
	}
	else if (H.DIR == VMPS::DIRECTION::LEFT)
	{
		LdotR = contract_LR(0, Vin.data, H.Reigen);
	}
	
	Vout = Vin;
	if (H.DIR == VMPS::DIRECTION::RIGHT)
	{
		Vout.data.addScale(-exp(+1.i*H.k), TxV.template cast<Matrix<complex<Scalar>,Dynamic,Dynamic> >());
		Tripod<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > ReigenTripod(H.Reigen);
		Vout.data.addScale(+exp(+1.i*H.k)*LdotR, ReigenTripod);
	}
	else if (H.DIR == VMPS::DIRECTION::LEFT)
	{
		Vout.data.addScale(-exp(-1.i*H.k), TxV.template cast<Matrix<complex<Scalar>,Dynamic,Dynamic> >());
		Tripod<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > LeigenTripod(H.Leigen);
		Vout.data.addScale(+exp(-1.i*H.k)*LdotR, LeigenTripod);
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

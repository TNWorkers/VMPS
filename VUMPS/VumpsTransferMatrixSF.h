#ifndef VANILLA_VUMPS_TRANSFERMATRIX_STRUCTUREFACTOR
#define VANILLA_VUMPS_TRANSFERMATRIX_STRUCTUREFACTOR

#include "VUMPS/VumpsTransferMatrix.h"

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
	{}
	
	VMPS::DIRECTION::OPTION DIR;
	
	double k;
	
	///\{
	vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > Abra;
	vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > Aket;
	///\}
	
	Biped<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > Leigen;
	Biped<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > Reigen;
	
	vector<vector<qarray<Symmetry::Nq> > > qloc;
};

template<typename Symmetry, typename Scalar>
void HxV (const TransferMatrixSF<Symmetry,Scalar> &H, const TransferVector<Symmetry,complex<Scalar> > &Vin, TransferVector<Symmetry,complex<Scalar> > &Vout)
{
	Vout.data.clear();
	size_t Lcell = H.qloc.size();
	
	Biped<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > TxV;
	
	if (H.DIR == VMPS::DIRECTION::RIGHT)
	{
		Biped<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > Rnext;
		Biped<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > R = Vin.data;
		for (int l=Lcell-1; l>=0; --l)
		{
			contract_R(R, H.Abra[l], H.Aket[l], H.qloc[l], Rnext); // RANDOMIZE=false, CLEAR=false
			R.clear();
			R = Rnext;
			Rnext.clear();
		}
		TxV = R;
	}
	else if (H.DIR == VMPS::DIRECTION::LEFT)
	{
		Biped<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > Lnext;
		Biped<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > L = Vin.data;
		for (size_t l=0; l<Lcell; ++l)
		{
			contract_L(L, H.Abra[l], H.Aket[l], H.qloc[l], Lnext); // RANDOMIZE=false, CLEAR=false
			L.clear();
			L = Lnext;
			Lnext.clear();
		}
		TxV = L;
	}
	else
	{
		assert(1!=0 and "Unknown VMPS::DIRECTION::OPTION in TransferMatrixSF!");
	}
	
	complex<Scalar> LdotR;
	if (H.DIR == VMPS::DIRECTION::RIGHT)
	{
		LdotR = H.Leigen.contract(Vin.data).trace();
//		LdotR = (H.Leigen * Vin.data).trace();
	}
	else if (H.DIR == VMPS::DIRECTION::LEFT)
	{
		LdotR = Vin.data.contract(H.Reigen).trace();
//		LdotR = (Vin.data * H.Reigen).trace();
	}
	
	Vout = Vin;
	if (H.DIR == VMPS::DIRECTION::RIGHT)
	{
		Vout.data.addScale(-exp(+1.i*H.k), TxV);
		Vout.data.addScale(+exp(+1.i*H.k)*LdotR, H.Reigen);
	}
	else if (H.DIR == VMPS::DIRECTION::LEFT)
	{
		Vout.data.addScale(-exp(-1.i*H.k), TxV);
		Vout.data.addScale(+exp(-1.i*H.k)*LdotR, H.Leigen);
	}
}

template<typename Symmetry, typename Scalar1, typename Scalar2>
void HxV (const TransferMatrixSF<Symmetry,Scalar1> &H, TransferVector<Symmetry,Scalar2> &Vinout)
{
	TransferVector<Symmetry,Scalar2> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

template<typename Symmetry, typename Scalar>
inline size_t dim (const TransferMatrixSF<Symmetry,Scalar> &H)
{
	return 0;
}

#endif

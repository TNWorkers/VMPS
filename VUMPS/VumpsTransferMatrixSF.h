#ifndef VANILLA_VUMPSTRANSFERMATRIX_STRUCTUREFACTOR
#define VANILLA_VUMPSTRANSFERMATRIX_STRUCTUREFACTOR

/// \cond
#include "termcolor.hpp"
/// \endcond

#include "VUMPS/VumpsTypedefs.h"
#include "pivot/DmrgPivotVector.h"

/**
Operators \f$1-T_L+|R)(1|\f$, \f$1-T_R+|1)(R|\f$ for solving eq. C25ab.
\ingroup VUMPS
*/
template<typename Symmetry, typename Scalar>
struct TransferMatrixSF
{
	TransferMatrixSF(){};
	
	TransferMatrixSF (GAUGE::OPTION ketGauge_input, 
	                  const vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > &Abra_input, 
	                  const vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > &Aket_input, 
	                  const Biped<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > &Leigen_input, 
	                  const Biped<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > &Reigen_input, 
	                  const vector<vector<qarray<Symmetry::Nq> > > &qloc_input,
	                  const vector<vector<qarray<Symmetry::Nq> > > &qOp_input,
	                  double k_input)
	:ketGauge(ketGauge_input), Abra(Abra_input), Aket(Aket_input), 
	 Leigen(Leigen_input), Reigen(Leigen_input),
	 qloc(qloc_input), qOp(qOp_input), k(k_input)
	{}
	
	/**Gauge (L or R).*/
	GAUGE::OPTION ketGauge;
	
	double k;
	
	///\{
	vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > Abra;
	vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > Aket;
	///\}
	
	Biped<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > Leigen;
	Biped<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > Reigen;
	
	vector<vector<qarray<Symmetry::Nq> > > qloc;
	vector<vector<qarray<Symmetry::Nq> > > qOp;
};

/**Matrix-vector multiplication in eq. (25ab)
\ingroup VUMPS*/
template<typename Symmetry, typename Scalar>
void HxV (const TransferMatrixSF<Symmetry,Scalar> &H, const PivotVector<Symmetry,complex<Scalar> > &Vin, PivotVector<Symmetry,complex<Scalar> > &Vout)
{
//	Vout.data.setZero();
	size_t Lcell = H.qloc.size();
	
	Biped<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > TxV;
	
	if (H.ketGauge == GAUGE::L)
	{
		Biped<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > Rnext;
		Biped<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > R = Vin.data[0];
		for (int l=Lcell-1; l>=0; --l)
		{
			contract_R(R, H.Abra[l], H.Aket[l], H.qloc[l], Rnext, false, false); // RANDOMIZE=false, CLEAR=false
			R.clear();
			R = Rnext;
			Rnext.clear();
		}
		TxV = R;
	}
	else if (H.ketGauge == GAUGE::R)
	{
		Biped<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > Lnext;
		Biped<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > L = Vin.data[0];
		for (size_t l=0; l<Lcell; ++l)
		{
			contract_L(L, H.Abra[l], H.Aket[l], H.qloc[l], Lnext, false, false); // RANDOMIZE=false, CLEAR=false
			L.clear();
			L = Lnext;
			Lnext.clear();
		}
		TxV = L;
	}
	else
	{
		throw;
	}
	
	complex<Scalar> LdotR;
	if (H.ketGauge == GAUGE::L)
	{
		LdotR = H.Leigen.contract(Vin.data[0]).trace();
	}
	else if (H.ketGauge == GAUGE::R)
	{
		LdotR = Vin.data[0].contract(H.Reigen).trace();
	}
	
	Vout = Vin;
	if (H.ketGauge == GAUGE::L)
	{
		Vout.data[0].addScale(-exp(+1.i*H.k), TxV);
		Vout.data[0].addScale(+exp(+1.i*H.k)*LdotR, H.Reigen);
	}
	else if (H.ketGauge == GAUGE::R)
	{
		Vout.data[0].addScale(-exp(-1.i*H.k), TxV);
		Vout.data[0].addScale(+exp(-1.i*H.k)*LdotR, H.Leigen);
	}
}

template<typename Symmetry, typename Scalar1, typename Scalar2>
void HxV (const TransferMatrixSF<Symmetry,Scalar1> &H, PivotVector<Symmetry,Scalar2> &Vinout)
{
	PivotVector<Symmetry,Scalar2> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

template<typename Symmetry, typename Scalar>
inline size_t dim (const TransferMatrixSF<Symmetry,Scalar> &H)
{
	return 0;
}

#endif

#ifndef VANILLA_VUMPSTRANSFERMATRIXAA
#define VANILLA_VUMPSTRANSFERMATRIXAA

template<typename Symmetry, typename Scalar>
struct TransferMatrixAA
{
	TransferMatrixAA(){};
	
	TransferMatrixAA (GAUGE::OPTION gauge_input, 
	                const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Abra_input, 
	                const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Aket_input, 
	                const vector<qarray<Symmetry::Nq> > &qloc_input)
	:gauge(gauge_input), Abra(Abra_input), Aket(Aket_input), qloc(qloc_input)
	{}
	
	GAUGE::OPTION gauge;
	
	///\{
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > Aket;
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > Abra;
	///\}
	
	vector<qarray<Symmetry::Nq> > qloc;
};

template<typename Symmetry, typename Scalar>
inline size_t dim (const TransferMatrixAA<Symmetry,Scalar> &H)
{
	return 0;
}

template<typename Symmetry, typename Scalar1, typename Scalar2>
void HxV (const TransferMatrixAA<Symmetry,Scalar1> &H, const PivotVector<Symmetry,Scalar2> &Vin, PivotVector<Symmetry,Scalar2> &Vout)
{
	Vout.outerResize(Vin);
	
	if (H.gauge == GAUGE::L)
	{
		contract_R (Vin.data[0], H.Abra, H.Aket, H.qloc, Vout.data[0]);
	}
	else if (H.gauge == GAUGE::R)
	{
		contract_L (Vin.data[0], H.Abra, H.Aket, H.qloc, Vout.data[0]);
	}
}

template<typename Symmetry, typename Scalar1, typename Scalar2>
void HxV (const TransferMatrixAA<Symmetry,Scalar1> &H, PivotVector<Symmetry,Scalar2> &Vinout)
{
	PivotVector<Symmetry,Scalar2> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

#endif

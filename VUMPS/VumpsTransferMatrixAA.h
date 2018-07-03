#ifndef VANILLA_VUMPSTRANSFERMATRIXAA
#define VANILLA_VUMPSTRANSFERMATRIXAA

enum TM_MULT_MODE {UNSHIFTED, SHIFTED};

template<typename Symmetry, typename Scalar>
struct TransferMatrixAA
{
	TransferMatrixAA(){};
	
	TransferMatrixAA (GAUGE::OPTION gauge_input, 
	                const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Abra_input, 
	                const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Aket_input, 
	                const vector<qarray<Symmetry::Nq> > &qloc_input,
	                TM_MULT_MODE MODE_input = UNSHIFTED)
	:gauge(gauge_input), Abra(Abra_input), Aket(Aket_input), qloc(qloc_input), CURRENT_MODE(MODE_input)
	{}
	
	GAUGE::OPTION gauge;
	
	///\{
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > Aket;
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > Abra;
	///\}
	
	Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > LReigen;
	
	TM_MULT_MODE CURRENT_MODE = UNSHIFTED;
	
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
	
	if (H.CURRENT_MODE == UNSHIFTED)
	{
		if (H.gauge == GAUGE::L)
		{
			contract_R (Vin.data[0], H.Abra, H.Aket, H.qloc, Vout.data[0]);
		}
		else if (H.gauge == GAUGE::R)
		{
			contract_L (Vin.data[0], H.Abra, H.Aket, H.qloc, Vout.data[0]);
		}
	}
	else if (H.CURRENT_MODE == SHIFTED)
	{
		Vout = Vin;
		Vout.setZero();
		
		PivotVector<Symmetry,Scalar2> TxV = Vin;
		TxV.setZero();
		
		if (H.gauge == GAUGE::R)
		{
			contract_R (Vin.data[0], H.Abra, H.Aket, H.qloc, TxV.data[0]);
		}
		else if (H.gauge == GAUGE::L)
		{
			contract_L (Vin.data[0], H.Abra, H.Aket, H.qloc, TxV.data[0]);
		}
		
		Scalar2 LdotR;
		if (H.gauge == GAUGE::R)
		{
			LdotR = (H.LReigen.contract(Vin.data[0])).trace();
		}
		else if (H.gauge == GAUGE::L)
		{
			LdotR = (Vin.data[0].contract(H.LReigen)).trace();
		}
		
		for (size_t q=0; q<TxV.data[0].dim; ++q)
		{
			qarray2<Symmetry::Nq> quple = {TxV.data[0].in[q], TxV.data[0].out[q]};
			auto it = Vin.data[0].dict.find(quple);
			
			Matrix<Scalar2,Dynamic,Dynamic> Mtmp;
			if (it != Vin.data[0].dict.end())
			{
				Mtmp = Vin.data[0].block[it->second] - TxV.data[0].block[q];
				
				if (quple[2] == Symmetry::qvacuum())
				{
					Mtmp += LdotR * Matrix<Scalar2,Dynamic,Dynamic>::Identity(Vin.data[0].block[it->second].rows(),
					                                                          Vin.data[0].block[it->second].cols());
				}
			}
			
			if (Mtmp.size() != 0)
			{
				auto ip = Vout.data[0].dict.find(quple);
				if (ip != Vout.data[0].dict.end())
				{
					if (Vout.data[0].block[ip->second].rows() != Mtmp.rows() or 
						Vout.data[0].block[ip->second].cols() != Mtmp.cols())
					{
						Vout.data[0].block[ip->second] = Mtmp;
					}
					else
					{
						Vout.data[0].block[ip->second] += Mtmp;
					}
				}
				else
				{
					cout << termcolor::red << "push_back that shouldn't be" << termcolor::reset << endl;
					Vout.data[0].push_back(quple, Mtmp);
				}
			}
		}
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

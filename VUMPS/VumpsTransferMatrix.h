#ifndef VANILLA_VUMPSTRANSFERMATRIX
#define VANILLA_VUMPSTRANSFERMATRIX

template<typename Symmetry, typename Scalar>
struct TransferMatrix
{
	TransferMatrix(){};
	
	TransferMatrix (GAUGE::OPTION gauge_input, 
	                const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Abra_input, 
	                const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Aket_input, 
	                const Matrix<Scalar,Dynamic,Dynamic> &LReigen_input, 
	                vector<Scalar> Wvec_input,
	                vector<size_t> D_input)
	:Abra(Abra_input), Aket(Aket_input), gauge(gauge_input), LReigen(LReigen_input), Wvec(Wvec_input), D(D_input)
	{
		assert(Aket.size() == Abra.size());
		
		if (Wvec.size() == 0)
		{
			Wvec.resize(Aket.size());
			for (size_t s=0; s<Aket.size(); ++s)
			{
				Wvec[s] = 1.;
			}
		}
	}
	
	TransferMatrix (GAUGE::OPTION gauge_input, 
	                const vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > &ApairBra_input, 
	                const vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > &ApairKet_input, 
	                const Matrix<Scalar,Dynamic,Dynamic> &LReigen_input, 
	                boost::multi_array<double,4> Warray_input,
	                vector<size_t> D_input)
	:ApairBra(ApairBra_input), ApairKet(ApairKet_input), gauge(gauge_input), LReigen(LReigen_input), D(D_input)
	{
		assert(ApairKet.size() == ApairBra.size());
		assert(D_input.size() == 2);
		
		Warray.resize(boost::extents[D[0]][D[0]][D[1]][D[1]]);
		Warray = Warray_input;
	}
	
	TransferMatrix (GAUGE::OPTION gauge_input, 
	                const boost::multi_array<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> >,4> &AquartettBra_input, 
	                const boost::multi_array<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> >,4> &AquartettKet_input, 
	                const Matrix<Scalar,Dynamic,Dynamic> &LReigen_input, 
	                boost::multi_array<double,8> Warray4_input,
	                vector<size_t> D_input)
	:AquartettBra(AquartettBra_input), AquartettKet(AquartettKet_input), gauge(gauge_input), LReigen(LReigen_input), D(D_input)
	{
		assert(ApairKet.size() == ApairBra.size());
		assert(D_input.size() == 4);
		Warray4.resize(boost::extents[D[0]][D[0]][D[1]][D[1]][D[2]][D[2]][D[3]][D[3]]);
		Warray4 = Warray4_input;
	}
	
	GAUGE::OPTION gauge;
	vector<size_t> D;
	
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > Aket;
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > Abra;
	vector<Scalar> Wvec;
	
	vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > ApairKet;
	vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > ApairBra;
	boost::multi_array<double,4> Warray;
	
	boost::multi_array<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> >,4> AquartettKet;
	boost::multi_array<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> >,4> AquartettBra;
	boost::multi_array<double,8> Warray4;
	
	Matrix<Scalar,Dynamic,Dynamic> LReigen;
};

template<typename Scalar>
inline void setZero (Matrix<Scalar,Dynamic,Dynamic> &M)
{
	M.setZero();
}

// Note:
// if H.LReigen.rows()==0, only T is used
// if H.LReigen.rows()!=0, 1-T+|1><LReigen| is used
template<typename Symmetry, typename Scalar1, typename Scalar2>
void HxV (const TransferMatrix<Symmetry,Scalar1> &H, const Matrix<Scalar2,Dynamic,Dynamic> &Vin, Matrix<Scalar2,Dynamic,Dynamic> &Vout)
{
	Vout = Vin;
	
	if (H.LReigen.rows() == 0)
	{
		setZero(Vout);
	}
	
	double factor = (H.LReigen.rows()==0)? +1.:-1.;
	
	if (H.Aket.size() != 0)
	{
		if (H.gauge == GAUGE::R)
		{
			for (size_t s=0; s<H.Aket.size(); ++s)
			{
				Vout += factor * H.Wvec[s] * H.Aket[s].block[0] * Vin * H.Abra[s].block[0].adjoint();
			}
		}
		else if (H.gauge == GAUGE::L)
		{
			for (size_t s=0; s<H.Aket.size(); ++s)
			{
				Vout += factor * H.Wvec[s] * H.Abra[s].block[0].adjoint() * Vin * H.Aket[s].block[0];
			}
		}
	}
	else if (H.ApairKet.size() != 0)
	{
		if (H.gauge == GAUGE::R)
		{
			for (size_t s1=0; s1<H.D[0]; ++s1)
			for (size_t s2=0; s2<H.D[0]; ++s2)
			for (size_t s3=0; s3<H.D[1]; ++s3)
			for (size_t s4=0; s4<H.D[1]; ++s4)
			{
				if (H.Warray[s1][s2][s3][s4] != 0.)
				{
					Vout += factor * H.Warray[s1][s2][s3][s4] * H.ApairKet[s2][s4].block[0] * Vin * H.ApairBra[s1][s3].block[0].adjoint();
				}
			}
		}
		else if (H.gauge == GAUGE::L)
		{
			for (size_t s1=0; s1<H.D[0]; ++s1)
			for (size_t s2=0; s2<H.D[0]; ++s2)
			for (size_t s3=0; s3<H.D[1]; ++s3)
			for (size_t s4=0; s4<H.D[1]; ++s4)
			{
				if (H.Warray[s1][s2][s3][s4] != 0.)
				{
					Vout += factor * H.Warray[s1][s2][s3][s4] * H.ApairBra[s1][s3].block[0].adjoint() * Vin * H.ApairKet[s2][s4].block[0];
				}
			}
		}
	}
	else if (H.AquartettKet.size() != 0)
	{
		if (H.gauge == GAUGE::R)
		{
			for (size_t s1=0; s1<H.D[0]; ++s1)
			for (size_t s2=0; s2<H.D[0]; ++s2)
			for (size_t s3=0; s3<H.D[1]; ++s3)
			for (size_t s4=0; s4<H.D[1]; ++s4)
			for (size_t s5=0; s5<H.D[2]; ++s5)
			for (size_t s6=0; s6<H.D[2]; ++s6)
			for (size_t s7=0; s7<H.D[3]; ++s7)
			for (size_t s8=0; s8<H.D[3]; ++s8)
			{
				if (H.Warray4[s1][s2][s3][s4][s5][s6][s7][s8] != 0.)
				{
					Vout += factor * H.Warray4[s1][s2][s3][s4][s5][s6][s7][s8] * 
					        H.AquartettKet[s2][s4][s6][s8].block[0] * Vin * H.AquartettBra[s1][s3][s5][s7].block[0].adjoint();
				}
			}
		}
		else if (H.gauge == GAUGE::L)
		{
			for (size_t s1=0; s1<H.D[0]; ++s1)
			for (size_t s2=0; s2<H.D[0]; ++s2)
			for (size_t s3=0; s3<H.D[1]; ++s3)
			for (size_t s4=0; s4<H.D[1]; ++s4)
			for (size_t s5=0; s5<H.D[2]; ++s5)
			for (size_t s6=0; s6<H.D[2]; ++s6)
			for (size_t s7=0; s7<H.D[3]; ++s7)
			for (size_t s8=0; s8<H.D[3]; ++s8)
			{
				if (H.Warray4[s1][s2][s3][s4][s5][s6][s7][s8] != 0.)
				{
					Vout += factor * H.Warray4[s1][s2][s3][s4][s5][s6][s7][s8] * 
					        H.AquartettBra[s1][s3][s5][s7].block[0].adjoint() * Vin * H.AquartettKet[s2][s4][s6][s8].block[0];
				}
			}
		}
	}
	
	if (H.LReigen.rows() != 0)
	{
		Vout += (H.LReigen * Vin).trace() * Matrix<Scalar2,Dynamic,Dynamic>::Identity(Vin.rows(),Vin.cols());
	}
}

template<typename Symmetry, typename Scalar1, typename Scalar2>
void HxV (const TransferMatrix<Symmetry,Scalar1> &H, Matrix<Scalar2,Dynamic,Dynamic> &Vinout)
{
	Matrix<Scalar2,Dynamic,Dynamic> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

template<typename Symmetry, typename Scalar>
inline size_t dim (const TransferMatrix<Symmetry,Scalar> &H)
{
	if (H.Aket.size() != 0)
	{
		return H.Aket[0].block[0].cols() * H.Abra[0].block[0].rows();
	}
	else if (H.ApairKet.size() != 0)
	{
		return H.ApairKet[0][0].block[0].cols() * H.ApairBra[0][0].block[0].rows();
	}
	else if (H.AquartettKet.size() != 0)
	{
		return H.AquartettKet[0][0][0][0].block[0].cols() * H.AquartettBra[0][0][0][0].block[0].rows();
	}
}

template<typename Scalar>
inline Scalar squaredNorm (const Matrix<Scalar,Dynamic,Dynamic> &V)
{
	return V.squaredNorm();
}

template<typename Scalar>
inline Scalar norm (const Matrix<Scalar,Dynamic,Dynamic> &V)
{
	return V.norm();
}

template<typename Scalar>
inline void normalize (Matrix<Scalar,Dynamic,Dynamic> &V)
{
	V /= norm(V);
}

template<typename Scalar>
inline Scalar infNorm (const Matrix<Scalar,Dynamic,Dynamic> &V1, const Matrix<Scalar,Dynamic,Dynamic> &V2)
{
	return (V1-V2).template lpNorm<Eigen::Infinity>();
}

template<typename Scalar>
void swap (Matrix<Scalar,Dynamic,Dynamic> &V1, Matrix<Scalar,Dynamic,Dynamic> &V2)
{
	V1.swap(V2);
}

template<typename Scalar>
inline Scalar dot (const Matrix<Scalar,Dynamic,Dynamic> &V1, const Matrix<Scalar,Dynamic,Dynamic> &V2)
{
	return (V1.adjoint() * V2).trace();
}

#include "RandomVector.h"

template<typename Scalar>
struct GaussianRandomVector<Matrix<Scalar,Dynamic,Dynamic>,Scalar>
{
	static void fill (size_t N, Matrix<Scalar,Dynamic,Dynamic> &Vout)
	{
		for (size_t i=0; i<Vout.rows(); ++i)
		for (size_t j=0; j<Vout.cols(); ++j)
		{
			Vout(i,j) = threadSafeRandUniform<Scalar>(-1.,1.);
		}
		normalize(Vout);
	}
};

#endif

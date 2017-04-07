#ifndef VANILLA_VUMPSTRANSFERMATRIX
#define VANILLA_VUMPSTRANSFERMATRIX

template<size_t Nq, typename Scalar>
struct TransferMatrix
{
	TransferMatrix(){};
	
	TransferMatrix (GAUGE::OPTION gauge_input, 
	                const vector<Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> > > &Abra_input, 
	                const vector<Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> > > &Aket_input, 
	                const Matrix<Scalar,Dynamic,Dynamic> &LReigen_input, 
	                vector<Scalar> Wvec_input)
	:Abra(Abra_input), Aket(Aket_input), gauge(gauge_input), LReigen(LReigen_input), Wvec(Wvec_input)
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
	                const vector<vector<Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> > > > &ApairBra_input, 
	                const vector<vector<Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> > > > &ApairKet_input, 
	                const Matrix<Scalar,Dynamic,Dynamic> &LReigen_input, 
	                boost::multi_array<double,4> Warray_input,
	                std::array<size_t,2> D_input)
	:ApairBra(ApairBra_input), ApairKet(ApairKet_input), gauge(gauge_input), LReigen(LReigen_input), D(D_input)
	{
		assert(ApairKet.size() == ApairBra.size());
		Warray.resize(boost::extents[D[0]][D[0]][D[1]][D[1]]);
		Warray = Warray_input;
	}
	
	GAUGE::OPTION gauge;
	
	vector<Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> > > Aket;
	vector<Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> > > Abra;
	vector<Scalar> Wvec;
	
	vector<vector<Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> > > > ApairKet;
	vector<vector<Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> > > > ApairBra;
	boost::multi_array<double,4> Warray;
	std::array<size_t,2> D;
	
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
template<size_t Nq, typename Scalar1, typename Scalar2>
void HxV (const TransferMatrix<Nq,Scalar1> &H, const Matrix<Scalar2,Dynamic,Dynamic> &Vin, Matrix<Scalar2,Dynamic,Dynamic> &Vout)
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
	
	if (H.LReigen.rows() != 0)
	{
		Vout += (H.LReigen * Vin).trace() * Matrix<Scalar2,Dynamic,Dynamic>::Identity(Vin.rows(),Vin.cols());
	}
}

template<size_t Nq, typename Scalar1, typename Scalar2>
void HxV (const TransferMatrix<Nq,Scalar1> &H, Matrix<Scalar2,Dynamic,Dynamic> &Vinout)
{
	Matrix<Scalar2,Dynamic,Dynamic> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

template<size_t Nq, typename Scalar>
inline size_t dim (const TransferMatrix<Nq,Scalar> &H)
{
	if (H.Aket.size() != 0)
	{
		return H.Aket[0].block[0].cols() * H.Abra[0].block[0].rows();
	}
	else if (H.ApairKet.size() != 0)
	{
		return H.ApairKet[0][0].block[0].cols() * H.ApairBra[0][0].block[0].rows();
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

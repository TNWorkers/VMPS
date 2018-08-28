#ifndef DMRGPIVOTVECTOR
#define DMRGPIVOTVECTOR


#include "tensors/DmrgContractions.h" // for contract_AA
//include "numeric_limits.h"

template<typename Symmetry, typename Scalar>
struct PivotVector
{
	static constexpr std::size_t Nq = Symmetry::Nq;
	
	PivotVector()
	{
		data.resize(1); // needs to be set for 0-site
	};
	
	/**Set from a center matrix.*/
	PivotVector (const Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &C)
	{
		data.resize(1);
		data[0] = C;
	}
	
	/**Set from one A-tensor.*/
	PivotVector (const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &A12)
	:data(A12)
	{}
	
	/**Make contraction of two A-tensors.*/
	PivotVector (const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &A12,
	             const vector<qarray<Symmetry::Nq> > &qloc12,
	             const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &A34,
	             const vector<qarray<Symmetry::Nq> > &qloc34,
	             const qarray<Symmetry::Nq> &Qtop, 
                 const qarray<Symmetry::Nq> &Qbot, 
                 bool DRY = false)
	{
		contract_AA(A12, qloc12, A34, qloc34, Qtop, Qbot, data, DRY);
	}
	
	/**Set blocks as in Vrhs, but do not resize the matrices*/
	void outerResize (const PivotVector &Vrhs)
	{
		data.clear();
		data.resize(Vrhs.data.size());
		for (size_t i=0; i<data.size(); ++i)
		{
			data[i].in = Vrhs.data[i].in;
			data[i].out = Vrhs.data[i].out;
			data[i].dict = Vrhs.data[i].dict;
			data[i].block.resize(Vrhs.data[i].block.size());
			data[i].dim = Vrhs.data[i].dim;
		}
	}
	
	void setZero()
	{
		for (size_t i=0; i<data.size(); ++i)
		for (size_t q=0; q<data[i].dim; ++q)
		{
			data[i].block[q].setZero();
		}
	}
	
	inline size_t size() const {return data.size();}
	
	void print_dims() const
	{
		map<qarray<Symmetry::Nq>,set<int> > indim;
		map<qarray<Symmetry::Nq>,set<int> > outdim;
		for (size_t s1s3=0; s1s3<data.size(); ++s1s3)
		for (size_t q=0; q<data[s1s3].dim; ++q)
		{
			indim[data[s1s3].in[q]].insert(data[s1s3].block[q].rows());
			outdim[data[s1s3].out[q]].insert(data[s1s3].block[q].cols());
		}
		for (auto it=indim.begin(); it!=indim.end(); ++it)
		{
			cout << "in=" << it->first << ":";
			for (auto ip=it->second.begin(); ip!=it->second.end(); ++ip)
			{
				cout << *ip << " ";
			}
			cout << endl;
		}
		for (auto it=outdim.begin(); it!=outdim.end(); ++it)
		{
			cout << "out=" << it->first << ":";
			for (auto ip=it->second.begin(); ip!=it->second.end(); ++ip)
			{
				cout << *ip << " ";
			}
			cout << endl;
		}
	}
	
	Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &operator[] (size_t i) {return data[i];}
	Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &operator() (size_t i) {return data[i];}
	const Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &operator[] (size_t i) const {return data[i];}
	const Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &operator() (size_t i) const {return data[i];}
	
	PivotVector<Symmetry,Scalar>& operator+= (const PivotVector<Symmetry,Scalar> &Vrhs);
	PivotVector<Symmetry,Scalar>& operator-= (const PivotVector<Symmetry,Scalar> &Vrhs);
	template<typename OtherScalar> PivotVector<Symmetry,Scalar>& operator*= (const OtherScalar &alpha);
	template<typename OtherScalar> PivotVector<Symmetry,Scalar>& operator/= (const OtherScalar &alpha);
	
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > data;
};
//-----------</definitions>-----------

//-----------<vector arithmetics>-----------
template<typename Symmetry, typename Scalar>
PivotVector<Symmetry,Scalar>& PivotVector<Symmetry,Scalar>::operator+= (const PivotVector<Symmetry,Scalar> &Vrhs)
{
//	for (std::size_t s=0; s<data.size(); s++)
//	{
////		for (size_t q=0; q<data[s].dim; ++q)
////		{
////			cout << "+= 1=" << data[s].in[q] << ", " << data[s].out[q] << endl;
////			cout << "+= 1=" << Vrhs.data[s].in[q] << ", " << Vrhs.data[s].out[q] << endl;
////			print_size(data[s].block[q],"data[s].block[q]");
////			print_size(Vrhs.data[s].block[q],"Vrhs.data[s].block[q]");
////		}
////		cout << "dims(s)=" << data[s].dim << ", " << Vrhs.data[s].dim << endl;
//		data[s] = data[s] + Vrhs.data[s];
//	}
//	return *this;
	
	for (size_t s=0; s<data.size(); s++)
	for (size_t q=0; q<data[s].dim; ++q)
	{
		if (data[s].block[q].size() == 0)
		{
			data[s].block[q] = Vrhs.data[s].block[q];
		}
		else if (data[s].block[q].size() != 0 and Vrhs.data[s].block[q].size() != 0)
		{
			data[s].block[q] += Vrhs.data[s].block[q];
		}
	}
	return *this;
}

template<typename Symmetry, typename Scalar>
PivotVector<Symmetry,Scalar>& PivotVector<Symmetry,Scalar>::
operator-= (const PivotVector<Symmetry,Scalar> &Vrhs)
{
//	for (std::size_t s=0; s<data.size(); s++)
//	{
////		for (size_t q=0; q<data[s].dim; ++q)
////		{
////			cout << "-= 1=" << data[s].in[q] << ", " << data[s].out[q] << endl;
////			cout << "-= 2=" << Vrhs.data[s].in[q] << ", " << Vrhs.data[s].out[q] << endl;
////			print_size(data[s].block[q],"data[s].block[q]");
////			print_size(Vrhs.data[s].block[q],"Vrhs.data[s].block[q]");
////		}
////		cout << "dims(s)=" << data[s].dim << ", " << Vrhs.data[s].dim << endl;
//		data[s] = data[s] - Vrhs.data[s];
//	}
//	return *this;
	
	for (size_t s=0; s<data.size(); s++)
	for (size_t q=0; q<data[s].dim; ++q)
	{
//		cout << "s=" << s << ", q=" << q << endl;
//		cout << "-= 1=" << data[s].in[q] << ", " << data[s].out[q] << endl;
//		cout << "-= 2=" << Vrhs.data[s].in[q] << ", " << Vrhs.data[s].out[q] << endl;
//		print_size(data[s].block[q],"data[s].block[q]");
//		print_size(Vrhs.data[s].block[q],"Vrhs.data[s].block[q]");
		
		if (data[s].block[q].size() == 0)
		{
			data[s].block[q] = -Vrhs.data[s].block[q];
		}
		else if (data[s].block[q].size() != 0 and Vrhs.data[s].block[q].size() != 0)
		{
			data[s].block[q] -= Vrhs.data[s].block[q];
		}
	}
	return *this;
}

template<typename Symmetry, typename Scalar>
template<typename OtherScalar>
PivotVector<Symmetry,Scalar>& PivotVector<Symmetry,Scalar>::
operator*= (const OtherScalar &alpha)
{
	for (size_t s=0; s<data.size(); ++s)
	for (size_t q=0; q<data[s].dim; ++q)
	{
		data[s].block[q] *= alpha;
	}
	return *this;
}

template<typename Symmetry, typename Scalar>
template<typename OtherScalar>
PivotVector<Symmetry,Scalar>& PivotVector<Symmetry,Scalar>::
operator/= (const OtherScalar &alpha)
{
	for (size_t s=0; s<data.size(); ++s)
	for (size_t q=0; q<data[s].dim; ++q)
	{
		data[s].block[q] /= alpha;
	}
	return *this;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
PivotVector<Symmetry,Scalar> operator* (const OtherScalar &alpha, PivotVector<Symmetry,Scalar> V)
{
	return V *= alpha;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
PivotVector<Symmetry,Scalar> operator* (PivotVector<Symmetry,Scalar> V, const OtherScalar &alpha)
{
	return V *= alpha;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
PivotVector<Symmetry,Scalar> operator/ (PivotVector<Symmetry,Scalar> V, const OtherScalar &alpha)
{
	return V /= alpha;
}

template<typename Symmetry, typename Scalar>
PivotVector<Symmetry,Scalar> operator+ (const PivotVector<Symmetry,Scalar> &V1, const PivotVector<Symmetry,Scalar> &V2)
{
	PivotVector<Symmetry,Scalar> Vout = V1;
	Vout += V2;
	return Vout;
}

template<typename Symmetry, typename Scalar>
PivotVector<Symmetry,Scalar> operator- (const PivotVector<Symmetry,Scalar> &V1, const PivotVector<Symmetry,Scalar> &V2)
{
	PivotVector<Symmetry,Scalar> Vout = V1;
	Vout -= V2;
	return Vout;
}

//-----------<dot & vector norms>-----------
template<typename Symmetry, typename Scalar>
Scalar dot (const PivotVector<Symmetry,Scalar> &V1, const PivotVector<Symmetry,Scalar> &V2)
{
//	cout << "sizes in dot: " << V1.data.size() << ", " << V2.data.size() << endl;
	Scalar res = 0;
	for (size_t s=0; s<V2.data.size(); ++s)
	for (size_t q=0; q<V2.data[s].dim; ++q)
	{
//		if (V1.data[s].in[q] != V2.data[s].in[q] or V1.data[s].out[q] != V2.data[s].out[q])
//		{
////			cout << "s=" << s << ", q=" << q << endl;
////			cout << "V1 inout=" << V1.data[s].in[q] << ", " << V1.data[s].out[q] << endl;
////			cout << "V2 inout=" << V2.data[s].in[q] << ", " << V2.data[s].out[q] << endl;
////			print_size(V1.data[s].block[q],"V1.data[s].block[q]");
////			print_size(V2.data[s].block[q],"V2.data[s].block[q]");
////			cout << endl;
//			cout << termcolor::red << "Mismatching blocks in dot(PivotVector)" << termcolor::reset << endl;
//		}
		
		if (V1.data[s].block[q].size() > 0 and 
		    V2.data[s].block[q].size() > 0)
		{
			res += (V1.data[s].block[q].adjoint() * V2.data[s].block[q]).trace() * Symmetry::coeff_dot(V1.data[s].out[q]);
		}
		
//		qarray2<Symmetry::Nq> quple = {V2.data[s].in[q], V2.data[s].out[q]};
//		auto it = V1.data[s].dict.find(quple);
//		res += (V1.data[s].block[it->second].adjoint() * V2.data[s].block[q]).trace() * Symmetry::coeff_dot(V1.data[s].out[it->second]);
	}
	return res;
}

template<typename Symmetry, typename Scalar>
double squaredNorm (const PivotVector<Symmetry,Scalar> &V)
{
	double res = isReal(dot(V,V));
	return res;
}

template<typename Symmetry, typename Scalar>
inline double norm (const PivotVector<Symmetry,Scalar> &V)
{
	return sqrt(squaredNorm(V));
}

template<typename Symmetry, typename Scalar>
inline void normalize (PivotVector<Symmetry,Scalar> &V)
{
	V /= norm(V);
}

template<typename Symmetry, typename Scalar>
inline size_t dim (const PivotVector<Symmetry,Scalar> &V)
{
	size_t out = 0;
	for (size_t s=0; s<V.data.size(); ++s)
	for (size_t q=0; q<V.data[s].dim; ++q)
	{
		out += V.data[s].block[q].size();
	}
	return out;
}

template<typename Symmetry, typename Scalar>
double infNorm (const PivotVector<Symmetry,Scalar> &V1, const PivotVector<Symmetry,Scalar> &V2)
{
	double res = 0.;
	for (size_t s=0; s<V1.data.size(); ++s)
	{
		auto Mtmp = V1.data[s] - V2.data[s];
		for (size_t q=0; q<Mtmp.dim; ++q)
		{
			double tmp = Mtmp.block[q].template lpNorm<Eigen::Infinity>();
			if (tmp>res) {res = tmp;}
		}
	}
	return res;
}

template<typename Symmetry, typename Scalar>
void swap (PivotVector<Symmetry,Scalar> &V1, PivotVector<Symmetry,Scalar> &V2)
{
//	for (size_t s=0; s<V1.data.size(); ++s)
//	{
//		V1.data[s].block.swap(V2.data[s].block);
//	}
	swap(V1.data, V2.data);
}

template<typename Symmetry, typename Scalar>
void setZero (PivotVector<Symmetry,Scalar> &V)
{
	for (size_t s=0; s<V.data.size(); ++s)
	for (size_t q=0; q<V.data[s].dim; ++q)
	{
		V.data[s].block[q].setZero();
	}
};

#include "RandomVector.h"

template<typename Symmetry, typename Scalar>
struct GaussianRandomVector<PivotVector<Symmetry,Scalar>,Scalar>
{
	static void fill (size_t N, PivotVector<Symmetry,Scalar> &Vout)
	{
		for (size_t s=0; s<Vout.data.size(); ++s)
		for (size_t q=0; q<Vout.data[s].dim; ++q)
		for (size_t a1=0; a1<Vout.data[s].block[q].rows(); ++a1)
		for (size_t a2=0; a2<Vout.data[s].block[q].cols(); ++a2)
		{
			Vout.data[s].block[q](a1,a2) = threadSafeRandUniform<Scalar>(-1.,1.);
		}
		normalize(Vout);
	}
};

#endif

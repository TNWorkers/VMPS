#ifndef DMRGPIVOTVECTOR
#define DMRGPIVOTVECTOR

#include "tensors/DmrgContractions.h" // for contract_AA
#include "termcolor.hpp"
//include "numeric_limits.h"

template<typename Symmetry, typename Scalar_>
struct PivotVector
{
	typedef Scalar_ Scalar;
	
	static constexpr std::size_t Nq = Symmetry::Nq;
	
	PivotVector()
	{
		data.resize(1); // needs to be set for 0-site
	};
	
	/**Set from a center matrix.*/
	PivotVector (const Biped<Symmetry,Matrix<Scalar_,Dynamic,Dynamic> > &C)
	{
		data.resize(1);
		data[0] = C;
	}
	
	/**Set from one A-tensor.*/
	PivotVector (const vector<Biped<Symmetry,Matrix<Scalar_,Dynamic,Dynamic> > > &A12)
	:data(A12)
	{}
	
	/**Make contraction of two A-tensors.*/
	PivotVector (const vector<Biped<Symmetry,Matrix<Scalar_,Dynamic,Dynamic> > > &A12,
	             const vector<qarray<Symmetry::Nq> > &qloc12,
	             const vector<Biped<Symmetry,Matrix<Scalar_,Dynamic,Dynamic> > > &A34,
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
	
	Biped<Symmetry,Matrix<Scalar_,Dynamic,Dynamic> > &operator[] (size_t i) {return data[i];}
	Biped<Symmetry,Matrix<Scalar_,Dynamic,Dynamic> > &operator() (size_t i) {return data[i];}
	const Biped<Symmetry,Matrix<Scalar_,Dynamic,Dynamic> > &operator[] (size_t i) const {return data[i];}
	const Biped<Symmetry,Matrix<Scalar_,Dynamic,Dynamic> > &operator() (size_t i) const {return data[i];}
	
	PivotVector<Symmetry,Scalar_>& operator+= (const PivotVector<Symmetry,Scalar_> &Vrhs);
	PivotVector<Symmetry,Scalar_>& operator-= (const PivotVector<Symmetry,Scalar_> &Vrhs);
	template<typename OtherScalar> PivotVector<Symmetry,Scalar_>& operator*= (const OtherScalar &alpha);
	template<typename OtherScalar> PivotVector<Symmetry,Scalar_>& operator/= (const OtherScalar &alpha);
	
	vector<Biped<Symmetry,Matrix<Scalar_,Dynamic,Dynamic> > > data;
};
//-----------</definitions>-----------

//-----------<vector arithmetics>-----------
template<typename Symmetry, typename Scalar_>
PivotVector<Symmetry,Scalar_>& PivotVector<Symmetry,Scalar_>::operator+= (const PivotVector<Symmetry,Scalar_> &Vrhs)
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
	
	#ifdef DMRG_PIVOTVECTOR_PARALLELIZE
	#pragma omp parallel for schedule(dynamic)
	#endif
	for (size_t s=0; s<data.size(); s++)
	for (size_t q=0; q<data[s].dim; ++q)
	{
		
//		qarray2<Symmetry::Nq> quple = {data[s].in[q], data[s].out[q]};
//		auto it = Vrhs.data[s].dict.find(quple);
//		if (it != Vrhs.data[s].dict.end())
		{
//			cout << "+=" << endl;
//			print_size(data[s].block[q],"data[s].block[q]");
//			print_size(Vrhs.data[s].block[it->second],"Vrhs.data[s].block[it->second]");
//			cout << endl;
			
			if (data[s].block[q].size() == 0)
			{
				data[s].block[q] = Vrhs.data[s].block[q];
			}
			else if (data[s].block[q].size() != 0 and Vrhs.data[s].block[q].size() != 0)
			{
//				cout << data[s].block[q].rows() << "x" << data[s].block[q].cols() << endl;
//				cout << Vrhs.data[s].block[q].rows() << "x" << Vrhs.data[s].block[q].cols() << endl;
//				cout << endl;
				
				data[s].block[q] += Vrhs.data[s].block[q];
			}
		}
	}
	return *this;
}

template<typename Symmetry, typename Scalar_>
PivotVector<Symmetry,Scalar_>& PivotVector<Symmetry,Scalar_>::
operator-= (const PivotVector<Symmetry,Scalar_> &Vrhs)
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
	
	#ifdef DMRG_PIVOTVECTOR_PARALLELIZE
	#pragma omp parallel for schedule(dynamic)
	#endif
	for (size_t s=0; s<data.size(); s++)
	for (size_t q=0; q<data[s].dim; ++q)
	{
//		cout << "s=" << s << ", q=" << q << endl;
//		cout << "-= 1=" << data[s].in[q] << ", " << data[s].out[q] << endl;
//		cout << "-= 2=" << Vrhs.data[s].in[q] << ", " << Vrhs.data[s].out[q] << endl;
//		print_size(data[s].block[q],"data[s].block[q]");
//		print_size(Vrhs.data[s].block[q],"Vrhs.data[s].block[q]");
		
//		qarray2<Symmetry::Nq> quple = {data[s].in[q], data[s].out[q]};
//		auto it = Vrhs.data[s].dict.find(quple);
//		if (it != Vrhs.data[s].dict.end())
		{
//			cout << "-=" << endl;
//			print_size(data[s].block[q],"data[s].block[q]");
//			print_size(Vrhs.data[s].block[it->second],"Vrhs.data[s].block[it->second]");
//			cout << endl;
			
			if (data[s].block[q].size() == 0)
			{
				data[s].block[q] = -Vrhs.data[s].block[q];
			}
			else if (data[s].block[q].size() != 0 and Vrhs.data[s].block[q].size() != 0)
			{
				data[s].block[q] -= Vrhs.data[s].block[q];
			}
		}
	}
	return *this;
}

template<typename Symmetry, typename Scalar_>
template<typename OtherScalar>
PivotVector<Symmetry,Scalar_>& PivotVector<Symmetry,Scalar_>::
operator*= (const OtherScalar &alpha)
{
	#ifdef DMRG_PIVOTVECTOR_PARALLELIZE
	#pragma omp parallel for schedule(dynamic)
	#endif
	for (size_t s=0; s<data.size(); ++s)
	for (size_t q=0; q<data[s].dim; ++q)
	{
		data[s].block[q] *= alpha;
	}
	return *this;
}

template<typename Symmetry, typename Scalar_>
template<typename OtherScalar>
PivotVector<Symmetry,Scalar_>& PivotVector<Symmetry,Scalar_>::
operator/= (const OtherScalar &alpha)
{
	#ifdef DMRG_PIVOTVECTOR_PARALLELIZE
	#pragma omp parallel for schedule(dynamic)
	#endif
	for (size_t s=0; s<data.size(); ++s)
	for (size_t q=0; q<data[s].dim; ++q)
	{
		data[s].block[q] /= alpha;
	}
	return *this;
}

template<typename Symmetry, typename Scalar_, typename OtherScalar>
PivotVector<Symmetry,Scalar_> operator* (const OtherScalar &alpha, PivotVector<Symmetry,Scalar_> V)
{
	return V *= alpha;
}

template<typename Symmetry, typename Scalar_, typename OtherScalar>
PivotVector<Symmetry,Scalar_> operator* (PivotVector<Symmetry,Scalar_> V, const OtherScalar &alpha)
{
	return V *= alpha;
}

template<typename Symmetry, typename Scalar_, typename OtherScalar>
PivotVector<Symmetry,Scalar_> operator/ (PivotVector<Symmetry,Scalar_> V, const OtherScalar &alpha)
{
	return V /= alpha;
}

template<typename Symmetry, typename Scalar_>
PivotVector<Symmetry,Scalar_> operator+ (const PivotVector<Symmetry,Scalar_> &V1, const PivotVector<Symmetry,Scalar_> &V2)
{
	PivotVector<Symmetry,Scalar_> Vout = V1;
	Vout += V2;
	return Vout;
}

template<typename Symmetry, typename Scalar_>
PivotVector<Symmetry,Scalar_> operator- (const PivotVector<Symmetry,Scalar_> &V1, const PivotVector<Symmetry,Scalar_> &V2)
{
	PivotVector<Symmetry,Scalar_> Vout = V1;
	Vout -= V2;
	return Vout;
}

template<typename Symmetry, typename Scalar_, typename OtherScalar>
void addScale (const OtherScalar alpha, const PivotVector<Symmetry,Scalar_> &Vin, PivotVector<Symmetry,Scalar_> &Vout)
{
	Vout += alpha * Vin;
}

//-----------<dot & vector norms>-----------
template<typename Symmetry, typename Scalar_>
Scalar_ dot (const PivotVector<Symmetry,Scalar_> &V1, const PivotVector<Symmetry,Scalar_> &V2)
{
//	cout << "sizes in dot: " << V1.data.size() << ", " << V2.data.size() << endl;
	Scalar_ res = 0;
	#ifdef DMRG_PIVOTVECTOR_PARALLELIZE
	#pragma omp parallel for schedule(dynamic) reduction(+:res)
	#endif
	for (size_t s=0; s<V2.data.size(); ++s)
	for (size_t q=0; q<V2.data[s].dim; ++q)
	{
//		if (V1.data[s].in[q] != V2.data[s].in[q] or V1.data[s].out[q] != V2.data[s].out[q])
//		{
//			cout << "s=" << s << ", q=" << q << endl;
//			cout << "V1 inout=" << V1.data[s].in[q] << ", " << V1.data[s].out[q] << endl;
//			cout << "V2 inout=" << V2.data[s].in[q] << ", " << V2.data[s].out[q] << endl;
//			print_size(V1.data[s].block[q],"V1.data[s].block[q]");
//			print_size(V2.data[s].block[q],"V2.data[s].block[q]");
//			cout << endl;
//			cout << termcolor::red << "Mismatched blocks in dot(PivotVector)" << termcolor::reset << endl;
//		}
		
		if (V1.data[s].block[q].size() > 0 and 
		    V2.data[s].block[q].size() > 0)
		{
			res += (V1.data[s].block[q].adjoint() * V2.data[s].block[q]).trace() * Symmetry::coeff_dot(V1.data[s].out[q]);
		}
		
//		qarray2<Symmetry::Nq> quple = {V2.data[s].in[q], V2.data[s].out[q]};
//		auto it = V1.data[s].dict.find(quple);
//		if (it != V1.data[s].dict.end())
//		{
////			cout << "dot" << endl;
////			print_size(V1.data[s].block[it->second],"V1.data[s].block[it->second]");
////			print_size(V2.data[s].block[q],"V2.data[s].block[q]");
////			cout << endl;
////			
//			res += (V1.data[s].block[it->second].adjoint() * V2.data[s].block[q]).trace() * Symmetry::coeff_dot(V1.data[s].out[it->second]);
//		}
	}
	return res;
}

template<typename Symmetry, typename Scalar_>
double squaredNorm (const PivotVector<Symmetry,Scalar_> &V)
{
	double res = isReal(dot(V,V));
	return res;
}

template<typename Symmetry, typename Scalar_>
inline double norm (const PivotVector<Symmetry,Scalar_> &V)
{
	return sqrt(squaredNorm(V));
}

template<typename Symmetry, typename Scalar_>
inline void normalize (PivotVector<Symmetry,Scalar_> &V)
{
	V /= norm(V);
}

template<typename Symmetry, typename Scalar_>
inline size_t dim (const PivotVector<Symmetry,Scalar_> &V)
{
	size_t out = 0;
	for (size_t s=0; s<V.data.size(); ++s)
	for (size_t q=0; q<V.data[s].dim; ++q)
	{
		out += V.data[s].block[q].size();
	}
	return out;
}

template<typename Symmetry, typename Scalar_>
double infNorm (const PivotVector<Symmetry,Scalar_> &V1, const PivotVector<Symmetry,Scalar_> &V2)
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

template<typename Symmetry, typename Scalar_>
void swap (PivotVector<Symmetry,Scalar_> &V1, PivotVector<Symmetry,Scalar_> &V2)
{
//	for (size_t s=0; s<V1.data.size(); ++s)
//	{
//		V1.data[s].block.swap(V2.data[s].block);
//	}
	swap(V1.data, V2.data);
}

template<typename Symmetry, typename Scalar_>
void setZero (PivotVector<Symmetry,Scalar_> &V)
{
	for (size_t s=0; s<V.data.size(); ++s)
	for (size_t q=0; q<V.data[s].dim; ++q)
	{
		V.data[s].block[q].setZero();
	}
};

#include "RandomVector.h"

template<typename Symmetry, typename Scalar_>
struct GaussianRandomVector<PivotVector<Symmetry,Scalar_>,Scalar_>
{
	static void fill (size_t N, PivotVector<Symmetry,Scalar_> &Vout)
	{
		for (size_t s=0; s<Vout.data.size(); ++s)
		for (size_t q=0; q<Vout.data[s].dim; ++q)
		for (size_t a1=0; a1<Vout.data[s].block[q].rows(); ++a1)
		for (size_t a2=0; a2<Vout.data[s].block[q].cols(); ++a2)
		{
			Vout.data[s].block[q](a1,a2) = threadSafeRandUniform<Scalar_>(-1.,1.);
		}
		normalize(Vout);
	}
};

#endif

#ifndef STRAWBERRY_QARRAY
#define STRAWBERRY_QARRAY

#include <iostream>
#include <algorithm>
#include <array>
#include <set>

#include "NestedLoopIterator.h"

/**Array of quantum numbers corresponding to Abelian symmetries.
Just a thin wrapper over \p std::array<int,Nq> with some bracket operators, boolean functions, coefficient-wise arithmetics and output to with a streaming operator.
@describe_Nq
@note special typedefs not caught by Doxygen: \n 
\p qarray2<Nq> = \p std::array<qarray<Nq>,2> : pair of quantum numbers in a two-legged tensor Biped\n
\p qarray3<Nq> = \p std::array<qarray<Nq>,3> : triplet of quantum numbers in a three-legged tensor \p Tripod (see Multipede)\n
\p qarray4<Nq> = \p std::array<qarray<Nq>,4> : quadruplet of quantum numbers in a four-legged tensor \p Quadruped (see Multipede)*/
template<size_t Nq>
struct qarray
{
	/**Constructs with zeros.*/
	qarray() {for (size_t q=0; q<Nq; ++q) {data[q] = 0;}};
	/**Constructs with an \p initializer_list.*/
	qarray (std::initializer_list<int> a) {copy(a.begin(), a.end(), data.data());}
	
	std::array<int,Nq> data;
	
	int &operator[] (size_t i) {return data[i];}
	int &operator() (size_t i) {return data[i];}
	const int &operator[] (size_t i) const {return data[i];}
	const int &operator() (size_t i) const {return data[i];}
};

template<size_t Nq> using qarray2 = std::array<qarray<Nq>,2>;
template<size_t Nq> using qarray3 = std::array<qarray<Nq>,3>;
template<size_t Nq> using qarray4 = std::array<qarray<Nq>,4>;

// "<" and "==" need to be outside of class for map & set, don't ask me why
template<size_t Nq> bool operator== (const qarray<Nq>& lhs, const qarray<Nq>& rhs) {return lhs.data == rhs.data;}
template<size_t Nq> bool operator!= (const qarray<Nq>& lhs, const qarray<Nq>& rhs) {return lhs.data != rhs.data;}
template<size_t Nq> bool operator<= (const qarray<Nq>& lhs, const qarray<Nq>& rhs) {return lhs.data <= rhs.data;}
template<size_t Nq> bool operator>= (const qarray<Nq>& lhs, const qarray<Nq>& rhs) {return lhs.data >= rhs.data;}
template<size_t Nq> bool operator<  (const qarray<Nq>& lhs, const qarray<Nq>& rhs) {return lhs.data <  rhs.data;}
template<size_t Nq> bool operator>  (const qarray<Nq>& lhs, const qarray<Nq>& rhs) {return lhs.data >  rhs.data;}

/**Adds two qarrays coefficient-wise.*/
template<size_t Nq>
qarray<Nq> operator+ (const qarray<Nq> &a1, const qarray<Nq> &a2)
{
	qarray<Nq> aout;
	transform(a1.data.begin(),a1.data.end(), a2.data.begin(), aout.data.begin(), plus<int>());
	return aout;
}

/**Subtracts two qarrays coefficient-wise.*/
template<size_t Nq>
qarray<Nq> operator- (const qarray<Nq> &a1, const qarray<Nq> &a2)
{
	qarray<Nq> aout;
	transform(a1.data.begin(),a1.data.end(), a2.data.begin(), aout.data.begin(), minus<int>());
	return aout;
}

/**Multiplies a qarray with a factor coefficient-wise.*/
template<size_t Nq>
qarray<Nq> operator* (const size_t &alpha, const qarray<Nq> &a)
{
	qarray<Nq> aout;
	for (size_t q=0; q<Nq; ++q)
	{
		aout[q] = alpha*a[q];
	}
	return aout;
}

template<size_t Nq>
ostream& operator<< (ostream& os, const qarray<Nq> &a)
{
	os << "(";
	for (size_t q=0; q<Nq; ++q)
	{
		os << a[q];
		if (q!=Nq-1) {os << ",";}
	}
	os << ")";
	return os;
}

/**Constructs the vacuum (all quantum numbers equal to zero).*/
template<size_t Nq>
qarray<Nq> qvacuum()
{
	qarray<Nq> aout;
	for (size_t q=0; q<Nq; ++q) {aout[q] = 0;}
	return aout;
}

/**Iterator over all quantum numbers on a subchain.
Needed in creating the subspaces when constructing an MpsQ.*/
template<size_t D, size_t Nq>
class qarrayIterator
{
public:
	
	/**\param qloc_input : local basis
	\param L_input : length of subchain*/
	qarrayIterator (std::array<qarray<Nq>,D> qloc_input, size_t L_input)
	:qloc(qloc_input), N_sites(L_input)
	{
		// determine dq
		for (size_t q=0; q<Nq; ++q)
		{
			set<int> qset;
			for (size_t s=0; s<D; ++s) {qset.insert(qloc[s][q]);}
			set<int> diffqset;
			for (auto it=qset.begin(); it!=qset.end(); ++it)
			{
				int prev;
				if (it==qset.begin()) {prev=*it;}
				else
				{
					diffqset.insert(*it-prev);
					prev = *it;
				}
			}
			
			assert(diffqset.size()==1 and 
			       "Unable to understand quantum number increments!");
			dq[q] = *diffqset.begin();
		}
		
		// determine qmin, qmax
		qmin = N_sites * (*min_element(qloc.begin(),qloc.end()));
		qmax = N_sites * (*max_element(qloc.begin(),qloc.end()));
		
		// setup NestedLoopIterator
		vector<size_t> ranges(Nq);
		for (size_t q=0; q<Nq; ++q)
		{
			ranges[q] = (qmax[q]-qmin[q])/dq[q]+1;
		}
		Nelly = NestedLoopIterator(Nq,ranges);
	};
	
	/**Returns the value of the quantum number.*/
	qarray<Nq> operator*() {return value;}
	
	qarrayIterator& operator= (const qarray<Nq> a) {value=a;}
	bool operator!= (const qarray<Nq> a) {return value!=a;}
	bool operator<= (const qarray<Nq> a) {return value<=a;}
	bool operator<  (const qarray<Nq> a) {return value< a;}
	
	qarray<Nq> begin()
	{
		Nelly = Nelly.begin();
		return qmin;
	}
	
	qarray<Nq> end()
	{
		qarray<Nq> qout = qmax;
		qout[0] += dq[0];
		return qout;
	}
	
	void operator++()
	{
		++Nelly;
		if (Nelly==Nelly.end())
		{
			value = qmax;
			value[0] += dq[0];
		}
		else
		{
			value = qmin;
			for (size_t q=0; q<Nq; ++q)
			{
				value[q] += Nelly(q)*dq[q];
			}
		}
	}
	
private:
	
	qarray<Nq> value;
	
	NestedLoopIterator Nelly;
	
	qarray<Nq> qmin;
	qarray<Nq> qmax;
	
	std::array<qarray<Nq>,D> qloc;
	qarray<Nq> dq;
	size_t N_sites;
};

#endif

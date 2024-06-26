#ifndef STRAWBERRY_QARRAY
#define STRAWBERRY_QARRAY

/// \cond
#include <iostream>
#include <algorithm>
#include <array>
#include <set>
#include <limits>
/// \endcond

#include "NestedLoopIterator.h"
#include "JoinArray.h"

/**
 * Array of quantum numbers corresponding to Abelian or non Abelian symmetries.
 * Just a thin wrapper over \p std::array<int,Nq> with some bracket operators, boolean functions, coefficient-wise arithmetics and output to with a streaming operator.
 * \describe_Nq
 * \note special typedefs not caught by Doxygen: \n 
 * \p qarray2<Nq> = \p std::array<qarray<Nq>,2> : pair of quantum numbers in a two-legged tensor Biped\n
 * \p qarray3<Nq> = \p std::array<qarray<Nq>,3> : triplet of quantum numbers in a three-legged tensor \p Tripod (see Multipede)\n
 * \p qarray4<Nq> = \p std::array<qarray<Nq>,4> : quadruplet of quantum numbers in a four-legged tensor \p Quadruped (see Multipede)
*/
template<size_t Nq>
struct qarray
{
	/**Constructs with zeros.*/
	constexpr qarray() {for (size_t q=0; q<Nq; ++q) {data[q] = 0;}};
	
	/**Constructs with an input array.*/
	constexpr qarray(const std::array<int,Nq> &in) : data(in) {};
	
	/**Constructs with an \p initializer_list.*/
	qarray (std::initializer_list<int> a) {copy(a.begin(), a.end(), data.data());}
	
	std::array<int,Nq> data;
	
	int &operator[] (size_t i) {return data[i];}
	int &operator() (size_t i) {return data[i];}
	const int &operator[] (size_t i) const {return data[i];}
	const int &operator() (size_t i) const {return data[i];}
	
	int distance(const qarray<Nq>& other)
	{
		array<int,Nq> dists;
		for(size_t i=0; i<Nq; i++) { dists[i] = abs(this->data[i] - other[i]); }
		return *std::max_element(std::begin(dists), std::end(dists));
	}
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
	transform(a1.data.begin(),a1.data.end(), a2.data.begin(), aout.data.begin(), std::plus<int>());
	return aout;
}

/**Subtracts two qarrays coefficient-wise.*/
template<size_t Nq>
qarray<Nq> operator- (const qarray<Nq> &a1, const qarray<Nq> &a2)
{
	qarray<Nq> aout;
	transform(a1.data.begin(),a1.data.end(), a2.data.begin(), aout.data.begin(), std::minus<int>());
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

/**Join (concatenate) two qarrays.*/
template<std::size_t Nq1, std::size_t Nq2>
constexpr qarray<Nq1+Nq2> join(qarray<Nq1> rhs, qarray<Nq2> lhs)
{
	auto new_array = thirdparty::join(rhs.data, lhs.data);
	qarray<Nq1+Nq2> out;
	out.data = new_array;
	return out;
}

template<std::size_t Nq1, std::size_t Nq2, std::size_t Nql>
std::pair<qarray<Nq1>, qarray<Nq2> > disjoin(const qarray<Nql>& large_arr)
{
	qarray<Nq1> lhs;
	std::copy(large_arr.data.begin(),large_arr.data.begin()+Nq1,lhs.data.begin());
	qarray<Nq2> rhs;
	std::copy(large_arr.data.begin()+Nq1,large_arr.data.end(),rhs.data.begin());
	return make_pair(lhs,rhs);
}

template<size_t Nq>
std::ostream& operator<< (std::ostream& os, const qarray<Nq> &a)
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

///**Constructs the vacuum (all quantum numbers equal to zero).*/
//template<size_t Nq>
//qarray<Nq> qvacuum()
//{
//	qarray<Nq> aout;
//	for (size_t q=0; q<Nq; ++q) {aout[q] = 0;}
//	return aout;
//}

template<size_t Nq>
qarray<Nq> qplusinf()
{
	qarray<Nq> aout;
	for (size_t q=0; q<Nq; ++q) {aout[q] = numeric_limits<int>::max();}
	return aout;
}

template<size_t Nq>
qarray<Nq> qminusinf()
{
	qarray<Nq> aout;
	for (size_t q=0; q<Nq; ++q) {aout[q] = -numeric_limits<int>::max();}
	return aout;
}

#endif

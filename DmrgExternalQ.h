#ifndef STRAWBERRY_DMRGEXTERNALQ
#define STRAWBERRY_DMRGEXTERNALQ

#include <sstream>
#include <array>
#include <boost/functional/hash.hpp>

#include "qarray.h"

/**Default format for quantum number output: Print the integer as is.*/
template<size_t Nq>
string noFormat (qarray<Nq> qnum)
{
	stringstream ss;
	ss << qnum;
	return ss.str();
}

/**Makes a default label for conserved quantum numbers: "Q1", "Q2", "Q3"...
Is realized by a function to preserve the sanity of the programmer since default template-sized arguments seem to be tricky.*/
template<size_t Nq>
constexpr std::array<string,Nq> defaultQlabel()
{
	std::array<string,Nq> out;
	for (size_t q=0; q<Nq; ++q)
	{
		stringstream ss;
		ss << "Q" << q+1;
		out[q] = ss.str();
	}
	return out;
}

namespace std
{
/**Hashes an array of quantum numbers using boost's \p hash_combine for the dictionaries in Biped, Multipede.*/
template<size_t Nq, size_t Nlegs>
struct hash<std::array<qarray<Nq>,Nlegs> >
{
	inline size_t operator()(const std::array<qarray<Nq>,Nlegs> &ix) const
	{
		size_t seed = 0;
		for (size_t leg=0; leg<Nlegs; ++leg)
		for (size_t q=0; q<Nq; ++q)
		{
			boost::hash_combine(seed, ix[leg][q]);
		}
		return seed;
	}
};

/**Hashes a pair of doubles using boost's \p hash_combine.
Needed for \ref precalc_blockStructure.*/
template<>
struct hash<std::array<size_t,2> >
{
	inline size_t operator()(const std::array<size_t,2> &a) const
	{
		size_t seed = 0;
		boost::hash_combine(seed, a[0]);
		boost::hash_combine(seed, a[1]);
		return seed;
	}
};
}

#endif

#ifndef DICT_Q
#define DICT_Q

#include <utility>
#include <unordered_map>
#include <iostream>

#include "Tuples.h"
#include "qarray.h"

#include "DmrgExternalQ.h"

constexpr std::size_t factorial (std::size_t n)
{
    return n > 0 ? n * factorial( n - 1 ) : 1;
}

constexpr std::size_t binomial (std::size_t n, std::size_t k)
{
	return factorial(n)/(factorial(k)*factorial(n-k));
}

/**Class for quantum number dictionaries.
\describe_Nlegs
\describe_Nq*/
template<std::size_t Nlegs, typename Symmetry>
class DictQ
{
	typedef Eigen::Index Index;
	typedef typename Symmetry::qType qType;
	template<std::size_t i> using map = std::unordered_multimap<std::array<qType,i>, std::size_t>;
	template<std::size_t i> using array_map = std::array<map<i>,binomial(Nlegs,i)>;
	using quantum_block = std::array<qType,Nlegs>; 
public:
	DictQ () {};

	template<std::size_t i> inline typename map<i>::const_iterator
	find( const std::array<qType,i> &qvec, const std::array<Index,i> &legs) const;
	template<std::size_t i> inline std::pair<typename map<i>::const_iterator,typename map<i>::const_iterator>
	equal_range( const std::array<qType,i> &qvec, const std::array<Index,i> &legs) const;
	template<std::size_t i> std::size_t count( const std::array<qType,i> &qvec, const std::array<Index,i> &legs );
	
	template<std::size_t i> inline typename map<i>::const_iterator begin( const std::array<Index,i> &legs) const;
	template<std::size_t i> inline typename map<i>::const_iterator end( const std::array<Index,i> &legs) const;

	inline void insert( const std::pair<quantum_block,std::size_t> &qvec )
		{
			insert__(qvec, std::make_index_sequence<Nlegs>());
		}
	inline void clear()
		{
			clear__( std::make_index_sequence<Nlegs>());			
		}

private:
	template<std::size_t... legs>
	inline void insert__(const std::pair<quantum_block,std::size_t> &qvec, std::index_sequence<legs...>)
		{
			( insertBase(subArray<legs>(qvec)), ... );
			insertBase(subArray<Nlegs>(qvec));
		}

	template<std::size_t... legs>
	inline void clear__( std::index_sequence<legs...>)
		{
			( clearBase<legs>(), ... );
			clearBase<Nlegs>();
		}

	template<size_t i> void insertBase( const std::array<std::pair<std::array<qType,i>, std::size_t>,binomial(Nlegs,i)> &qvec );

	template<size_t i> void clearBase();

	template<size_t i> std::array<std::pair<std::array<qType,i>,std::size_t>,binomial(Nlegs,i)>
	subArray( const std::pair<std::array<qType,Nlegs>,std::size_t> &qvec );

	template<class U> struct dictType;
	
	template<std::size_t... legs>
	struct dictType<std::index_sequence<legs...> >
	{
		template<std::size_t i> using map__ = std::unordered_multimap<std::array<qType,i>, std::size_t>;
		template<std::size_t i> using array_map__ = std::array<map__<i>,binomial(Nlegs,i)>;
		using type = std::tuple<array_map__<legs>...,array_map__<Nlegs> >;
	};

	typename dictType<std::make_index_sequence<Nlegs> >::type data;
};

template<std::size_t Nlegs, typename Symmetry>
template<std::size_t i>
std::pair<typename std::unordered_multimap<std::array<typename Symmetry::qType,i>, std::size_t>::const_iterator,
		  typename std::unordered_multimap<std::array<typename Symmetry::qType,i>, std::size_t>::const_iterator> DictQ<Nlegs,Symmetry>::
equal_range( const std::array<qType,i> &qvec, const std::array<Index,i> &legs ) const
{
	std::array<Index,i> sorted_legs=legs;
	std::array<std::size_t,i> ind;
	std::iota(ind.begin(),ind.end(),0);
	std::sort(ind.begin(),ind.end(),[&](int i1, int i2) { return legs[i1] < legs[i2]; });
	std::array<qType,i> sorted_qvec=qvec;
	std::sort(sorted_legs.begin(),sorted_legs.end());
	for (std::size_t j=0; j<i; j++) {sorted_qvec[j] = qvec[ind[j]]; }
	return std::get<i>(data)[Tuples<Nlegs,i>::getNumber(sorted_legs)].equal_range(sorted_qvec);
}

template<std::size_t Nlegs, typename Symmetry>
template<std::size_t i>
typename std::unordered_multimap<std::array<typename Symmetry::qType,i>, std::size_t>::const_iterator DictQ<Nlegs,Symmetry>::
find( const std::array<qType,i> &qvec, const std::array<Index,i> &legs ) const
{
	return std::get<i>(data)[Tuples<Nlegs,i>::getNumber(legs)].find(qvec);
}

template<std::size_t Nlegs, typename Symmetry>
template<std::size_t i>
std::size_t DictQ<Nlegs,Symmetry>::
count( const std::array<qType,i> &qvec, const std::array<Index,i> &legs )
{
	return std::get<i>(data)[Tuples<Nlegs,i>::getNumber(legs)].count(qvec);
}

template<std::size_t Nlegs, typename Symmetry>
template<std::size_t i>
typename std::unordered_multimap<std::array<typename Symmetry::qType,i>, std::size_t>::const_iterator DictQ<Nlegs,Symmetry>::
begin( const std::array<Index,i> &legs ) const
{
	return std::get<i>(data)[Tuples<Nlegs,i>::getNumber(legs)].begin();
}

template<std::size_t Nlegs, typename Symmetry>
template<std::size_t i>
typename std::unordered_multimap<std::array<typename Symmetry::qType,i>, std::size_t>::const_iterator DictQ<Nlegs,Symmetry>::
end( const std::array<Index,i> &legs ) const
{
	return std::get<i>(data)[Tuples<Nlegs,i>::getNumber(legs)].end();
}

template<std::size_t Nlegs, typename Symmetry>
template<std::size_t i>
void DictQ<Nlegs,Symmetry>::
insertBase ( const std::array<std::pair<std::array<qType,i>, std::size_t>,binomial(Nlegs,i)> &qvec )
{
	for (std::size_t j=0; j<binomial(Nlegs,i); j++)
	{
		std::get<i>(data)[j].insert(qvec[j]);
	}
}

template<std::size_t Nlegs, typename Symmetry>
template<std::size_t i>
void DictQ<Nlegs,Symmetry>::
clearBase ()
{
	for (std::size_t j=0; j<binomial(Nlegs,i); j++)
	{
		std::get<i>(data)[j].clear();
	}
}

template<std::size_t Nlegs, typename Symmetry>
template<std::size_t i>
std::array<std::pair<std::array<typename Symmetry::qType,i>,std::size_t>,binomial(Nlegs,i)> DictQ<Nlegs,Symmetry>::
subArray ( const std::pair<std::array<qType,Nlegs>,std::size_t> &qvec )
{
	std::array<std::pair<std::array<qType,i>,std::size_t>,binomial(Nlegs,i)> out;
	for (std::size_t j=0; j<binomial(Nlegs,i); j++)
	{
		std::array<qType,i> tmp;
		std::size_t counter=0;
		for ( std::size_t k=0; k<Nlegs; k++ )
		{
			if (Tuples<Nlegs,i>::isPresent(j,k)) { tmp[counter] = std::get<0>(qvec)[k]; counter++; }
		}
		out[j] = std::make_pair(tmp,std::get<1>(qvec));		
	}
	return out;
}

#endif

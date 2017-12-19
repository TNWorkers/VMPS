#ifndef STRAWBERRY_DMRGEXTERNALQ
#define STRAWBERRY_DMRGEXTERNALQ

#include <sstream>
#include <array>
#include <boost/functional/hash.hpp>
#include <boost/rational.hpp>
typedef boost::rational<int> frac;

#include "qarray.h"

string print_frac_nice (frac r)
{
	stringstream ss;
	if (r.denominator() == 1) {ss << r.numerator();}
	else {ss << r;}
	return ss.str();
}

/**Default format for quantum number output: Print the integer as is.*/
template<size_t Nq>
string noFormat (qarray<Nq> qnum)
{
	stringstream ss;
	ss << qnum;
	return ss.str();
}

/**Makes half-integers in the output.*/
string halve (qarray<1> qnum)
{
	stringstream ss;
	ss << "(";
	boost::rational<int> m = boost::rational<int>(qnum[0],2);
	if      (m.numerator()   == 0) {ss << 0;}
	else if (m.denominator() == 1) {ss << m.numerator();}
	else {ss << m;}
	ss << ")";
	return ss.str();
}

/**Calculates the total spin \p S from the degeneracy \p D for a label.*/
string SfromD (qarray<1> qnum)
{
	stringstream ss;
	ss << "(";
	ss << (qnum[0]-1)/2;
	ss << ")";
	return ss.str();
}

string SfromD_noFormat (qarray<2> qnum)
{
	stringstream ss;
	ss << "(";
	ss << (qnum[0]-1)/2 << ",";
	ss << qnum[1];
	ss << ")";
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
	
/**Hashes one qarray using boost's \p hash_combine.
Needed in class \ref Qbasis.*/
template<size_t Nq>
struct hash<qarray<Nq> >
{
	inline size_t operator()(const qarray<Nq> &ix) const
	{
		size_t seed = 0;
		for (size_t q=0; q<Nq; ++q)
		{
			boost::hash_combine(seed, ix[q]);
		}
		return seed;
	}
};
}

/**Function to realize staggered fields.*/
inline double stagger (int i)
{
	return pow(-1.,i);
}

template<typename MatrixTypeA, typename MatrixTypeB>
size_t mult_cost (const MatrixTypeA &A, const MatrixTypeB &B)
{
	return A.rows()*A.cols()*B.cols();
}

template<typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC, typename MatrixTypeD>
vector<size_t> mult_cost (const MatrixTypeA &A, const MatrixTypeB &B, const MatrixTypeC &C, const MatrixTypeD &D)
{
	vector<size_t> out(5);
	// (AB)(CD)
	out[0] = mult_cost(A,B) + mult_cost(C,D) + A.rows()*B.cols()*C.cols();
	
	// ((AB)C)D
	out[1] = mult_cost(A,B) + A.rows()*C.rows()*C.cols() + A.rows()*D.rows()*D.cols();
	
	// (A(BC))D
	out[2] = mult_cost(B,C) + A.rows()*A.cols()*C.cols() + A.rows()*D.rows()*D.cols();
	
	// A((BC)D)
	out[3] = mult_cost(B,C) + B.rows()*D.rows()*D.cols() + A.rows()*A.cols()*D.cols();
	
	// A(B(CD))
	out[4] = mult_cost(C,D) + B.rows()*B.cols()*D.cols() + A.rows()*A.cols()*D.cols();
	return out;
}

template<typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC>
vector<size_t> mult_cost (const MatrixTypeA &A, const MatrixTypeB &B, const MatrixTypeC &C)
{
	vector<size_t> out(2);
	// (AB)C
	out[0] = mult_cost(A,B) + A.rows()*C.rows()*C.cols();
	
	// A(BC)
	out[1] = mult_cost(B,C) + A.rows()*A.cols()*C.cols();
	
	return out;
}

template<typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC, typename MatrixTypeD, typename MatrixTypeR, typename Scalar>
void optimal_multiply (Scalar alpha, const MatrixTypeA &A, const MatrixTypeB &B, const MatrixTypeC &C, const MatrixTypeD &D, 
                       MatrixTypeR &result)
{
	vector<size_t> cost(5);
	cost = mult_cost(A,B,C,D);
	size_t opt_mult = min_element(cost.begin(),cost.end())- cost.begin();
	
	if (opt_mult == 0)
	{
		MatrixTypeR Mtmp1 = A * B;
		MatrixTypeR Mtmp2 = C * D;
		result = alpha * Mtmp1 * Mtmp2;
	}
	else if (opt_mult == 1)
	{
		MatrixTypeR Mtmp = A * B;
		Mtmp = Mtmp * C;
		result = alpha * Mtmp * D;
	}
	else if (opt_mult == 2)
	{
		MatrixTypeR Mtmp = B * C;
		Mtmp = A * Mtmp;
		result = alpha * Mtmp * D;
	}
	else if (opt_mult == 3)
	{
		MatrixTypeR Mtmp = B * C;
		Mtmp = Mtmp * D;
		result = alpha * A * Mtmp;
	}
	else if (opt_mult == 4)
	{
		MatrixTypeR Mtmp = C * D;
		Mtmp = B * Mtmp;
		result = alpha * A * Mtmp;
	}
}

template<typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC, typename MatrixTypeR, typename Scalar>
void optimal_multiply (Scalar alpha, const MatrixTypeA &A, const MatrixTypeB &B, const MatrixTypeC &C, MatrixTypeR &result)
{
	vector<size_t> cost(2);
	cost = mult_cost(A,B,C);
	size_t opt_mult = min_element(cost.begin(),cost.end())- cost.begin();
	
	if (opt_mult == 0)
	{
		MatrixTypeR Mtmp = A * B;
		result.noalias() = alpha * Mtmp * C;
	}
	else if (opt_mult == 1)
	{
		MatrixTypeR Mtmp = B * C;
		result.noalias() = alpha * A * Mtmp;
	}
}

#endif

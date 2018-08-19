#ifndef STRAWBERRY_DMRGEXTERNALQ
#define STRAWBERRY_DMRGEXTERNALQ

#include <sstream>
#include <array>
#include <boost/functional/hash.hpp>
#include <boost/rational.hpp>
typedef boost::rational<int> frac;

#include "symmetry/qarray.h"
// #include "DmrgTypedefs.h"

/**Prints a boost fraction in such a way, that a "1" in the denominator is omitted.*/
std::string print_frac_nice (frac r)
{
	std::stringstream ss;
	if (r.denominator()==1) {ss << r.numerator();}
	else {ss << r;}
	return ss.str();
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

/**Hashes a tuple of indices and quantum numbers using boost's \p hash_combine for the Mpo V-Matrices.*/
template<size_t Nq>
struct hash<std::tuple<size_t,size_t,size_t,qarray<Nq>,qarray<Nq> > >
{
	inline size_t operator()(const std::tuple<size_t,size_t,size_t,qarray<Nq>,qarray<Nq> > &ix) const
	{
		size_t seed = 0;
		boost::hash_combine(seed, get<0>(ix));
		boost::hash_combine(seed, get<1>(ix));
		boost::hash_combine(seed, get<2>(ix));
		for (size_t q=0; q<Nq; ++q) { boost::hash_combine(seed, get<3>(ix)[q]); }
		for (size_t q=0; q<Nq; ++q) { boost::hash_combine(seed, get<4>(ix)[q]); }
		return seed;
	}
};

/**
 * Hashes a pair of doubles using boost's \p hash_combine.
 * Needed for \ref precalc_blockStructure.
 */
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
	
/**
 * Hashes one qarray using boost's \p hash_combine.
 * Needed in class \ref Qbasis.
 */
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

/**Cost to multiply 2 matrices.*/
template<typename MatrixTypeA, typename MatrixTypeB>
size_t mult_cost (const MatrixTypeA &A, const MatrixTypeB &B)
{
	return A.rows()*A.cols()*B.cols();
}

/**Cost to multiply 3 matrices in 2 possible ways.*/
template<typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC>
std::vector<size_t> mult_cost (const MatrixTypeA &A, const MatrixTypeB &B, const MatrixTypeC &C)
{
	std::vector<size_t> out(2);
	// (AB)C
	out[0] = mult_cost(A,B) + A.rows()*C.rows()*C.cols();
	
	// A(BC)
	out[1] = mult_cost(B,C) + A.rows()*A.cols()*C.cols();
	
	return out;
}

/**Cost to multiply 4 matrices in 5 possible ways.*/
template<typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC, typename MatrixTypeD>
std::vector<size_t> mult_cost (const MatrixTypeA &A, const MatrixTypeB &B, const MatrixTypeC &C, const MatrixTypeD &D)
{
	std::vector<size_t> out(5);
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

template<typename MatrixType>
inline void print_size (const MatrixType &M, string label)
{
	std::cout << label << ": " << M.rows() << "x" << M.cols() << std::endl;
}

/**Multiplies 3 matrices by using the optimal order of operations.*/
template<typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC, typename MatrixTypeR, typename Scalar>
void optimal_multiply (Scalar alpha, const MatrixTypeA &A, const MatrixTypeB &B, const MatrixTypeC &C, MatrixTypeR &result, bool DEBUG=false)
{
	if (DEBUG)
	{
		print_size(A,"A");
		print_size(B,"B");
		print_size(C,"C");
		std::cout << endl;
	}
	
	std::vector<size_t> cost(2);
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

/**Multiplies 4 matrices by using the optimal order of operations.*/
template<typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC, typename MatrixTypeD, typename MatrixTypeR, typename Scalar>
void optimal_multiply (Scalar alpha, const MatrixTypeA &A, const MatrixTypeB &B, const MatrixTypeC &C, const MatrixTypeD &D, MatrixTypeR &result, 
                       bool DEBUG=false)
{
	if (DEBUG)
	{
		print_size(A,"A");
		print_size(B,"B");
		print_size(C,"C");
		print_size(D,"D");
		std::cout << endl;
	}
	
	std::vector<size_t> cost(5);
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

/**Function to realize staggered fields.*/
inline double stagger (int i)
{
	return pow(-1.,i);
}

/**Dummy weight function for sums of local operators.*/
template<typename Scalar>
Scalar localSumTrivial (int i)
{
	return 1.;
}

// template<typename Symmetry>
// void transform_base (vector<vector<qarray<Symmetry::Nq> > > &qloc, qarray<Symmetry::Nq> Qtot, bool PRINT = false)
// {
// 	if (Qtot != Symmetry::qvacuum())
// 	{
// 		for (size_t l=0; l<qloc.size(); ++l)
// 		for (size_t i=0; i<qloc[l].size(); ++i)
// 		for (size_t q=0; q<Symmetry::Nq; ++q)
// 		{
// 			if (Symmetry::kind()[q] != Sym::KIND::S and Symmetry::kind()[q] != Sym::KIND::T) //Do not transform the base for non Abelian symmetries
// 			{
// 				qloc[l][i][q] = qloc[l][i][q] * static_cast<int>(qloc.size()) - Qtot[q];
// 			}
// 		}
		
// 		if (PRINT)
// 		{
// 			lout << "transformed base:" << endl;
// 			for (size_t l=0; l<qloc.size(); ++l)
// 			{
// 				lout << "l=" << l << endl;
// 				for (size_t i=0; i<qloc[l].size(); ++i)
// 				{
// 					cout << "qloc: " << qloc[l][i] << endl;
// 				}
// 			}
// 		}
// 	}
// };

#endif
